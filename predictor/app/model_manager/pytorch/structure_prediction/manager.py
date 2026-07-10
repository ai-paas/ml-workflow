import json

from app.model_manager.base import BaseModelManager
from kserve import logging


class StructurePredictionModelManager(BaseModelManager):
    """단백질 구조예측(task=protein-structure-prediction) 추론 매니저. ESMFold2 를 서빙한다.

    fill-mask/protein-classification 과 달리 AutoModel/transformers pipeline 경로가 아니라, vendored
    ESMFold2Model 을 직접 로드하고 model.infer_protein(seq, ...) 로 전원자 구조를 예측한다.

    ESMFold2 는 folding trunk(~1.3GB) 단독이 아니라 언어모델 백본 ESMC-6B(~25GB fp32)를 함께 로드해야
    추론이 성립한다(서열 임베딩이 구조 신호의 핵심). ESMFold2 원본 from_pretrained 는 config.esmc_id
    (=biohub/ESMC-6B)를 HF Hub 에서 런타임 다운로드하는데, 서빙에서는 이를 피하고 MLflow 에 등록된
    ESMC-6B(base_run_id/base_model_uri)를 로컬로 받아 load_esmc 로 얹는다(HF 무인증 25GB 다운로드 회피).
    base 위치가 없으면 원본 동작(config.esmc_id, HF)으로 폴백한다.

    입력 페이로드: {"sequence": "...", "num_loops": 3, "num_sampling_steps": 50}.
    출력: PDB 문자열 + 신뢰도(plddt 평균/ptm/iptm).
    """

    DEFAULT_NUM_LOOPS = 3
    DEFAULT_NUM_SAMPLING_STEPS = 50
    ESMC_PRECISION = "bf16"

    def __init__(self):
        super().__init__()
        self.model = None
        self.device = None
        # 백본 ESMC-6B 의 MLflow 위치(InferenceModel 이 주입). 없으면 config.esmc_id(HF Hub) 폴백.
        self.base_run_id = None
        self.base_model_uri = None

    def load_model(self, model_name: str, run_id: str):
        """MLflow run 아티팩트(ESMFold2 folding trunk)를 받고, 백본 ESMC-6B 를 로컬 로드해 결합한다."""
        import torch

        # ESMFold2/ESMC 는 upstream transformers 에 없어 vendored 구현을 import 해 Auto 레지스트리에 등록한다.
        # (load_esmc 가 ESMCModel 을 사용하므로 esmc 등록도 필요.)
        from app.vendored.esmfold2 import ESMFold2Model

        try:
            import app.vendored.esmc  # noqa: F401
        except Exception as e:
            logging.logger.warning(f"vendored esmc 등록 실패: {e}")

        local_path = self._load_artifacts(run_id, model_name)
        logging.logger.info(f"ESMFold2 folding trunk dir: {local_path}")

        # 백본(ESMC-6B)은 별도로 얹으므로 folding trunk 만 먼저 로드(load_esmc=False → HF 런타임 다운로드 차단).
        self.model = ESMFold2Model.from_pretrained(local_path, load_esmc=False)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # 백본 ESMC-6B: MLflow 등록본(base_run_id/base_model_uri)을 우선 로컬 로드, 없으면 config.esmc_id(HF) 폴백.
        esmc_source = self.model.config.esmc_id
        if self.base_run_id and self.base_model_uri:
            try:
                esmc_source = self._load_artifacts(self.base_run_id, self.base_model_uri)
                logging.logger.info(f"ESMC-6B backbone (MLflow) dir: {esmc_source}")
            except Exception as e:
                logging.logger.warning(f"MLflow ESMC-6B 다운로드 실패, HF Hub({esmc_source}) 폴백: {e}")
        else:
            logging.logger.warning(f"base(ESMC-6B) 위치 미전달 → HF Hub({esmc_source}) 런타임 다운로드로 폴백")

        self.model.load_esmc(esmc_source, precision=self.ESMC_PRECISION)
        self.model.eval()
        self.device = self.model.device
        logging.logger.info("ESMFold2 구조예측 모델(+ESMC-6B 백본) 로드 완료")

    @staticmethod
    def _scalar(value):
        """텐서/스칼라를 평균 float 로 요약(응답용). 실패 시 None."""
        try:
            if hasattr(value, "mean"):
                return float(value.float().mean().item())
            return float(value)
        except Exception:
            return None

    def predict(self, data, device_str: str = "cpu"):
        """단백질 서열을 받아 전원자 3D 구조(PDB)와 신뢰도 지표를 반환.

        반환 형식:
        {
          "predictions": [{"pdb": "<PDB 문자열>", "plddt_mean": float, "ptm": float, "iptm": float}],
          "input_info": {"sequence": "...", "num_loops": int, "num_sampling_steps": int}
        }
        """
        import torch

        if isinstance(data, (bytes, bytearray)):
            data = json.loads(data.decode("utf-8"))
        elif isinstance(data, str):
            data = json.loads(data)

        sequence = (data.get("sequence") or "").strip().upper()
        if not sequence:
            raise ValueError("sequence 는 필수이며 비어있을 수 없습니다.")
        num_loops = int(data.get("num_loops") or self.DEFAULT_NUM_LOOPS)
        num_sampling_steps = int(data.get("num_sampling_steps") or self.DEFAULT_NUM_SAMPLING_STEPS)

        with torch.no_grad():
            output = self.model.infer_protein(sequence, num_loops=num_loops, num_sampling_steps=num_sampling_steps)
        pdb = self.model.output_to_pdb(output)

        return {
            "predictions": [
                {
                    "pdb": pdb,
                    "plddt_mean": self._scalar(output.get("plddt")),
                    "ptm": self._scalar(output.get("ptm")),
                    "iptm": self._scalar(output.get("iptm")),
                }
            ],
            "input_info": {
                "sequence": sequence,
                "num_loops": num_loops,
                "num_sampling_steps": num_sampling_steps,
            },
        }
