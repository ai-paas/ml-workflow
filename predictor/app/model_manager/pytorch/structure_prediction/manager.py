import json

from app.model_manager.base import BaseModelManager
from kserve import logging


class StructurePredictionModelManager(BaseModelManager):
    """단백질 구조예측(task=protein-structure-prediction) 추론 매니저. ESMFold2 를 서빙한다.

    fill-mask/protein-classification 과 달리 AutoModel/transformers pipeline 경로가 아니라, vendored
    ESMFold2Model 을 직접 로드하고 model.infer_protein(seq, ...) 로 전원자 구조를 예측한다.
    (ESMFold2 는 소형(~1.3GB)이라 단일 GPU 로 충분 — device_map/멀티GPU 불필요.)

    입력 페이로드: {"sequence": "...", "num_loops": 3, "num_sampling_steps": 50}.
    출력: PDB 문자열 + 신뢰도(plddt 평균/ptm/iptm).
    """

    DEFAULT_NUM_LOOPS = 3
    DEFAULT_NUM_SAMPLING_STEPS = 50

    def __init__(self):
        super().__init__()
        self.model = None
        self.device = None

    def load_model(self, model_name: str, run_id: str):
        """MLflow run 아티팩트(ESMFold2 가중치)를 내려받아 로드한다."""
        import torch

        # ESMFold2 는 upstream transformers 에 없어 vendored 구현을 import 해 Auto 레지스트리에 등록한다.
        from app.vendored.esmfold2 import ESMFold2Model

        local_path = self._load_artifacts(run_id, model_name)
        logging.logger.info(f"ESMFold2 모델 dir: {local_path}")

        self.model = ESMFold2Model.from_pretrained(local_path)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        self.device = next(self.model.parameters()).device
        logging.logger.info("ESMFold2 구조예측 모델 로드 완료")

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
