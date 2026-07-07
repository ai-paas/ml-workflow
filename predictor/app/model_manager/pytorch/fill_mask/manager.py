import json

from app.model_manager.base import BaseModelManager
from kserve import logging


class FillMaskModelManager(BaseModelManager):
    """Masked-LM(task=fill-mask) 추론 매니저. base BFM(ESM2/ESMC/RNA-FM/MoLFormer 등)을 어댑터 없이
    그대로 서빙한다. protein-classification 과 달리 LoRA 어댑터가 없고 base 모델 자체가 배포 대상이므로,
    MLflow run 아티팩트를 AutoModelForMaskedLM 으로 로드한다.

    입력 페이로드: {"sequence": "...<mask>...", "top_k": N}. 마스크 토큰 자리마다 top-k 후보를 반환한다.
    """

    MAX_LENGTH = 1024
    DEFAULT_TOP_K = 5

    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.device = None

    def load_model(self, model_name: str, run_id: str):
        """MLflow run 아티팩트(base 모델 가중치)를 내려받아 MaskedLM 헤드로 로드.

        RNA-FM 처럼 transformers 표준에 없는 아키텍처는 multimolecule 를 사전 import 해 Auto 레지스트리에
        등록해야 하고(best-effort, ESM2/MoLFormer 에는 무영향), MoLFormer 처럼 config 에 auto_map(커스텀 코드)이
        있는 모델은 trust_remote_code 로 아티팩트에 포함된 modeling/tokenization 코드를 로드한다.
        """
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        # RNA-FM 등 등록형 아키텍처를 Auto 레지스트리에 사전 등록한다. 실패해도 여기서 죽지 않되,
        # 실제 로드가 "architecture 미인식"으로 실패하면 이 import 실패를 원인으로 함께 드러낸다.
        mm_error = None
        try:
            import multimolecule  # noqa: F401
        except Exception as e:
            mm_error = e
            logging.logger.warning(f"multimolecule import 실패(RNA 계열 로드 시 필요): {e}")

        local_path = self._load_artifacts(run_id, model_name)
        trust_remote_code = self._needs_remote_code(local_path)
        logging.logger.info(f"Fill-Mask 모델 dir: {local_path} (trust_remote_code={trust_remote_code})")

        # 단일 cuda:0 가정 제거: device_map="auto" 로 가용 GPU에 자동 배치/샤딩한다.
        # 작은 모델(ESM2/RNA-FM/MoLFormer)은 cuda:0 한 장, 6B급(ESMC)은 여러 장에 분산되어 1장 초과 OOM 을 피한다.
        device_map = "auto" if torch.cuda.is_available() else None
        try:
            self.model = AutoModelForMaskedLM.from_pretrained(
                local_path, trust_remote_code=trust_remote_code, device_map=device_map
            )
        except (KeyError, ValueError) as e:
            if mm_error is not None:
                raise RuntimeError(
                    f"등록형 아키텍처 로드 실패 — multimolecule import 가 선행 실패했습니다: {mm_error}"
                ) from e
            raise
        self.tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        self.model.eval()
        # 입력 텐서 기준 디바이스(샤딩 시 첫 모듈이 위치한 디바이스). accelerate 가 이후 디바이스 이동을 처리.
        self.device = next(self.model.parameters()).device
        logging.logger.info("Fill-Mask 모델 로드 완료")

    @staticmethod
    def _needs_remote_code(local_path: str) -> bool:
        """아티팩트 config/tokenizer_config 에 auto_map(커스텀 코드 매핑)이 있으면 True.

        MoLFormer 처럼 허브 modeling/tokenization .py 를 함께 받은(스냅샷) 모델 로드에 필요.
        ESM2/RNA-FM 은 auto_map 이 없어 False. 신뢰 경계는 등록 단계(PREDEFINED 화이트리스트)에서 이미 좁혀진다.
        """
        import os

        for fname in ("config.json", "tokenizer_config.json"):
            fpath = os.path.join(local_path, fname)
            if os.path.isfile(fpath):
                try:
                    with open(fpath, encoding="utf-8") as f:
                        if "auto_map" in json.load(f):
                            return True
                except Exception:
                    pass
        return False

    def predict(self, data, device_str: str = "cpu"):
        """마스크가 포함된 서열을 받아 각 마스크 위치의 top-k 토큰 예측을 반환.

        반환 형식:
        {
          "predictions": [
            {"position": int, "predictions": [{"token": str, "score": float}, ...]}, ...
          ],
          "input_info": {"sequence": "...", "top_k": int}
        }
        """
        import torch

        if isinstance(data, (bytes, bytearray)):
            data = json.loads(data.decode("utf-8"))
        elif isinstance(data, str):
            data = json.loads(data)

        sequence = (data.get("sequence") or "").strip()
        if not sequence:
            raise ValueError("sequence 는 필수이며 비어있을 수 없습니다.")
        top_k = int(data.get("top_k") or self.DEFAULT_TOP_K)

        mask_token = self.tokenizer.mask_token
        if mask_token is None:
            raise ValueError("토크나이저에 mask 토큰이 없어 fill-mask 를 수행할 수 없습니다.")
        if mask_token not in sequence:
            raise ValueError(f"입력 sequence 에 마스크 토큰({mask_token})이 최소 1개 필요합니다.")

        inputs = self.tokenizer(sequence, truncation=True, max_length=self.MAX_LENGTH, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits

        mask_positions = (inputs["input_ids"][0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        results = []
        for pos in mask_positions.tolist():
            probs = torch.softmax(logits[0, pos], dim=-1)
            topk = torch.topk(probs, k=min(top_k, probs.shape[-1]))
            preds = [
                {"token": self.tokenizer.decode([tid]).strip(), "score": float(score)}
                for tid, score in zip(topk.indices.tolist(), topk.values.tolist())
            ]
            results.append({"position": int(pos), "predictions": preds})

        return {
            "predictions": results,
            "input_info": {"sequence": sequence, "top_k": top_k},
        }
