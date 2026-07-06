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
        """MLflow run 아티팩트(base 모델 가중치)를 내려받아 MaskedLM 헤드로 로드."""
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        local_path = self._load_artifacts(run_id, model_name)
        logging.logger.info(f"Fill-Mask 모델 dir: {local_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForMaskedLM.from_pretrained(local_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(local_path)
        self.model.eval()
        logging.logger.info("Fill-Mask 모델 로드 완료")

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
