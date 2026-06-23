import json
import os

from app.model_manager.base import BaseModelManager
from kserve import logging


class ProteinLanguageModelManager(BaseModelManager):
    """ESM2 (facebook/esm2_t6_8M_UR50D) LoRA 어댑터 시퀀스 분류 추론 매니저.

    학습 산출물은 `.pth` 단일 파일이 아니라 PEFT 어댑터 디렉토리(adapter_model.safetensors +
    adapter_config.json + tokenizer*)이다. 베이스 모델을 HF 에서 로드하고 그 위에 어댑터를 얹는다.

    입력 페이로드는 이미지가 아니라 단백질 서열 dict: {"epitope": "...", "cdr3b": "..."}.
    """

    BASE_MODEL_ID = "facebook/esm2_t6_8M_UR50D"
    MAX_LENGTH = 80

    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.device = None
        # base 모델 MLflow 위치(InferenceModel 이 주입). 없으면 HF Hub(BASE_MODEL_ID) 사용.
        self.base_run_id = None
        self.base_model_uri = None

    @staticmethod
    def _find_adapter_dir(path: str) -> str:
        """adapter_config.json 을 포함한 디렉토리를 찾는다. 못 찾으면 path 자체를 반환."""
        for root, _dirs, files in os.walk(path):
            if "adapter_config.json" in files:
                return root
        return path

    def load_model(self, model_name: str, run_id: str):
        """MLflow 'adapter' 아티팩트를 내려받아 베이스+LoRA 어댑터를 로드."""
        import torch
        from peft import PeftModel
        from transformers import AutoTokenizer, EsmForSequenceClassification

        local_path = self._load_artifacts(run_id, model_name)
        adapter_dir = self._find_adapter_dir(local_path)
        logging.logger.info(f"ESM2 adapter dir: {adapter_dir}")

        # base 모델: MLflow 등록본(base_run_id/base_model_uri)을 우선 사용, 없으면 HF Hub(BASE_MODEL_ID)
        base_source = self.BASE_MODEL_ID
        if self.base_run_id and self.base_model_uri:
            try:
                base_source = self._load_artifacts(self.base_run_id, self.base_model_uri)
                logging.logger.info(f"ESM2 base (MLflow) dir: {base_source}")
            except Exception as e:
                logging.logger.warning(f"MLflow base 다운로드 실패, HF Hub fallback: {e}")
                base_source = self.BASE_MODEL_ID

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model = EsmForSequenceClassification.from_pretrained(base_source, num_labels=2)
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        self.model = PeftModel.from_pretrained(base_model, adapter_dir).to(self.device)
        self.model.eval()
        logging.logger.info("ESM2 LoRA 모델 로드 완료")

    def predict(self, data, device_str: str = "cpu"):
        """단백질 서열 dict 를 받아 이진 분류 확률을 반환.

        반환 형식:
        {
          "predictions": [{"label": int, "score": float, "probabilities": {"0": p0, "1": p1}}],
          "input_info": {"epitope": "...", "cdr3b": "..."}
        }
        """
        import torch

        if isinstance(data, (bytes, bytearray)):
            data = json.loads(data.decode("utf-8"))
        elif isinstance(data, str):
            data = json.loads(data)

        epitope = (data.get("epitope") or "").strip()
        cdr3b = (data.get("cdr3b") or "").strip()
        if not epitope or not cdr3b:
            raise ValueError("epitope 와 cdr3b 는 필수이며 비어있을 수 없습니다.")

        text = epitope + cdr3b
        inputs = self.tokenizer(text, truncation=True, max_length=self.MAX_LENGTH, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        label = int(torch.argmax(probs).item())

        return {
            "predictions": [
                {
                    "label": label,
                    "score": float(probs[label].item()),
                    "probabilities": {"0": float(probs[0].item()), "1": float(probs[1].item())},
                }
            ],
            "input_info": {"epitope": epitope, "cdr3b": cdr3b},
        }
