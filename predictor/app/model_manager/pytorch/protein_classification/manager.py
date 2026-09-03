import json
import os

from app.model_manager.base import BaseModelManager
from kserve import logging


class ProteinClassificationModelManager(BaseModelManager):
    """BFM(ESM2/ESMC) LoRA 어댑터 시퀀스 분류(task=protein-classification) 추론 매니저.

    학습 산출물은 `.pth` 단일 파일이 아니라 PEFT 어댑터 디렉토리(adapter_model.safetensors +
    adapter_config.json + tokenizer*)이다. 베이스 모델을 로드하고 그 위에 어댑터를 얹는다.

    base 는 AutoModelForSequenceClassification 로 로드하므로 base config 의 model_type 으로
    esm2(내장)·esmc(vendored 등록) 를 자동 분기한다(별도 분기 코드 불필요).

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

    @staticmethod
    def _bf16_if_esmc(base_source: str):
        """ESMC base 는 크기 무관 bf16 로 통일 로드(6B fp32=OOM 회피 + 300M 도 동일 정밀도). 그 외(esm2)는 native.

        base_source 의 config.json model_type 로 판정. 로컬 dir 이 아니면(HF repo id 폴백) None(=default fp32).
        """
        import torch

        cfg = os.path.join(base_source, "config.json")
        if os.path.isfile(cfg):
            try:
                with open(cfg, encoding="utf-8") as f:
                    if json.load(f).get("model_type") == "esmc":
                        return torch.bfloat16
            except Exception:
                pass
        return None

    def load_model(self, model_name: str, run_id: str):
        """MLflow 'adapter' 아티팩트를 내려받아 베이스+LoRA 어댑터를 로드."""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # ESMC 는 upstream transformers 에 없어 vendored 구현을 import 해 Auto 레지스트리에 등록한다.
        # (base config model_type=esmc 로드 및 ESMCTokenizer 해석에 필요. ESM2 에는 무영향.)
        try:
            import app.vendored.esmc  # noqa: F401
        except Exception as e:
            logging.logger.warning(f"vendored esmc 등록 실패(ESMC 외에는 무관): {e}")

        local_path = self._load_artifacts(run_id, model_name)
        adapter_dir = self._find_adapter_dir(local_path)
        logging.logger.info(f"BFM adapter dir: {adapter_dir}")

        # base 모델: MLflow 등록본(base_run_id/base_model_uri)을 우선 사용, 없으면 HF Hub(BASE_MODEL_ID)
        base_source = self.BASE_MODEL_ID
        if self.base_run_id and self.base_model_uri:
            try:
                base_source = self._load_artifacts(self.base_run_id, self.base_model_uri)
                logging.logger.info(f"BFM base (MLflow) dir: {base_source}")
            except Exception as e:
                logging.logger.warning(f"MLflow base 다운로드 실패, HF Hub fallback: {e}")
                base_source = self.BASE_MODEL_ID

        # 단일 cuda:0 가정 제거: base 를 device_map="auto" 로 로드하고, LoRA 어댑터는 그 배치를 계승한다.
        # 작은 base(ESM2/ESMC-300M)는 cuda:0 한 장, 6B급(ESMC-6B)은 여러 장에 분산되어 1장 초과 OOM 을 피한다.
        # AutoModelForSequenceClassification 이 base config 의 model_type 으로 esm2/esmc 를 자동 분기한다.
        device_map = "auto" if torch.cuda.is_available() else None
        load_dtype = self._bf16_if_esmc(base_source)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_source, num_labels=2, device_map=device_map, dtype=load_dtype
        )
        # ESMC(_bf16_if_esmc 가 bf16 반환): 학습과 동일하게 fused qkv/ffn 을 등가 nn.Linear 로 노출(surgery)해야
        # 표준 target_modules LoRA 어댑터(layernorm_qkv.linear/ffn.fc1/ffn.fc2)가 이름 매칭돼 로드된다.
        # 원본 vendored 코드는 불변, forward 수치 동일, 가중치 텐서 재사용. esm2 등 대상 모듈 없는 base 엔 no-op.
        if load_dtype is not None:
            from app.esmc_lora_ready import make_esmc_lora_ready

            make_esmc_lora_ready(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        self.model = PeftModel.from_pretrained(base_model, adapter_dir)
        self.model.eval()
        # 입력 텐서 기준 디바이스(샤딩 시 첫 모듈 디바이스). accelerate 가 이후 디바이스 이동을 처리.
        self.device = next(self.model.parameters()).device
        logging.logger.info("BFM LoRA 모델 로드 완료")

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
