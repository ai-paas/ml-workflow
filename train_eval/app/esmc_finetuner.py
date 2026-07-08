#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ESM-C (biohub/ESMC-*) 파인튜닝.

ESM2 파인튜너(EsmFineTuner)의 preprocess/데이터셋/Trainer 파이프라인을 그대로 상속하고,
base 로드와 LoRA 대상만 ESMC 아키텍처에 맞게 재정의한다. train_eval.main() 의
`--model_kind esmc` 분기에서만 lazy import 된다.

■ LoRA 대상이 ESMC 공식 예시와 다른 이유
  공식 예시는 target_modules=["layernorm_qkv.1","out_proj","ffn.1","ffn.3"] 인데, 이는
  transformer_engine 이 설치돼 layernorm_qkv/ffn 이 nn.Sequential(그래서 .1/.3 이 nn.Linear)일 때의
  이름이다. 우리 이미지엔 transformer_engine 이 없어 vendored 구현이 순수 PyTorch 로 폴백하며, 이때
  QKV·FFN 가중치는 nn.Linear 가 아니라 커스텀 모듈 안의 raw nn.Parameter 다:
    - attn.layernorm_qkv.weight (fused QKV, shape [3*d, d])
    - ffn.fc1_weight, ffn.fc2_weight
  그래서 이 셋은 peft target_parameters 로 잡고, 어텐션 출력 투영(attn.out_proj, nn.Linear)만
  target_modules 로 잡는다. bare "out_proj" 는 분류 헤드의 classifier.out_proj 까지 매칭하므로
  "attn.out_proj" 로 좁힌다. target_parameters(ParamWrapper)는 lora_dropout!=0 을 지원하지 않아
  dropout=0 을 쓴다. train 컨테이너와 predictor 가 동일한 순수 PyTorch 폴백 구조라 어댑터가 그대로 호환된다.

  task_type 은 None(generic PeftModel)으로 둔다. SEQ_CLS 래퍼는 base 로 inputs_embeds 를 주입하는데
  vendored ESMC forward 는 이를 받지 않아(표준 transformers 모델과 달리) 학습·추론 양쪽에서 깨진다.
  base ESMCForSequenceClassification 이 이미 head+loss 를 계산하므로 generic 래퍼로 충분하다.
"""
import torch
from app.esm2_finetuner import EsmFineTuner
from app.vendored.esmc import ESMCForSequenceClassification  # import 부수효과로 esmc 를 Auto 레지스트리에 등록
from peft import LoraConfig, get_peft_model

MODEL_ID = "biohub/ESMC-300M"


class EsmcFineTuner(EsmFineTuner):
    """ESMC LoRA 파인튜너. EsmFineTuner 를 상속해 base 로드·LoRA 대상만 ESMC 용으로 재정의."""

    MODEL_KIND = "ESMC"
    # ESMC 는 bf16 로 학습(서빙과 정밀도 통일 + fp16 대비 넓은 동적범위로 안정).
    USE_BF16 = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 산출물은 esm2 와 분리된 하위 디렉토리로.
        self.output_dir = self.output_dir.parent / "esmc"

    def _resolve_base_path(self) -> str:
        # MLflow 카탈로그 base 아티팩트가 있으면 그걸, 없으면 HF Hub(MODEL_ID) 폴백.
        return self.base_dir or MODEL_ID

    def _build_peft_model(self, base_path: str):
        # bf16 로 로드(frozen base 라 master-weight 정밀도 불필요, 서빙과 통일). LoRA 어댑터는 fp32 로 학습된다.
        model = ESMCForSequenceClassification.from_pretrained(base_path, num_labels=2, dtype=torch.bfloat16)
        peft_config = LoraConfig(
            task_type=None,
            inference_mode=False,
            bias="none",
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["attn.out_proj"],
            target_parameters=["layernorm_qkv.weight", "ffn.fc1_weight", "ffn.fc2_weight"],
            modules_to_save=["classifier"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model
