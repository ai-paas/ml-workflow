#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ESM-C (biohub/ESMC-*) 파인튜닝.

ESM2 파인튜너(EsmFineTuner)의 preprocess/데이터셋/Trainer 파이프라인을 그대로 상속하고,
base 로드와 LoRA 대상만 ESMC 아키텍처에 맞게 재정의한다. train_eval.main() 의
`--model_kind esmc` 분기에서만 lazy import 된다.

■ LoRA 대상 — fused qkv/ffn 을 nn.Linear 로 노출한 뒤 표준 target_modules 로 잡는다
  ESMC(swiglu)는 attention 의 layernorm_qkv 와 ffn 이 nn.Linear 가 아니라 커스텀 모듈 안의 raw
  nn.Parameter(layernorm_qkv.weight[3d,d], ffn.fc1_weight, ffn.fc2_weight)다. 이를 peft
  target_parameters 로 잡으면 매 forward 마다 full-size 델타(weight_B@weight_A)를 base 에 병합해,
  대형(6B) 학습에서 그 병합 사본 누적으로 단일 24GB GPU 에서 OOM 난다.
  그래서 로드 후 esmc_lora_ready.make_esmc_lora_ready 로 이 fused 모듈을 등가 nn.Linear 로 노출하고
  (원본 vendored 코드는 불변, forward 수치 동일, 가중치 텐서 재사용), 표준 target_modules 로 저랭크
  LoRA(활성 저랭크 투영, full-size 델타 없음)를 건다:
    layernorm_qkv.linear / ffn.fc1 / ffn.fc2 (노출된 nn.Linear) + attn.out_proj (원래 nn.Linear).
  bare "out_proj" 는 분류 헤드의 classifier.out_proj 까지 매칭하므로 "attn.out_proj" 로 좁힌다.
  표준 target_modules 경로라 lora_dropout!=0 도 지원된다. train 과 predictor 가 동일 surgery 를
  적용하므로 어댑터 이름(...linear/fc1/fc2)이 양쪽에서 일치한다.

  task_type 은 None(generic PeftModel)으로 둔다. SEQ_CLS 래퍼는 base 로 inputs_embeds 를 주입하는데
  vendored ESMC forward 는 이를 받지 않아(표준 transformers 모델과 달리) 학습·추론 양쪽에서 깨진다.
  base ESMCForSequenceClassification 이 이미 head+loss 를 계산하므로 generic 래퍼로 충분하다.
"""
import torch
from app.esm2_finetuner import EsmFineTuner
from app.esmc_lora_ready import ESMC_LORA_TARGET_MODULES, make_esmc_lora_ready
from app.vendored.esmc import ESMCForSequenceClassification  # import 부수효과로 esmc 를 Auto 레지스트리에 등록
from loguru import logger
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
        # fused qkv/ffn 을 등가 nn.Linear 로 노출(원본 불변, 수치 동일) → 표준 target_modules 저랭크 LoRA 가능.
        # 이 surgery 로 6B 도 full-size 델타 병합 없이 단일 GPU 에 들어간다.
        make_esmc_lora_ready(model)
        logger.info(f"[esmc] LoRA-ready surgery 적용, target_modules={ESMC_LORA_TARGET_MODULES}")
        peft_config = LoraConfig(
            task_type=None,
            inference_mode=False,
            bias="none",
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=ESMC_LORA_TARGET_MODULES,
            modules_to_save=["classifier"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model

    @staticmethod
    def _preprocess_logits_for_metrics(logits, labels):
        """ESMC forward 는 (logits, last_hidden_state, ...) 를 반환한다(분류 logits 는 첫 원소).
        그대로 누적하면 compute_metrics 의 np.asarray 가 ragged 로 실패하고, 대형 val 셋에서는
        last_hidden_state 누적으로 OOM 위험도 있어, eval 누적 전에 분류 logits 만 남긴다.
        """
        if isinstance(logits, (tuple, list)):
            return logits[0]
        return logits
