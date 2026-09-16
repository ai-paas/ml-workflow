#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ESMC(swiglu) 의 fused QKV/FFN 가중치를 nn.Linear 서브모듈로 "노출"해, peft 표준 LoRA
(target_modules)로 학습할 수 있게 하는 로드-후 후처리(surgery).

■ 왜 필요한가
  ESMC 는 swiglu 라 attention 의 layernorm_qkv 와 ffn 이 nn.Linear 가 아니라 커스텀 모듈
  (_PyTorchLayerNormLinear / _PyTorchLayerNormMLP) 안의 raw nn.Parameter(`weight`,
  `fc1_weight`, `fc2_weight`)로 저장된다. peft 의 target_modules 는 nn.Linear 만 감쌀 수 있어,
  이 가중치들은 target_parameters(ParamWrapper)로만 잡히는데 — 그 경로는 매 forward 마다
  full-size 델타(weight_B @ weight_A)를 만들어 base 가중치에 병합하므로, 대형(ESMC-6B) 학습에서
  이 병합 사본들이 누적돼 단일 GPU(24GB)에서 OOM 난다.
  raw Parameter 를 등가 nn.Linear 로 감싸면 peft 가 저랭크 표준 LoRA 경로(활성 저랭크 투영,
  full-size 델타 없음)를 타 메모리가 급감하고, 공식과 동등한 qkv/ffn 커버리지를 24GB 한 장에서 얻는다.

■ 원본을 안 건드리는 이유
  vendored/esmc/modeling_esmc.py 는 upstream(Biohub) 충실 복사본이라 손대지 않는다. 대신 모델을
  로드한 뒤 여기서 해당 모듈만 등가 래퍼로 in-place 교체한다. forward 연산은 완전히 동일하고
  (F.linear(x, W) == nn.Linear(bias=False)(x)), 기존 Parameter 텐서를 그대로 재사용하므로
  별도 state_dict 키 리맵 없이 값·device·dtype 이 보존된다.

■ 학습/서빙 정합
  어댑터는 아래 노출된 이름(layernorm_qkv.linear / ffn.fc1 / ffn.fc2)으로 저장되므로,
  서빙(predictor)도 base 로드 후 동일하게 이 surgery 를 적용해야 어댑터가 매칭·로드된다.
"""
import torch.nn as nn
import torch.nn.functional as F
from app.vendored.esmc.modeling_esmc import _PyTorchLayerNormLinear, _PyTorchLayerNormMLP

# surgery 후 표준 LoRA 로 잡을 대상. layernorm_qkv/ffn 은 아래 래퍼가 노출한 nn.Linear,
# attn.out_proj 는 원래부터 nn.Linear(분류 헤드의 classifier.out_proj 와 구분하려 attn. 접두).
ESMC_LORA_TARGET_MODULES = ["layernorm_qkv.linear", "attn.out_proj", "ffn.fc1", "ffn.fc2"]


class _LoRAReadyLayerNormLinear(nn.Module):
    """_PyTorchLayerNormLinear 등가: raw `weight` 대신 nn.Linear(bias=False) 서브모듈 `linear` 로 노출.
    forward 는 원본과 동일(LayerNorm → Linear)."""

    def __init__(self, d_in, eps, layer_norm_weight, layer_norm_bias, linear):
        super().__init__()
        self.d_in = d_in
        self.eps = eps
        self.layer_norm_weight = layer_norm_weight
        self.layer_norm_bias = layer_norm_bias
        self.linear = linear

    def forward(self, x):
        x = F.layer_norm(x, (self.d_in,), self.layer_norm_weight, self.layer_norm_bias, self.eps)
        return self.linear(x)


class _LoRAReadyLayerNormMLP(nn.Module):
    """_PyTorchLayerNormMLP 등가: raw `fc1_weight`/`fc2_weight` 대신 nn.Linear 서브모듈 `fc1`/`fc2` 로 노출.
    forward 는 원본과 동일(LayerNorm → fc1 → SwiGLU → fc2)."""

    def __init__(self, hidden_size, eps, layer_norm_weight, layer_norm_bias, fc1, fc2):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.layer_norm_weight = layer_norm_weight
        self.layer_norm_bias = layer_norm_bias
        self.fc1 = fc1
        self.fc2 = fc2

    def forward(self, x):
        x = F.layer_norm(x, (self.hidden_size,), self.layer_norm_weight, self.layer_norm_bias, self.eps)
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = F.silu(x1) * x2
        return self.fc2(x)


def _wrap_ln_linear(child: _PyTorchLayerNormLinear) -> _LoRAReadyLayerNormLinear:
    d_out, d_in = child.weight.shape
    linear = nn.Linear(d_in, d_out, bias=False)
    linear.weight = child.weight  # 기존 Parameter 재사용(값·device·dtype 보존)
    return _LoRAReadyLayerNormLinear(child.d_in, child.eps, child.layer_norm_weight, child.layer_norm_bias, linear)


def _wrap_ln_mlp(child: _PyTorchLayerNormMLP) -> _LoRAReadyLayerNormMLP:
    fc1 = nn.Linear(child.hidden_size, 2 * child.ffn_hidden_size, bias=False)
    fc1.weight = child.fc1_weight
    fc2 = nn.Linear(child.ffn_hidden_size, child.hidden_size, bias=False)
    fc2.weight = child.fc2_weight
    return _LoRAReadyLayerNormMLP(
        child.hidden_size, child.eps, child.layer_norm_weight, child.layer_norm_bias, fc1, fc2
    )


def make_esmc_lora_ready(model: nn.Module) -> nn.Module:
    """로드된 ESMC 모델의 fused LN-Linear/LN-MLP 모듈을 nn.Linear 노출형으로 in-place 교체한다.
    가중치 텐서를 재사용하므로 값·배치는 불변이고, forward 결과도 수치적으로 동일하다.
    surgery 후 ESMC_LORA_TARGET_MODULES 로 qkv/ffn 에 표준 LoRA 를 걸 수 있다. model 을 반환한다.
    (transformer_engine 이 설치된 경우 layernorm_qkv/ffn 은 te.* 타입이라 여기서 교체되지 않는다 —
    현재 이미지엔 TE 가 없어 항상 _PyTorch* 폴백이므로 대상이 된다.)
    """
    replacements = []
    for parent in model.modules():
        for name, child in parent.named_children():
            if isinstance(child, _PyTorchLayerNormLinear):
                replacements.append((parent, name, _wrap_ln_linear(child)))
            elif isinstance(child, _PyTorchLayerNormMLP):
                replacements.append((parent, name, _wrap_ln_mlp(child)))
    for parent, name, new_module in replacements:
        setattr(parent, name, new_module)
    return model
