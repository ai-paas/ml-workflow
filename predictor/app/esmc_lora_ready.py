#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ESMC(swiglu) 의 fused QKV/FFN 가중치를 nn.Linear 서브모듈로 "노출"해, peft 표준 LoRA
어댑터(target_modules)를 얹을 수 있게 하는 로드-후 후처리(surgery). 서빙(predictor)용.

학습(train_eval/app/esmc_lora_ready.py)과 반드시 동일한 surgery 를 적용해야, 학습이 저장한 어댑터
이름(layernorm_qkv.linear / ffn.fc1 / ffn.fc2)이 서빙 base 의 모듈 이름과 일치해 로드된다.

원본 vendored/esmc/modeling_esmc.py 는 건드리지 않는다. 로드된 모델의 _PyTorchLayerNormLinear /
_PyTorchLayerNormMLP 모듈만 등가 nn.Linear 래퍼로 in-place 교체하며(가중치 텐서 재사용), forward
연산은 완전히 동일하다(F.linear(x, W) == nn.Linear(bias=False)(x)).
"""
import torch.nn as nn
import torch.nn.functional as F
from app.vendored.esmc.modeling_esmc import _PyTorchLayerNormLinear, _PyTorchLayerNormMLP

# 학습과 동일해야 하는 표준 LoRA 대상(어댑터 이름 정합). 서빙은 어댑터가 이미 이 이름을 담고 있어
# 직접 참조하진 않지만, 학습/서빙 계약을 한곳에서 보이도록 함께 둔다.
ESMC_LORA_TARGET_MODULES = ["layernorm_qkv.linear", "attn.out_proj", "ffn.fc1", "ffn.fc2"]


class _LoRAReadyLayerNormLinear(nn.Module):
    """_PyTorchLayerNormLinear 등가: raw `weight` 대신 nn.Linear(bias=False) 서브모듈 `linear` 로 노출."""

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
    """_PyTorchLayerNormMLP 등가: raw `fc1_weight`/`fc2_weight` 대신 nn.Linear 서브모듈 `fc1`/`fc2` 로 노출."""

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
    가중치 텐서를 재사용하므로 값·forward 결과가 불변이다. esmc 가 아닌 모델(esm2 등)엔 대상 모듈이
    없어 no-op 이다. model 을 반환한다.
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
