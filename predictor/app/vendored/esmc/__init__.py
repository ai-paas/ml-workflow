"""Vendored ESM-C (ESMC) 모델 코드 — 설치된 transformers 에 `esmc` 아키텍처를 등록한다.

■ 왜 vendoring 인가 (트레이드오프 기록)
  ESMC(biohub/ESMC-300M · biohub/ESMC-6B)는 config 의 model_type 이 "esmc" 인데, upstream
  transformers 에는 esmc 구현이 없다(5.13 및 main 모두 부재 — GitHub API 로 확인). 즉
  `AutoModelForMaskedLM.from_pretrained("biohub/ESMC-*")` 는 upstream 에서 "unrecognized
  architecture 'esmc'" 로 실패한다. esmc 를 제공하는 경로는 둘뿐이고 모두 우리 스택과 충돌한다:
    (a) Biohub 의 transformers 포크(내부에 esmc 내장, 버전 4.57.6) — esm 패키지가 끌어온다.
    (b) EvolutionaryScale `esm` SDK — transformers<4.48.2 를 요구한다.
  둘 다 transformers 4.57.x 대에 묶여 있는데, 우리 스택은 RNA-FM(multimolecule)·RF-DETR·MoLFormer
  때문에 transformers 5.x 가 필요하다(4.57.6 에는 이들 지원이 없다). 한 이미지에 esmc 와 5.x 모델을
  동시에 담으려면, esmc 를 4.57.6 에 맞추는 게 아니라 esmc 코드만 가져와 5.x 에 얹어야 한다.

  그래서 esmc 모델 구현 파일만 이 repo 로 복사(vendoring)하고, 런타임에 설치된 transformers 의
  Auto 레지스트리에 등록한다(아래 _register). 포크·SDK·Python 3.12 없이 upstream transformers 위에서
  esmc 가 동작한다(multimolecule 가 rnafm 을 등록하는 것과 같은 원리, 코드만 우리가 소유).

■ 대가
  - esmc 구현 사본을 우리가 소유·유지보수한다. 원본이 갱신되면 재-vendoring 이 필요하다.
    다만 코드가 상위 안정 API(PreTrainedModel/modeling_outputs/utils)만 사용해 upstream 상위호환이 확인됐다.
  - 선택적 fused 커널(transformer_engine/xformers/flash-attn) 미설치 시 순수 PyTorch 로 폴백한다(동작 정상,
    미세한 수치 차이만). 필수 아님.

■ 출처(provenance)
  github.com/Biohub/transformers  src/transformers/models/esmc/
  @ ef32577f55da19a4989cd7b22e004dc43a4998cb
  원본 대비 변경: 상대 import `from ...X` 를 `from transformers.X` 로 치환(패키지 위치 이동에 따른 것)한 것뿐.
  라이선스는 원본(Apache-2.0)을 따른다.
"""

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from .configuration_esmc import ESMCConfig
from .modeling_esmc import ESMCForMaskedLM, ESMCForSequenceClassification, ESMCForTokenClassification, ESMCModel
from .tokenization_esmc import ESMCTokenizer


def _register():
    """esmc 를 설치된 transformers 의 Auto 레지스트리에 등록한다(중복 등록/미지원은 무시)."""
    try:
        AutoConfig.register("esmc", ESMCConfig)
    except (ValueError, KeyError):
        pass  # 이미 등록됨(포크 환경 등)
    for auto_cls, model_cls in (
        (AutoModel, ESMCModel),
        (AutoModelForMaskedLM, ESMCForMaskedLM),
        (AutoModelForSequenceClassification, ESMCForSequenceClassification),
        (AutoModelForTokenClassification, ESMCForTokenClassification),
    ):
        try:
            auto_cls.register(ESMCConfig, model_cls)
        except (ValueError, KeyError):
            pass
    try:
        AutoTokenizer.register(ESMCConfig, fast_tokenizer_class=ESMCTokenizer)
    except (ValueError, KeyError):
        pass


_register()

__all__ = [
    "ESMCConfig",
    "ESMCModel",
    "ESMCForMaskedLM",
    "ESMCForSequenceClassification",
    "ESMCForTokenClassification",
    "ESMCTokenizer",
]
