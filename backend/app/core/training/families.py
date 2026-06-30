"""학습 가능 모델군(family) 단일 레지스트리.

학습 검증의 세 질문 — (1) 학습 가능한가, (2) 어떤 데이터셋 분류를 요구하나,
(3) 어떤 학습 컨테이너로 가나 — 의 답은 모두 "이 모델이 어느 학습 가능 모델군에
속하나"라는 하나의 사실에서 나온다. 그 판정을 여기 한 곳에 모은다.

모델군 식별은 (format, repo_id) 로 한다:
  - yolox 는 고유 포맷(model_format=yolox)을 가지므로 format 으로 식별.
  - esm2 는 포맷이 pytorch 로 detr/yolos(학습 불가)와 공유되므로 repo_id 로만 유일 식별.
매칭 코드는 균일하고, 이 비대칭은 "어느 판별자 집합을 채우느냐"는 데이터로 표현된다.

설계: docs/fine-tuning-and-inference/training-validation-refactor.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config.db.enums import DatasetKindEnum, ModelFormatEnum, ModelTypeEnum

# 학습 가능 ESM2 repo_id 집합 — 변형 추가 시 여기 한 곳만 늘리면 된다.
ESM2_T6_8M_REPO_ID = "facebook/esm2_t6_8M_UR50D"
ESM2_SUPPORTED_REPO_IDS = frozenset({ESM2_T6_8M_REPO_ID})


@dataclass(frozen=True)
class TrainableFamily:
    """학습 가능 모델군 1개의 정의(판별 기준 + 데이터셋 분류 + 컨테이너 키)."""

    key: str  # train_eval 컨테이너 디스패치 키(= 기존 model_kind): "yolox" | "esm2"
    model_type: ModelTypeEnum  # 이 군이 가져야 할 모델 타입(ODM | pLM)
    dataset_kind: DatasetKindEnum  # 이 군이 요구하는 데이터셋 분류
    match_formats: frozenset = frozenset()  # 이 model_format 이면 이 군(yolox 처럼 고유 포맷)
    match_repo_ids: frozenset = frozenset()  # 이 repo_id 면 이 군(esm2 처럼 포맷 공유 → repo_id 로만 구분)

    def matches(self, *, format_name: str, repo_id: str) -> bool:
        fmt = (format_name or "").strip().lower()
        rid = (repo_id or "").strip()
        return fmt in self.match_formats or rid in self.match_repo_ids


TRAINABLE_FAMILIES: tuple[TrainableFamily, ...] = (
    TrainableFamily(
        key="yolox",
        model_type=ModelTypeEnum.ODM,
        dataset_kind=DatasetKindEnum.OBJECT_DETECTION,
        match_formats=frozenset({ModelFormatEnum.YOLOX.value}),
    ),
    TrainableFamily(
        key="esm2",
        model_type=ModelTypeEnum.PLM,
        dataset_kind=DatasetKindEnum.PROTEIN_CLASSIFICATION,
        match_repo_ids=ESM2_SUPPORTED_REPO_IDS,
    ),
)


def resolve_family(*, format_name: str, repo_id: str) -> Optional[TrainableFamily]:
    """(format, repo_id) 로 학습 가능 모델군을 찾는다. 매칭이 없으면 None(= 학습 불가/미지원)."""
    return next(
        (f for f in TRAINABLE_FAMILIES if f.matches(format_name=format_name, repo_id=repo_id)),
        None,
    )
