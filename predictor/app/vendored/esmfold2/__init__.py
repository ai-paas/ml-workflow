# Copyright 2026 Biohub. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Vendored ESMFold2 (단백질 구조예측) — 설치된 transformers 의 Auto 레지스트리에 esmfold2 를 등록한다.

■ 왜 vendoring 인가
  ESMFold2(biohub/ESMFold2)는 config 의 model_type 이 "esmfold2" 인데, upstream transformers(5.13)에는
  esmfold2 구현이 없다(models 목록 실측 확인 — esmc 와 동일 상황). esm SDK 경로는 transformers<4.48 을
  요구해 5.x 스택과 충돌하므로, esmc 와 같은 방식으로 modeling 코드만 vendoring 하고 런타임에 등록한다.

■ 대가 / 벤더링 시 원본과 다른 점 (원본 대비 수정한 지점은 코드에 `[VENDOR PATCH]` 로 표시)
  - transformers-root 상대 import(`from ...X`)를 `from transformers.X` 로 치환(패키지 위치 이동).
  - 멀티GPU 전용 `distributed/` 서브패키지 제외 — Biohub 모노레포 `projects.*` 절대경로에 의존하고,
    소형(1.3GB)·단일GPU 서빙엔 불필요하다. 메인 modeling 은 distributed 를 참조하지 않음(실측 확인).
  - 선택적 fused 커널(transformer_engine / triton)은 로드 실패 시 순수 PyTorch 로 폴백(동작 정상, 속도만 차이).
    `kernels/` 는 triton .py 라 triton 설치 시 JIT 컴파일되고, 미설치/초기화실패면 폴백 경로로 우회된다.
  - [VENDOR PATCH] `modeling_esmfold2_common.py` 의 `.kernels` import 가드를 원본 `except ImportError`
    → `except Exception` 으로 확장. triton 이 설치돼 있어도 C 컴파일러(CC)가 없는 서빙 이미지에서는
    autotune 데코레이터가 import 시점에 RuntimeError("Failed to find C compiler")를 던지는데, 이는
    ImportError 가 아니라 원본 가드에 안 걸려 로드가 크래시했다. 폭넓게 잡아 순수 PyTorch 로 폴백한다.

■ 출처(provenance)
  https://github.com/Biohub/transformers/tree/main/src/transformers/models/esmfold2
  라이선스는 원본(Apache-2.0)을 따른다.
"""
from transformers import AutoConfig, AutoModel

from .configuration_esmfold2 import ESMFold2Config
from .modeling_esmfold2 import ESMFold2Model


def _register():
    """esmfold2 를 설치된 transformers 의 Auto 레지스트리에 등록한다(중복 등록/미지원은 무시)."""
    try:
        AutoConfig.register("esmfold2", ESMFold2Config)
    except (ValueError, KeyError):
        pass  # 이미 등록됨(포크 환경 등)
    try:
        AutoModel.register(ESMFold2Config, ESMFold2Model)
    except (ValueError, KeyError):
        pass


_register()

__all__ = ["ESMFold2Config", "ESMFold2Model"]
