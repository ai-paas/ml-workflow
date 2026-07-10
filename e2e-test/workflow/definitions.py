"""워크플로우 시나리오 정의 모듈

§5.2 개선 후 정상 동작하는 시나리오(1~9: LLM/RAG, 10: ODM 단순)의 메타데이터, 프롬프트, workflow_definition 빌더를 제공한다.

CLI 실행 (SCENARIO 필수):
  python workflow/definitions.py 3      # 3번 시나리오 상세
"""

from __future__ import annotations

import json
import sys
import uuid


def _ref(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _model_component(ref_id: str, name: str, model_id: int, prompt_id: int | None = None) -> dict:
    comp = {
        "ref_id": ref_id,
        "name": name,
        "type": "MODEL",
        "model_id": model_id,
        "config": {"temperature": 0.7, "max_tokens": 4096},
    }
    if prompt_id is not None:
        comp["prompt_id"] = prompt_id
    return comp


def _kb_component(ref_id: str, name: str, kb_id: int, top_k: int = 3) -> dict:
    return {
        "ref_id": ref_id,
        "name": name,
        "type": "KNOWLEDGE_BASE",
        "knowledge_base_id": kb_id,
        "config": {"top_k": top_k},
    }


def _odm_model_component(ref_id: str, name: str, model_id: int) -> dict:
    """ODM 전용 MODEL — temperature/max_tokens 등 LLM 설정 없음(백엔드는 model_id로 LLM vs ODM 구분)."""
    return {
        "ref_id": ref_id,
        "name": name,
        "type": "MODEL",
        "model_id": model_id,
    }


# ── 시나리오 메타데이터 ──────────────────────────────────────────────

SCENARIOS: dict[int, dict] = {
    1: {
        "name": "단순 LLM",
        "graph": "시작 → LLM → 종료",
        "kb_count": 0,
        "kb_labels": [],
        "inference_text": "대한민국의 수도는 어디인가요?",
        "prompts": {},
    },
    2: {
        "name": "단순 RAG",
        "graph": "시작 → KB → LLM → 종료",
        "kb_count": 1,
        "kb_labels": ["고압가스 안전관리법"],
        "inference_text": "고압가스 안전관리법의 목적은 무엇인가요?",
        "prompts": {},
    },
    3: {
        "name": "LLM 체인",
        "graph": "시작 → LLM → LLM → 종료",
        "kb_count": 0,
        "kb_labels": [],
        "inference_text": "대한민국의 수도는 어디인가요?",
        "prompts": {
            "LLM-A": {
                "name": "배경지식 탐색",
                "content": (
                    "당신은 지식 조사 전문가입니다.\n"
                    "사용자의 질문에 대해 관련 배경 지식, 핵심 개념, 그리고 다양한 관점을 폭넓게 조사하여 정리해주세요.\n"
                    "구체적인 사실과 맥락 정보를 포함하여 후속 분석에 활용할 수 있도록 작성해주세요."
                ),
                "has_context": False,
            },
            "LLM-B": {
                "name": "보고서 생성",
                "content": (
                    "당신은 전문 보고서 작성자입니다.\n"
                    "이전 단계에서 조사된 배경 지식과 분석 내용을 바탕으로, 체계적이고 읽기 쉬운 보고서를 작성해주세요.\n"
                    "서론, 본론, 결론 구조로 핵심 내용을 요약하고 실용적인 시사점을 도출해주세요."
                ),
                "has_context": False,
            },
        },
    },
    4: {
        "name": "RAG + LLM 체인",
        "graph": "시작 → KB → LLM → LLM → 종료",
        "kb_count": 1,
        "kb_labels": ["고압가스 안전관리법"],
        "inference_text": "고압가스 안전관리법의 목적은 무엇인가요?",
        "prompts": {
            "LLM-A": {
                "name": "참고자료 기반 초안 작성",
                "content": (
                    "당신은 법률 문서 분석 전문가입니다.\n\n"
                    "아래 참고자료를 바탕으로 사용자의 질문에 대한 초안 답변을 작성해주세요.\n"
                    "참고자료에서 관련 조항과 핵심 내용을 정확히 인용하여 답변하세요.\n\n"
                    "[참고자료]\n{context}"
                ),
                "has_context": True,
            },
            "LLM-B": {
                "name": "최종 보고서 정제",
                "content": (
                    "당신은 법률 보고서 편집 전문가입니다.\n"
                    "이전 단계의 초안 분석을 검토하여 더 정확하고 체계적인 최종 보고서로 정제해주세요.\n"
                    "논리적 흐름을 개선하고, 핵심 결론을 명확히 제시해주세요."
                ),
                "has_context": False,
            },
        },
    },
    5: {
        "name": "쿼리 정제 RAG",
        "graph": "시작 → LLM → KB → LLM → 종료",
        "kb_count": 1,
        "kb_labels": ["고압가스 안전관리법"],
        "inference_text": "고압가스 안전관리법의 목적은 무엇인가요?",
        "prompts": {
            "LLM-A (쿼리 정제)": {
                "name": "검색 쿼리 최적화",
                "content": (
                    "당신은 검색 쿼리 최적화 전문가입니다.\n"
                    "사용자의 질문을 분석하여 법률 문서 검색에 최적화된 핵심 키워드와 검색 쿼리를 추출해주세요.\n"
                    "원래 질문의 의도를 유지하면서, 관련 법률 용어와 조항명을 포함한 정제된 검색어를 생성해주세요."
                ),
                "has_context": False,
            },
            "LLM-B (최종 답변)": {
                "name": "참고자료 기반 최종 답변",
                "content": (
                    "당신은 법률 자문 전문가입니다.\n\n"
                    "아래 참고자료를 바탕으로 사용자의 질문에 대해 정확하고 상세한 답변을 생성해주세요.\n"
                    "참고자료의 내용을 근거로 제시하며, 참고자료에 없는 내용은 추측하지 마세요.\n\n"
                    "[참고자료]\n{context}"
                ),
                "has_context": True,
            },
        },
    },
    6: {
        "name": "병렬 분기",
        "graph": "시작 → [LLM1 / LLM2] → LLM3 → 종료",
        "kb_count": 0,
        "kb_labels": [],
        "inference_text": "대한민국의 수도는 어디인가요?",
        "prompts": {
            "LLM-1 (분기A)": {
                "name": "사실 기반 분석",
                "content": (
                    "당신은 사실 기반 분석 전문가입니다.\n"
                    "사용자의 질문에 대해 객관적 사실, 통계, 역사적 배경을 중심으로 분석해주세요.\n"
                    "검증된 정보와 구체적인 데이터를 바탕으로 답변하세요."
                ),
                "has_context": False,
            },
            "LLM-2 (분기B)": {
                "name": "비판적 관점 분석",
                "content": (
                    "당신은 비판적 분석 전문가입니다.\n"
                    "사용자의 질문에 대해 잠재적 문제점, 리스크, 다양한 이해관계자의 관점을 분석해주세요.\n"
                    "장단점을 균형 있게 평가하고, 간과하기 쉬운 측면을 짚어주세요."
                ),
                "has_context": False,
            },
            "LLM-3 (합류)": {
                "name": "종합 분석 보고서",
                "content": (
                    "당신은 종합 분석 보고서 작성자입니다.\n"
                    "이전 단계의 다양한 분석 결과를 종합하여 균형 잡힌 최종 보고서를 작성해주세요.\n"
                    "각 분석의 핵심 포인트를 통합하고, 실행 가능한 결론과 권고사항을 제시해주세요."
                ),
                "has_context": False,
            },
        },
    },
    7: {
        "name": "비대칭 병렬",
        "graph": "시작 → [LLM1 / KB+LLM2] → LLM3 → 종료",
        "kb_count": 1,
        "kb_labels": ["고압가스 안전관리법"],
        "inference_text": "고압가스 안전관리법의 목적은 무엇인가요?",
        "prompts": {
            "LLM-1 (분기A)": {
                "name": "일반 지식 분석",
                "content": (
                    "당신은 일반 지식 분석 전문가입니다.\n"
                    "사용자의 질문에 대해 일반적으로 알려진 배경 지식과 상식적 맥락을 분석해주세요.\n"
                    "폭넓은 관점에서 핵심 개념과 관련 정보를 정리해주세요."
                ),
                "has_context": False,
            },
            "LLM-2 (분기B)": {
                "name": "문서 기반 전문 분석",
                "content": (
                    "당신은 문서 기반 전문 분석가입니다.\n\n"
                    "아래 참고자료를 바탕으로 사용자의 질문에 대한 전문적인 분석을 제공해주세요.\n"
                    "참고자료의 구체적인 내용을 인용하여 근거 있는 답변을 작성하세요.\n\n"
                    "[참고자료]\n{context}"
                ),
                "has_context": True,
            },
            "LLM-3 (합류)": {
                "name": "종합 보고서 작성",
                "content": (
                    "당신은 종합 보고서 작성 전문가입니다.\n"
                    "일반 지식 분석과 문서 기반 전문 분석을 종합하여 최종 보고서를 작성해주세요.\n"
                    "양쪽 분석의 핵심을 통합하고, 일관된 결론을 도출해주세요."
                ),
                "has_context": False,
            },
        },
    },
    8: {
        "name": "대칭 병렬 RAG",
        "graph": "시작 → [KB(고압가스)+LLM1 / KB(건축서비스)+LLM2] → LLM3 → 종료",
        "kb_count": 2,
        "kb_labels": ["고압가스 안전관리법", "건축서비스산업 진흥법"],
        "inference_text": "고압가스 안전관리와 건축서비스산업 관련 법률의 주요 내용을 비교 분석해주세요.",
        "prompts": {
            "LLM-1 (분기A)": {
                "name": "고압가스 법률 조문 분석",
                "content": (
                    "당신은 법률 조문 분석 전문가입니다.\n\n"
                    "아래 참고자료에서 고압가스 안전관리에 관한 법률 조항을 찾아 조문별로 핵심 내용을 정리해주세요.\n"
                    "각 조항의 목적, 적용 범위, 주요 규정 사항을 체계적으로 분석해주세요.\n\n"
                    "[참고자료]\n{context}"
                ),
                "has_context": True,
            },
            "LLM-2 (분기B)": {
                "name": "건축서비스 법률 실무 분석",
                "content": (
                    "당신은 법률 실무 적용 전문가입니다.\n\n"
                    "아래 참고자료를 바탕으로 건축서비스산업 진흥에 관한 법률의 실무 적용 관점에서 "
                    "주의사항, 준수 절차, 위반 시 제재를 분석해주세요.\n"
                    "현장에서 실제로 적용할 때 알아야 할 실무적 해석과 유의점을 중심으로 정리해주세요.\n\n"
                    "[참고자료]\n{context}"
                ),
                "has_context": True,
            },
            "LLM-3 (합류)": {
                "name": "종합 법률 비교 보고서",
                "content": (
                    "당신은 종합 법률 보고서 작성 전문가입니다.\n"
                    "고압가스 안전관리법 분석 결과와 건축서비스산업 진흥법 분석 결과를 종합하여 "
                    "최종 비교 보고서를 작성해주세요.\n"
                    "두 법률의 공통점, 차이점, 그리고 각각의 핵심 규정을 포함한 체계적인 보고서를 생성해주세요."
                ),
                "has_context": False,
            },
        },
    },
    9: {
        "name": "대칭 병렬 쿼리 정제 RAG",
        "graph": "시작 → [LLM→KB(고압가스) / LLM→KB(건축서비스)] → LLM → 종료",
        "kb_count": 2,
        "kb_labels": ["고압가스 안전관리법", "건축서비스산업 진흥법"],
        "inference_text": "고압가스 안전관리와 건축서비스산업 관련 법률의 주요 내용을 비교 분석해주세요.",
        "prompts": {
            "LLM-A (분기A 쿼리정제)": {
                "name": "고압가스 법률 키워드 추출",
                "content": (
                    "당신은 법률 키워드 추출 전문가입니다.\n"
                    "사용자의 질문에서 고압가스 안전관리에 관한 핵심 법률 용어, 조항명, "
                    "주요 개념을 추출하여 검색 쿼리를 생성해주세요.\n"
                    "고압가스 관련 정확한 법률 용어를 사용하여 문서 검색의 정밀도를 높이는 데 집중해주세요."
                ),
                "has_context": False,
            },
            "LLM-B (분기B 쿼리정제)": {
                "name": "건축서비스 법률 개념 확장",
                "content": (
                    "당신은 법률 개념 확장 전문가입니다.\n"
                    "사용자의 질문과 연관된 건축서비스산업 진흥에 관한 상위 개념, 관련 법률, "
                    "유사 규정을 탐색하여 확장된 검색 쿼리를 생성해주세요.\n"
                    "건축서비스 관련 직접적인 키워드뿐 아니라 관련 분야와 연관 조항까지 포괄하는 검색어를 만들어주세요."
                ),
                "has_context": False,
            },
            "LLM-C (합류/최종답변)": {
                "name": "종합 법률 자문 답변",
                "content": (
                    "당신은 법률 자문 종합 답변 전문가입니다.\n\n"
                    "고압가스 안전관리법과 건축서비스산업 진흥법에서 수집된 아래 참고자료를 종합하여 "
                    "사용자의 질문에 대한 포괄적인 답변을 생성해주세요.\n"
                    "두 법률의 내용을 교차 검증하고, 일관되고 정확한 최종 비교 분석을 작성해주세요.\n\n"
                    "[참고자료]\n{context}"
                ),
                "has_context": True,
            },
        },
    },
    10: {
        "name": "ODM 단순",
        "graph": "시작 → MODEL → 종료",
        "kb_count": 0,
        "kb_labels": [],
        "inference_text": "",
        "inference_kind": "ml",
        "prompts": {},
    },
    11: {
        "name": "pLM 단순",
        "graph": "시작 → MODEL → 종료",
        "kb_count": 0,
        "kb_labels": [],
        "inference_text": "",
        "inference_kind": "plm",
        "prompts": {},
    },
    12: {
        "name": "BFM fill-mask 단순",
        "graph": "시작 → MODEL → 종료",
        "kb_count": 0,
        "kb_labels": [],
        "inference_text": "",
        "inference_kind": "fill_mask",
        "prompts": {},
    },
    13: {
        "name": "BFM structure-prediction 단순",
        "graph": "시작 → MODEL → 종료",
        "kb_count": 0,
        "kb_labels": [],
        "inference_text": "",
        "inference_kind": "structure_prediction",
        "prompts": {},
    },
}


# ── 워크플로우 정의 빌더 ─────────────────────────────────────────────


def _build_scenario_1(model_id: int, kb_ids: list[int], top_k: int, prompt_ids: dict) -> dict:
    """시작 → LLM → 종료"""
    start, llm, end = _ref("start"), _ref("llm"), _ref("end")
    return {
        "components": [
            {"ref_id": start, "name": "시작", "type": "START"},
            _model_component(llm, "LLM", model_id),
            {"ref_id": end, "name": "끝", "type": "END"},
        ],
        "connections": [
            {"source_ref_id": start, "target_ref_id": llm},
            {"source_ref_id": llm, "target_ref_id": end},
        ],
    }


def _build_scenario_10(model_id: int, kb_ids: list[int], top_k: int, prompt_ids: dict) -> dict:
    """시작 → ODM(MODEL) → 종료 — 시나리오 1과 위상만 같고 LLM 전용 설정·이름을 쓰지 않는다."""
    start, odm, end = _ref("start"), _ref("odm"), _ref("end")
    return {
        "components": [
            {"ref_id": start, "name": "시작", "type": "START"},
            _odm_model_component(odm, "MODEL", model_id),
            {"ref_id": end, "name": "끝", "type": "END"},
        ],
        "connections": [
            {"source_ref_id": start, "target_ref_id": odm},
            {"source_ref_id": odm, "target_ref_id": end},
        ],
    }


def _build_scenario_11(model_id: int, kb_ids: list[int], top_k: int, prompt_ids: dict) -> dict:
    """시작 → pLM(MODEL) → 종료 — ODM 단순과 위상 동일. 백엔드는 model_type(pLM)으로 추론 경로를 가른다."""
    start, plm, end = _ref("start"), _ref("plm"), _ref("end")
    return {
        "components": [
            {"ref_id": start, "name": "시작", "type": "START"},
            _odm_model_component(plm, "MODEL", model_id),
            {"ref_id": end, "name": "끝", "type": "END"},
        ],
        "connections": [
            {"source_ref_id": start, "target_ref_id": plm},
            {"source_ref_id": plm, "target_ref_id": end},
        ],
    }


def _build_scenario_2(model_id: int, kb_ids: list[int], top_k: int, prompt_ids: dict) -> dict:
    """시작 → KB → LLM → 종료"""
    start, kb, llm, end = _ref("start"), _ref("kb"), _ref("llm"), _ref("end")
    return {
        "components": [
            {"ref_id": start, "name": "시작", "type": "START"},
            _kb_component(kb, "Knowledge Base", kb_ids[0], top_k),
            _model_component(llm, "LLM", model_id),
            {"ref_id": end, "name": "끝", "type": "END"},
        ],
        "connections": [
            {"source_ref_id": start, "target_ref_id": kb},
            {"source_ref_id": kb, "target_ref_id": llm},
            {"source_ref_id": llm, "target_ref_id": end},
        ],
    }


def _build_scenario_3(model_id: int, kb_ids: list[int], top_k: int, prompt_ids: dict) -> dict:
    """시작 → LLM_A → LLM_B → 종료"""
    start, llm_a, llm_b, end = _ref("start"), _ref("llm-a"), _ref("llm-b"), _ref("end")
    return {
        "components": [
            {"ref_id": start, "name": "시작", "type": "START"},
            _model_component(llm_a, "LLM-A", model_id, prompt_ids.get("LLM-A")),
            _model_component(llm_b, "LLM-B", model_id, prompt_ids.get("LLM-B")),
            {"ref_id": end, "name": "끝", "type": "END"},
        ],
        "connections": [
            {"source_ref_id": start, "target_ref_id": llm_a},
            {"source_ref_id": llm_a, "target_ref_id": llm_b},
            {"source_ref_id": llm_b, "target_ref_id": end},
        ],
    }


def _build_scenario_4(model_id: int, kb_ids: list[int], top_k: int, prompt_ids: dict) -> dict:
    """시작 → KB → LLM_A → LLM_B → 종료"""
    start, kb, llm_a, llm_b, end = (
        _ref("start"),
        _ref("kb"),
        _ref("llm-a"),
        _ref("llm-b"),
        _ref("end"),
    )
    return {
        "components": [
            {"ref_id": start, "name": "시작", "type": "START"},
            _kb_component(kb, "Knowledge Base", kb_ids[0], top_k),
            _model_component(llm_a, "LLM-A", model_id, prompt_ids.get("LLM-A")),
            _model_component(llm_b, "LLM-B", model_id, prompt_ids.get("LLM-B")),
            {"ref_id": end, "name": "끝", "type": "END"},
        ],
        "connections": [
            {"source_ref_id": start, "target_ref_id": kb},
            {"source_ref_id": kb, "target_ref_id": llm_a},
            {"source_ref_id": llm_a, "target_ref_id": llm_b},
            {"source_ref_id": llm_b, "target_ref_id": end},
        ],
    }


def _build_scenario_5(model_id: int, kb_ids: list[int], top_k: int, prompt_ids: dict) -> dict:
    """시작 → LLM_A → KB → LLM_B → 종료"""
    start, llm_a, kb, llm_b, end = (
        _ref("start"),
        _ref("llm-a"),
        _ref("kb"),
        _ref("llm-b"),
        _ref("end"),
    )
    return {
        "components": [
            {"ref_id": start, "name": "시작", "type": "START"},
            _model_component(llm_a, "LLM-A (쿼리 정제)", model_id, prompt_ids.get("LLM-A (쿼리 정제)")),
            _kb_component(kb, "Knowledge Base", kb_ids[0], top_k),
            _model_component(llm_b, "LLM-B (최종 답변)", model_id, prompt_ids.get("LLM-B (최종 답변)")),
            {"ref_id": end, "name": "끝", "type": "END"},
        ],
        "connections": [
            {"source_ref_id": start, "target_ref_id": llm_a},
            {"source_ref_id": llm_a, "target_ref_id": kb},
            {"source_ref_id": kb, "target_ref_id": llm_b},
            {"source_ref_id": llm_b, "target_ref_id": end},
        ],
    }


def _build_scenario_6(model_id: int, kb_ids: list[int], top_k: int, prompt_ids: dict) -> dict:
    """시작 → [LLM1 / LLM2] → LLM3 → 종료"""
    start, llm1, llm2, llm3, end = (
        _ref("start"),
        _ref("llm-1"),
        _ref("llm-2"),
        _ref("llm-3"),
        _ref("end"),
    )
    return {
        "components": [
            {"ref_id": start, "name": "시작", "type": "START"},
            _model_component(llm1, "LLM-1 (분기A)", model_id, prompt_ids.get("LLM-1 (분기A)")),
            _model_component(llm2, "LLM-2 (분기B)", model_id, prompt_ids.get("LLM-2 (분기B)")),
            _model_component(llm3, "LLM-3 (합류)", model_id, prompt_ids.get("LLM-3 (합류)")),
            {"ref_id": end, "name": "끝", "type": "END"},
        ],
        "connections": [
            {"source_ref_id": start, "target_ref_id": llm1},
            {"source_ref_id": start, "target_ref_id": llm2},
            {"source_ref_id": llm1, "target_ref_id": llm3},
            {"source_ref_id": llm2, "target_ref_id": llm3},
            {"source_ref_id": llm3, "target_ref_id": end},
        ],
    }


def _build_scenario_7(model_id: int, kb_ids: list[int], top_k: int, prompt_ids: dict) -> dict:
    """시작 → [LLM1 / KB+LLM2] → LLM3 → 종료"""
    start, llm1, kb, llm2, llm3, end = (
        _ref("start"),
        _ref("llm-1"),
        _ref("kb"),
        _ref("llm-2"),
        _ref("llm-3"),
        _ref("end"),
    )
    return {
        "components": [
            {"ref_id": start, "name": "시작", "type": "START"},
            _model_component(llm1, "LLM-1 (분기A)", model_id, prompt_ids.get("LLM-1 (분기A)")),
            _kb_component(kb, "KB (분기B)", kb_ids[0], top_k),
            _model_component(llm2, "LLM-2 (분기B)", model_id, prompt_ids.get("LLM-2 (분기B)")),
            _model_component(llm3, "LLM-3 (합류)", model_id, prompt_ids.get("LLM-3 (합류)")),
            {"ref_id": end, "name": "끝", "type": "END"},
        ],
        "connections": [
            {"source_ref_id": start, "target_ref_id": llm1},
            {"source_ref_id": start, "target_ref_id": kb},
            {"source_ref_id": kb, "target_ref_id": llm2},
            {"source_ref_id": llm1, "target_ref_id": llm3},
            {"source_ref_id": llm2, "target_ref_id": llm3},
            {"source_ref_id": llm3, "target_ref_id": end},
        ],
    }


def _build_scenario_8(model_id: int, kb_ids: list[int], top_k: int, prompt_ids: dict) -> dict:
    """시작 → [KB1+LLM1 / KB2+LLM2] → LLM3 → 종료"""
    start, kb1, llm1, kb2, llm2, llm3, end = (
        _ref("start"),
        _ref("kb-1"),
        _ref("llm-1"),
        _ref("kb-2"),
        _ref("llm-2"),
        _ref("llm-3"),
        _ref("end"),
    )
    return {
        "components": [
            {"ref_id": start, "name": "시작", "type": "START"},
            _kb_component(kb1, "KB-1 (고압가스 안전관리법)", kb_ids[0], top_k),
            _model_component(llm1, "LLM-1 (분기A)", model_id, prompt_ids.get("LLM-1 (분기A)")),
            _kb_component(kb2, "KB-2 (건축서비스산업 진흥법)", kb_ids[1], top_k),
            _model_component(llm2, "LLM-2 (분기B)", model_id, prompt_ids.get("LLM-2 (분기B)")),
            _model_component(llm3, "LLM-3 (합류)", model_id, prompt_ids.get("LLM-3 (합류)")),
            {"ref_id": end, "name": "끝", "type": "END"},
        ],
        "connections": [
            {"source_ref_id": start, "target_ref_id": kb1},
            {"source_ref_id": kb1, "target_ref_id": llm1},
            {"source_ref_id": start, "target_ref_id": kb2},
            {"source_ref_id": kb2, "target_ref_id": llm2},
            {"source_ref_id": llm1, "target_ref_id": llm3},
            {"source_ref_id": llm2, "target_ref_id": llm3},
            {"source_ref_id": llm3, "target_ref_id": end},
        ],
    }


def _build_scenario_9(model_id: int, kb_ids: list[int], top_k: int, prompt_ids: dict) -> dict:
    """시작 → [LLM_A→KB_A / LLM_B→KB_B] → LLM_C → 종료"""
    start, llm_a, kb_a, llm_b, kb_b, llm_c, end = (
        _ref("start"),
        _ref("llm-a"),
        _ref("kb-a"),
        _ref("llm-b"),
        _ref("kb-b"),
        _ref("llm-c"),
        _ref("end"),
    )
    return {
        "components": [
            {"ref_id": start, "name": "시작", "type": "START"},
            _model_component(llm_a, "LLM-A (분기A 쿼리정제)", model_id, prompt_ids.get("LLM-A (분기A 쿼리정제)")),
            _kb_component(kb_a, "KB-A (고압가스 안전관리법)", kb_ids[0], top_k),
            _model_component(llm_b, "LLM-B (분기B 쿼리정제)", model_id, prompt_ids.get("LLM-B (분기B 쿼리정제)")),
            _kb_component(kb_b, "KB-B (건축서비스산업 진흥법)", kb_ids[1], top_k),
            _model_component(llm_c, "LLM-C (합류/최종답변)", model_id, prompt_ids.get("LLM-C (합류/최종답변)")),
            {"ref_id": end, "name": "끝", "type": "END"},
        ],
        "connections": [
            {"source_ref_id": start, "target_ref_id": llm_a},
            {"source_ref_id": llm_a, "target_ref_id": kb_a},
            {"source_ref_id": start, "target_ref_id": llm_b},
            {"source_ref_id": llm_b, "target_ref_id": kb_b},
            {"source_ref_id": kb_a, "target_ref_id": llm_c},
            {"source_ref_id": kb_b, "target_ref_id": llm_c},
            {"source_ref_id": llm_c, "target_ref_id": end},
        ],
    }


def _build_scenario_12(model_id: int, kb_ids: list[int], top_k: int, prompt_ids: dict) -> dict:
    """시작 → BFM(MODEL, base fill-mask) → 종료 — base 모델을 어댑터 없이 직접 서빙, task=fill-mask 로 추론 경로를 가른다."""
    start, bfm, end = _ref("start"), _ref("bfm"), _ref("end")
    return {
        "components": [
            {"ref_id": start, "name": "시작", "type": "START"},
            _odm_model_component(bfm, "MODEL", model_id),
            {"ref_id": end, "name": "끝", "type": "END"},
        ],
        "connections": [
            {"source_ref_id": start, "target_ref_id": bfm},
            {"source_ref_id": bfm, "target_ref_id": end},
        ],
    }


def _build_scenario_13(model_id: int, kb_ids: list[int], top_k: int, prompt_ids: dict) -> dict:
    """시작 → BFM(MODEL, structure-prediction) → 종료 — ESMFold2 구조예측 모델을 직접 서빙.

    task=protein-structure-prediction 로 추론 경로를 가른다.
    """
    start, bfm, end = _ref("start"), _ref("bfm"), _ref("end")
    return {
        "components": [
            {"ref_id": start, "name": "시작", "type": "START"},
            _odm_model_component(bfm, "MODEL", model_id),
            {"ref_id": end, "name": "끝", "type": "END"},
        ],
        "connections": [
            {"source_ref_id": start, "target_ref_id": bfm},
            {"source_ref_id": bfm, "target_ref_id": end},
        ],
    }


_BUILDERS = {
    1: _build_scenario_1,
    2: _build_scenario_2,
    3: _build_scenario_3,
    4: _build_scenario_4,
    5: _build_scenario_5,
    6: _build_scenario_6,
    7: _build_scenario_7,
    8: _build_scenario_8,
    9: _build_scenario_9,
    10: _build_scenario_10,
    11: _build_scenario_11,
    12: _build_scenario_12,
    13: _build_scenario_13,
}


# ── Public API ───────────────────────────────────────────────────────


def get_scenario(num: int) -> dict:
    """시나리오 메타데이터를 반환한다."""
    if num not in SCENARIOS:
        mx = max(SCENARIOS.keys())
        raise ValueError(f"시나리오 #{num} 은 존재하지 않습니다. (유효: 1~{mx})")
    return SCENARIOS[num]


def build_workflow_definition(
    num: int,
    model_id: int,
    kb_ids: list[int] | None = None,
    top_k: int = 3,
    prompt_ids: dict | None = None,
) -> dict:
    """시나리오 번호에 맞는 workflow_definition 을 생성한다."""
    scenario = get_scenario(num)
    ids = kb_ids or []
    kb_count = scenario["kb_count"]
    if kb_count > 0 and len(ids) < kb_count:
        raise ValueError(
            f"시나리오 #{num} ({scenario['name']}) 은 KB {kb_count}개가 필요합니다. " f"(제공된 kb_ids: {len(ids)}개)"
        )
    definition = _BUILDERS[num](model_id, ids, top_k, prompt_ids or {})
    # x, y 캔버스 좌표 자동 부여 — 짝수 인덱스는 양수 y, 홀수는 음수 y 로 음수 허용도 함께 검증한다.
    for i, comp in enumerate(definition["components"]):
        comp["x"] = i * 200
        comp["y"] = 50 if i % 2 == 0 else -50
    return definition


def state_key_wf(num: int) -> str:
    """시나리오별 workflow_id 상태 키 (레거시 단일 배포; deployments 로 이전됨)"""
    return f"scenario_{num}_workflow_id"


def state_key_kb_ids(num: int) -> str:
    """시나리오별 kb_ids 상태 키 (JSON list, 레거시)"""
    return f"scenario_{num}_kb_ids"


def state_key_prompts(num: int) -> str:
    """시나리오별 prompt_ids 상태 키 (JSON dict, 레거시)"""
    return f"scenario_{num}_prompt_ids"


def state_key_deployments(num: int) -> str:
    """시나리오별 배포 기록 목록 (JSON list of deployment dict)"""
    return f"scenario_{num}_deployments"


def load_deployment_entries(num: int) -> list[dict]:
    """시나리오별 저장된 배포 기록을 읽는다. 레거시 단일 키만 있으면 1건으로 승격한다."""
    from config import load_state

    raw = load_state(state_key_deployments(num))
    if raw:
        return json.loads(raw)

    wf = load_state(state_key_wf(num))
    if not wf:
        return []
    kb_raw = load_state(state_key_kb_ids(num)) or "[]"
    pr_raw = load_state(state_key_prompts(num)) or "{}"
    return [
        {
            "workflow_id": wf,
            "kb_ids": json.loads(kb_raw),
            "prompt_ids": json.loads(pr_raw),
            "workflow_name": "",
        }
    ]


def save_deployment_entries(num: int, entries: list[dict]) -> None:
    """배포 기록 전체를 저장하고, 레거시 단일 키는 제거한다."""
    from config import clear_state, save_state

    key_dep = state_key_deployments(num)
    if entries:
        save_state(key_dep, json.dumps(entries))
    else:
        clear_state(key_dep)
    clear_state(state_key_wf(num))
    clear_state(state_key_kb_ids(num))
    clear_state(state_key_prompts(num))


def append_deployment_entry(num: int, entry: dict) -> None:
    """배포 1건을 기존 목록 끝에 추가한다 (이전 배포와 kb/프롬프트를 덮어쓰지 않음)."""
    entries = load_deployment_entries(num)
    entries.append(entry)
    save_deployment_entries(num, entries)


def remove_deployment_entry_by_workflow_id(num: int, workflow_id: str) -> None:
    """삭제 완료 후 해당 workflow_id 인 기록만 목록에서 제거한다."""
    entries = [e for e in load_deployment_entries(num) if e.get("workflow_id") != workflow_id]
    save_deployment_entries(num, entries)


# ── CLI: 시나리오 정보 출력 ──────────────────────────────────────────


def _print_scenario(num: int, s: dict) -> None:
    kb_count = s["kb_count"]
    if kb_count == 0:
        kb_label = "아니오"
    else:
        labels = s.get("kb_labels", [])
        kb_label = f"{kb_count}개 ({', '.join(labels)})"
    print(f"  #{num}  {s['name']}")
    print(f"      구성: {s['graph']}")
    print(f"      KB: {kb_label}")
    prompts = s.get("prompts", {})
    if prompts:
        print("      LLM 프롬프트:")
        for comp_name, p in prompts.items():
            ctx_mark = " ({context} 포함)" if p["has_context"] else ""
            print(f"        - {comp_name}: {p['name']}{ctx_mark}")
    else:
        print("      LLM 프롬프트: 없음 (기본 설정)")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERROR: SCENARIO 번호를 지정하세요.")
        mx = max(SCENARIOS.keys())
        print(f"사용법: python workflow/definitions.py <1~{mx}>")
        print()
        print("시나리오 목록:")
        for n, s in SCENARIOS.items():
            print(f"  #{n}  {s['name']}  —  {s['graph']}")
        sys.exit(1)

    num = int(sys.argv[1])
    if num not in SCENARIOS:
        mx = max(SCENARIOS.keys())
        print(f"ERROR: 시나리오 #{num} 은 존재하지 않습니다. (유효: 1~{mx})")
        sys.exit(1)

    print(f"\n=== 시나리오 #{num} 상세 ===\n")
    _print_scenario(num, SCENARIOS[num])
