"""
§7.3.2: 동일 워크플로 내 MODEL 컴포넌트 확정 순서.

START/END를 제외한 컴포넌트 그래프로 위상 정렬 후, MODEL만 순서를 추출한다.
순환 시 설계서대로 MODEL은 component.id 사전순으로 처리한다.
"""

from __future__ import annotations

from collections import defaultdict, deque

from db.models.service import ComponentType, Workflow, WorkflowComponent


def topological_order_model_components(workflow: Workflow) -> list[WorkflowComponent]:
    """
    MODEL + model_id 있는 컴포넌트만, 워크플로 의존성(비-MODEL 경유 간선 포함) 순서.
    동일 깊이(동시 ready)는 component.id 사전순.
    """
    models = [c for c in workflow.components if c.type == ComponentType.MODEL and c.model_id]
    if not models:
        return []
    if len(models) == 1:
        return models

    skip = {ComponentType.START, ComponentType.END}
    comps = [c for c in workflow.components if c.type not in skip]
    comp_ids = {c.id for c in comps}

    out_edges: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = {c.id: 0 for c in comps}

    for conn in workflow.component_connections:
        s, t = conn.source_component_id, conn.target_component_id
        if s in comp_ids and t in comp_ids:
            out_edges[s].append(t)
            in_degree[t] += 1

    ready = deque(sorted([c for c in comps if in_degree[c.id] == 0], key=lambda x: x.id))
    order_all: list[WorkflowComponent] = []

    while ready:
        u = ready.popleft()
        order_all.append(u)
        newly: list[WorkflowComponent] = []
        for v in sorted(out_edges[u.id]):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                vc = next(c for c in comps if c.id == v)
                newly.append(vc)
        newly.sort(key=lambda x: x.id)
        ready.extend(newly)

    if len(order_all) != len(comps):
        return sorted(models, key=lambda x: x.id)

    pos = {c.id: i for i, c in enumerate(order_all)}
    return sorted(models, key=lambda m: (pos[m.id], m.id))
