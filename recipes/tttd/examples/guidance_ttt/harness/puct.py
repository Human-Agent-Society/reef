from __future__ import annotations

import math

from .state import LibraryNode

PUCT_Q_BLEND = "blended"
PUCT_Q_BEST_CHILD = "best_child"
SUPPORTED_PUCT_Q_MODES = frozenset({PUCT_Q_BLEND, PUCT_Q_BEST_CHILD})
OWN_VALUE_Q_WEIGHT = 0.8
REACHABLE_VALUE_Q_WEIGHT = 0.2


def normalize_puct_q_mode(mode: str | None) -> str:
    normalized = str(mode or PUCT_Q_BLEND).strip().lower()
    if normalized not in SUPPORTED_PUCT_Q_MODES:
        supported = ", ".join(sorted(SUPPORTED_PUCT_Q_MODES))
        raise ValueError(f"Unsupported PUCT Q mode {mode!r}; expected one of: {supported}")
    return normalized


def compute_scale(nodes: list[LibraryNode], *, initial_ids: set[str] | None = None) -> float:
    """Reward scale used by TTT-Discover's PUCT sampler."""
    if not nodes:
        return 1.0
    candidates = [node for node in nodes if initial_ids is None or node.id not in initial_ids]
    if not candidates:
        candidates = nodes
    values = [float(node.value) for node in candidates]
    return max(max(values) - min(values), 1e-6)


def rank_priors(nodes: list[LibraryNode]) -> dict[str, float]:
    """Rank-based prior P(i), matching discover's descending-value rank weights."""
    if not nodes:
        return {}
    ranked = sorted(enumerate(nodes), key=lambda item: float(item[1].value), reverse=True)
    weights_by_index: dict[int, float] = {}
    total = 0.0
    n_nodes = len(nodes)
    for rank, (index, _node) in enumerate(ranked):
        weight = float(n_nodes - rank)
        weights_by_index[index] = weight
        total += weight
    return {node.id: weights_by_index[index] / total for index, node in enumerate(nodes)}


def archive_puct_score(
    *,
    node: LibraryNode,
    visit_count: int,
    best_reachable_value: float | None,
    prior: float,
    scale: float,
    total_visits: int,
    puct_c: float,
    q_mode: str = PUCT_Q_BLEND,
) -> float:
    """Guidance-TTT archive PUCT score.

    score(i) = Q(i) + c * scale * P(i) * sqrt(1 + T) / (1 + n[i])
    Q(i) = R(i) if n[i] == 0
    Q(i) = m[i] if q_mode == "best_child" and n[i] > 0 and m[i] exists
    Q(i) = 0.8 * R(i) + 0.2 * m[i] otherwise
    """
    q_value = _q_value(
        node=node,
        visit_count=visit_count,
        best_reachable_value=best_reachable_value,
        q_mode=q_mode,
    )
    bonus = (
        float(puct_c) * float(scale) * float(prior) * math.sqrt(1.0 + float(total_visits)) / (1.0 + float(visit_count))
    )
    return q_value + bonus


def _q_value(
    *,
    node: LibraryNode,
    visit_count: int,
    best_reachable_value: float | None,
    q_mode: str = PUCT_Q_BLEND,
) -> float:
    own_value = float(node.value)
    if visit_count <= 0 or best_reachable_value is None:
        return own_value
    if normalize_puct_q_mode(q_mode) == PUCT_Q_BEST_CHILD:
        return float(best_reachable_value)
    return OWN_VALUE_Q_WEIGHT * own_value + REACHABLE_VALUE_Q_WEIGHT * float(best_reachable_value)


def rank_archive_nodes(
    nodes: list[LibraryNode],
    *,
    initial_ids: set[str],
    visit_counts: dict[str, int],
    best_reachable_values: dict[str, float],
    total_visits: int,
    puct_c: float,
    q_mode: str = PUCT_Q_BLEND,
) -> list[tuple[float, float, LibraryNode, int, float, float, float]]:
    """Return nodes sorted by Guidance-TTT archive PUCT score.

    Tuple layout mirrors discover's logging order:
    (score, value, node, n, Q, P, bonus).
    """
    scale = compute_scale(nodes, initial_ids=initial_ids)
    priors = rank_priors(nodes)
    scored = []
    for node in nodes:
        n_visits = int(visit_counts.get(node.id, 0))
        q_value = _q_value(
            node=node,
            visit_count=n_visits,
            best_reachable_value=best_reachable_values.get(node.id),
            q_mode=q_mode,
        )
        prior = float(priors.get(node.id, 0.0))
        bonus = float(puct_c) * scale * prior * math.sqrt(1.0 + float(total_visits)) / (1.0 + float(n_visits))
        score = q_value + bonus
        scored.append((score, float(node.value), node, n_visits, q_value, prior, bonus))
    scored.sort(key=lambda item: (item[0], item[1], item[2].id), reverse=True)
    return scored
