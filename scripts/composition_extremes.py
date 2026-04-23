"""
Resolve which workflow construction indices are the grouped vs ungrouped extremes.

Used by analysis scripts that compare ``Const <n>`` designs. Values are taken from
``total_groups`` in result metrics: **most grouped** = smallest count (ties: lowest
composition id); **most ungrouped** = largest count (ties: highest id). If all
compositions share the same ``total_groups``, this is equivalent to
``(min id, max id)`` over the data.

Each composition may be stored as a list of per-run dicts (e.g. per failure rate
or per target job length) or a single dict.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

CompMetrics = Union[List[Dict[str, Any]], Dict[str, Any]]
# One metrics dict per composition (typical for a single fr0 / one scenario dir).
SingleMap = Dict[int, Dict[str, Any]]


def _per_composition_rows(comp: CompMetrics) -> List[Dict[str, Any]]:
    if isinstance(comp, list):
        return comp
    if isinstance(comp, dict):
        return [comp]
    return []


def canonical_total_groups(entries: List[Dict[str, Any]]) -> int:
    """Return one ``total_groups`` for a composition, preferring the fr0 run."""
    if not entries:
        return 0
    for e in sorted(entries, key=lambda x: x.get("failure_rate", 0.0)):
        if abs(e.get("failure_rate", 0.0) - 0.0) < 0.1:
            return int(e.get("total_groups", 0))
    first = min(entries, key=lambda x: x.get("failure_rate", 0.0))
    return int(first.get("total_groups", 0))


def composition_extremes(
    data_by_composition: Dict[int, CompMetrics],
) -> Tuple[int, int]:
    """Return ``(most_grouped, most_ungrouped)`` composition numbers.

    Args:
        data_by_composition: composition id -> list of metric dicts, or one dict

    Raises:
        ValueError: if the mapping is empty
    """
    keys = sorted(data_by_composition.keys())
    if not keys:
        raise ValueError(
            "data_by_composition is empty; cannot determine composition extremes"
        )
    tgroups = {
        c: canonical_total_groups(_per_composition_rows(data_by_composition[c]))
        for c in keys
    }
    min_g = min(tgroups.values())
    max_g = max(tgroups.values())
    grouped = min(c for c in keys if tgroups[c] == min_g)
    indep = max(c for c in keys if tgroups[c] == max_g)
    return (grouped, indep)


def composition_extremes_from_single_map(data_by_composition: SingleMap) -> Tuple[int, int]:
    """Run :func:`composition_extremes` on one dict of metrics per composition id."""
    if not data_by_composition:
        raise ValueError(
            "data_by_composition is empty; cannot determine composition extremes"
        )
    return composition_extremes({k: [v] for k, v in data_by_composition.items()})
