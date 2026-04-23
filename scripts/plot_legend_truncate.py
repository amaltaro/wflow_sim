"""
Legend trimming for analysis plots: cap height when many series are present.

``apply_truncated_legend`` trims by **composition id** (parsed from labels) so low/high
const numbers stay visible, and always shows most grouped and most ungrouped. Multiple
lines per composition (e.g. Read/Write) are kept or dropped together.

``apply_truncated_construction_legend`` trims by **legend row index** (one row per
composition, order 0..n-1) for plots where that matches sorted composition id.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from matplotlib.lines import Line2D

_MAX_LEGEND_FULL = 30
_MAX_LEGEND_AFTER_TRUNCATE = 29
_HEAD_COMPS = 14
_TAIL_COMPS = 15

_CONST = re.compile(r"^Const (\d+)")
_BEST = re.compile(r"Best [Hh]ybrid \(C(\d+)\)")


def parse_legend_comp(label: str) -> Optional[int]:
    """Return composition id from a legend label, or None if it does not refer to a const."""
    s = label.strip()
    m = _CONST.match(s)
    if m:
        return int(m.group(1))
    m2 = _BEST.search(s)
    if m2:
        return int(m2.group(1))
    return None


def _lines_for_comps(
    by_comp: Dict[int, List[int]], comps: Set[int]
) -> int:
    return sum(len(by_comp[c]) for c in comps if c in by_comp)


def _legend_kw(
    bbox: Optional[Tuple[float, float]],
    loc: str,
    ncol: int,
    fontsize: int,
) -> dict:
    kw: dict = {"loc": loc, "ncol": ncol, "fontsize": fontsize}
    if bbox is not None:
        kw["bbox_to_anchor"] = bbox
    return kw


def apply_truncated_legend(
    ax: Any,
    grouped_comp: int,
    independent_comp: int,
    max_full: int = _MAX_LEGEND_FULL,
    max_after_truncate: int = _MAX_LEGEND_AFTER_TRUNCATE,
    bbox: Optional[Tuple[float, float]] = (1.05, 1),
    loc: str = "upper left",
    ncol: int = 1,
    fontsize: int = 9,
) -> None:
    """Set legend, trimming the middle (by composition id) when over ``max_full`` lines.

    Pass ``bbox=None`` to use ``loc`` only (e.g. bar charts).
    """
    handles, labels = ax.get_legend_handles_labels()
    n = len(labels)
    lkw = _legend_kw(bbox, loc, ncol, fontsize)
    if n <= max_full:
        ax.legend(handles, labels, **lkw)
        return

    by_comp: Dict[int, List[int]] = defaultdict(list)
    unassigned: List[int] = []
    for i, lab in enumerate(labels):
        c = parse_legend_comp(lab)
        if c is None:
            unassigned.append(i)
        else:
            by_comp[c].append(i)
    for v in by_comp.values():
        v.sort()

    unique_sorted = sorted(by_comp.keys())
    must_comp: Set[int] = {
        c for c in (grouped_comp, independent_comp) if c in by_comp
    }

    n_un = len(unassigned)
    ncomp = len(unique_sorted)
    kset: Set[int] = set()
    for c in unique_sorted[: min(_HEAD_COMPS, ncomp)]:
        kset.add(c)
    for c in unique_sorted[max(0, ncomp - _TAIL_COMPS) :]:
        kset.add(c)
    kset |= must_comp

    while (
        unique_sorted
        and n_un + _lines_for_comps(by_comp, kset) > max_after_truncate
    ):
        rem = [c for c in kset if c not in must_comp]
        if not rem:
            break
        lo, hi = min(unique_sorted), max(unique_sorted)
        mid = (lo + hi) / 2.0
        kset.remove(min(rem, key=lambda c: (abs(c - mid), -c)))

    keep_idx: Set[int] = set(unassigned)
    for c in kset:
        keep_idx.update(by_comp.get(c, []))

    new_h, new_l = _build_legend_with_ellipsis_gaps(handles, labels, keep_idx)
    ax.legend(new_h, new_l, **lkw)


def apply_truncated_construction_legend(
    ax: Any,
    n_legend_entries: int,
    must_idx: set,
    bbox: Optional[Tuple[float, float]] = (1.05, 1),
    loc: str = "upper left",
    ncol: int = 1,
    fontsize: int = 9,
) -> None:
    """Index-based trim when each legend row is one composition in order 0..n-1."""
    handles, labels = ax.get_legend_handles_labels()
    lkw = _legend_kw(bbox, loc, ncol, fontsize)
    if n_legend_entries <= _MAX_LEGEND_FULL or len(labels) != n_legend_entries:
        ax.legend(handles, labels, **lkw)
        return
    show = _trim_legend_index_set(n_legend_entries, must_idx)
    new_h, new_l = _build_legend_with_ellipsis_gaps(handles, labels, show)
    ax.legend(new_h, new_l, **lkw)


def _trim_legend_index_set(
    n: int,
    must: set,
    head: int = _HEAD_COMPS,
    tail: int = _TAIL_COMPS,
    max_real: int = _MAX_LEGEND_AFTER_TRUNCATE,
) -> set:
    if n == 0:
        return set()
    base: set = set()
    base |= {i for i in range(min(head, n))}
    base |= {i for i in range(max(0, n - tail), n)}
    base |= set(must)
    base = {i for i in base if 0 <= i < n}
    while len(base) > max_real:
        removable = [i for i in base if i not in must]
        if not removable:
            break
        mid = (n - 1) / 2.0
        to_drop = min(removable, key=lambda j: (abs(j - mid), -j))
        base.remove(to_drop)
    return base


def _build_legend_with_ellipsis_gaps(
    handles: List,
    labels: List[str],
    show_idx: Set[int],
) -> Tuple[List, List[str]]:
    """Keep rows whose indices are in ``show_idx``; add ellipsis for gaps in index order."""
    ordered = sorted(show_idx)
    new_h: List = []
    new_l: List[str] = []
    ellipsis = Line2D(
        [0], [0], linestyle="None", marker="None", color="none", linewidth=0, label=""
    )
    for k, idx in enumerate(ordered):
        if k > 0 and idx - ordered[k - 1] > 1:
            gap = idx - ordered[k - 1] - 1
            new_h.append(ellipsis)
            new_l.append(f"… ({gap} omitted) …")
        new_h.append(handles[idx])
        new_l.append(labels[idx])
    return new_h, new_l
