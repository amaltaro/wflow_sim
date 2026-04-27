"""Plot an overview of task grouping for all workflow constructions in a summary JSON.

This produces a compact "barcode" figure: each row is a construction (composition),
each column is a taskset index, and the color encodes the group id assigned to that
taskset in that construction.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


_TASKSET_RE = re.compile(r"^Taskset(\d+)$")


@dataclass(frozen=True)
class ConstructionGroupsMatrix:
    """Matrix representation of group assignments per composition."""

    template_name: str
    composition_numbers: List[int]
    taskset_names: List[str]
    group_id_matrix: List[List[str | None]]  # [composition_idx][task_idx] -> group_id


def _taskset_index(taskset_name: str) -> int:
    match = _TASKSET_RE.match(taskset_name)
    if not match:
        raise ValueError(f"Unexpected taskset name: {taskset_name!r}")
    return int(match.group(1))


def load_compositions_summary(path: Path) -> Dict[str, Any]:
    """Load a compositions summary JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_group_id_matrix(summary: Mapping[str, Any]) -> ConstructionGroupsMatrix:
    """Build a (composition x taskset) matrix of group ids from a summary JSON structure."""
    template_name = str(summary.get("template_name", "workflow"))
    compositions = summary.get("compositions")
    if not isinstance(compositions, list) or not compositions:
        raise ValueError("Summary JSON missing non-empty 'compositions' list")

    # Determine taskset universe from group_details across compositions.
    taskset_indices: set[int] = set()
    for comp in compositions:
        group_details = comp.get("group_details", [])
        if not isinstance(group_details, list):
            raise ValueError("Each composition must have 'group_details' as a list")
        for group in group_details:
            tasks = group.get("tasks", [])
            if not isinstance(tasks, list):
                raise ValueError("Each group_detail must have 'tasks' as a list")
            for t in tasks:
                taskset_indices.add(_taskset_index(str(t)))

    if not taskset_indices:
        raise ValueError("No tasksets found in summary JSON")

    max_taskset = max(taskset_indices)
    taskset_names = [f"Taskset{i}" for i in range(1, max_taskset + 1)]

    composition_numbers: List[int] = []
    group_id_matrix: List[List[str | None]] = []

    for comp in compositions:
        comp_num = int(comp.get("composition_number"))
        composition_numbers.append(comp_num)

        group_by_task: Dict[str, str] = {}
        for group in comp.get("group_details", []):
            gid = str(group.get("group_id"))
            for t in group.get("tasks", []):
                task_name = str(t)
                if task_name in group_by_task and group_by_task[task_name] != gid:
                    raise ValueError(
                        f"Task {task_name} assigned to multiple groups in composition {comp_num}"
                    )
                group_by_task[task_name] = gid

        group_row: List[str | None] = []
        for task_name in taskset_names:
            group_row.append(group_by_task.get(task_name))
        group_id_matrix.append(group_row)

    return ConstructionGroupsMatrix(
        template_name=template_name,
        composition_numbers=composition_numbers,
        taskset_names=taskset_names,
        group_id_matrix=group_id_matrix,
    )


def _stable_unique(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _to_numeric_matrix(
    group_id_matrix: Sequence[Sequence[str | None]],
) -> Tuple[List[List[int]], List[str]]:
    group_ids = _stable_unique(
        gid for row in group_id_matrix for gid in row if gid is not None
    )
    gid_to_int = {gid: idx for idx, gid in enumerate(group_ids)}
    missing_value = len(group_ids)  # last color reserved for "missing"

    numeric: List[List[int]] = []
    for row in group_id_matrix:
        numeric.append([gid_to_int.get(gid, missing_value) for gid in row])
    return numeric, group_ids


def _group_runs(row: Sequence[str | None]) -> List[Tuple[int, int, str]]:
    """Return contiguous (start_idx, end_idx_exclusive, group_id) runs for one composition row."""
    runs: List[Tuple[int, int, str]] = []
    start = 0
    current = row[0]
    for idx in range(1, len(row) + 1):
        value = row[idx] if idx < len(row) else None
        if value != current:
            if current is not None:
                runs.append((start, idx, str(current)))
            start = idx
            current = value
    return runs


def _relative_luminance(rgba: Tuple[float, float, float, float]) -> float:
    r, g, b, _ = rgba
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def plot_group_overview(
    matrix: ConstructionGroupsMatrix,
    output_dir: Path,
    *,
    filename: str | None = None,
    dpi: int = 200,
    show_colorbar: bool = False,
) -> Path:
    """Render and save the overview figure."""
    # Import here so pure-parsing tests don't require matplotlib.
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    output_dir.mkdir(parents=True, exist_ok=True)

    numeric, group_ids = _to_numeric_matrix(matrix.group_id_matrix)
    n_groups = len(group_ids)

    base_cmap = plt.get_cmap("tab20", max(n_groups, 1))
    colors = [base_cmap(i) for i in range(n_groups)]
    colors.append((0.85, 0.85, 0.85, 1.0))  # missing / unknown
    cmap = ListedColormap(colors)

    fig_w = max(6.0, 0.9 * len(matrix.taskset_names))
    fig_h = max(4.0, 0.28 * len(matrix.composition_numbers))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(numeric, aspect="auto", interpolation="nearest", cmap=cmap)

    ax.set_title("Taskset grouping for all compositions")
    ax.set_xlabel("Taskset")
    ax.set_ylabel("Composition")

    ax.set_xticks(
        range(len(matrix.taskset_names)),
        labels=matrix.taskset_names,
        rotation=45,
        ha="right",
    )
    ax.set_yticks(
        range(len(matrix.composition_numbers)),
        labels=[str(n) for n in matrix.composition_numbers],
    )

    # Gridlines separating compositions (rows).
    n_rows = len(matrix.composition_numbers)
    n_cols = len(matrix.taskset_names)
    ax.set_yticks([y - 0.5 for y in range(1, n_rows)], minor=True)
    ax.grid(which="minor", axis="y", color="black", linewidth=0.6, alpha=0.5)
    ax.tick_params(which="minor", left=False)

    # Per-row group boundaries (vertical lines only where group changes).
    for row_idx, row in enumerate(matrix.group_id_matrix):
        for col_idx in range(1, n_cols):
            if row[col_idx] != row[col_idx - 1]:
                x = col_idx - 0.5
                ax.vlines(
                    x,
                    row_idx - 0.5,
                    row_idx + 0.5,
                    colors="black",
                    linewidth=1.2,
                    alpha=0.8,
                )

    # Labels inside merged blocks (one label per contiguous group run).
    gid_to_color = {gid: colors[i] for i, gid in enumerate(group_ids)}
    for row_idx, row in enumerate(matrix.group_id_matrix):
        for start, end, gid in _group_runs(row):
            x_center = (start + end - 1) / 2.0
            rgba = gid_to_color.get(gid, (1.0, 1.0, 1.0, 1.0))
            text_color = "black" if _relative_luminance(rgba) > 0.6 else "white"
            ax.text(
                x_center,
                row_idx,
                gid,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=text_color,
            )

    if show_colorbar:
        # Colorbar with group ids (skip missing marker).
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        ticks = list(range(n_groups))
        cbar.set_ticks(ticks, labels=[group_ids[i] for i in ticks])

    fig.tight_layout()

    out_name = filename or f"{matrix.template_name}_construction_groups_overview.png"
    out_path = output_dir / out_name
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot an overview figure of workflow construction groupings from a summary JSON."
    )
    parser.add_argument(
        "--summary-json",
        "-s",
        type=Path,
        required=True,
        help="Path to the compositions summary JSON (e.g. seq_real_compositions_summary.json)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Output directory for the plot",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI (default: 200)")
    parser.add_argument(
        "--show-colorbar",
        action="store_true",
        help="Show the colorbar legend (default: hidden because group labels are drawn inside blocks)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = load_compositions_summary(args.summary_json)
    matrix = build_group_id_matrix(summary)
    out_path = plot_group_overview(
        matrix,
        args.output_dir,
        dpi=args.dpi,
        show_colorbar=bool(args.show_colorbar),
    )
    print(f"=> Wrote overview figure to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

