#!/usr/bin/env python3
"""Real (normalized) vs simulated I/O comparison for StepChain and TaskChain."""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from real_workflow_visualization import (  # noqa: E402
    extract_construction_number,
    transform_real_data_to_simulation_format,
)
from workflow_visualization import (  # noqa: E402
    STACKED_COMPARISON_FIG_H_IN,
    STACKED_COMPARISON_FIG_W_IN,
    _stacked_total_volume_scale_and_unit_from_max_mb,
    _style_stacked_total_data_volume_axis,
    find_simulation_files,
)

# StepChain (const 1) and TaskChain (const 16)
REAL_VS_SIM_COMPOSITIONS: Tuple[Tuple[int, str], ...] = (
    (1, "StepChain"),
    (16, "TaskChain"),
)

# Extra height vs standard I/O stacked plots to fit two legend rows below the figure.
REAL_VS_SIM_FIG_H_EXTRA_IN = 1.25

# Real/sim pair: shared center spacing and stacked bar width on both panels.
REAL_SIM_CLUSTER_SEP = 0.30
STACKED_BAR_WIDTH = 0.22


def _real_sim_x_centers(x_idx: float, cluster_sep: float = REAL_SIM_CLUSTER_SEP) -> Tuple[float, float]:
    """X centers for the real (left) and simulated (right) bar groups."""
    half = cluster_sep / 2.0
    return x_idx - half, x_idx + half


def _per_event_bar_width(n_metrics: int) -> float:
    """Bar width so n_metrics abutting bars span exactly ``STACKED_BAR_WIDTH``."""
    return STACKED_BAR_WIDTH / n_metrics

IO_PATTERN_COLORS: Dict[str, str] = {
    "Local Read": "#1f77b4",
    "Remote Read": "#ff7f0e",
    "Local Write": "#2ca02c",
    "Remote Write": "#d62728",
}


@dataclass(frozen=True)
class IoVolumeMetrics:
    """Per-event and total I/O volumes in MB (same fields as ``plot_io_patterns``)."""

    read_local_pevt: float
    read_remote_pevt: float
    write_local_pevt: float
    write_remote_pevt: float
    total_read_local_mb: float
    total_read_remote_mb: float
    total_write_local_mb: float
    total_write_remote_mb: float


def io_volumes_from_workflow(workflow: Dict[str, Any]) -> IoVolumeMetrics:
    """Extract I/O fields from a workflow dict (simulation or transformed real metrics)."""
    return IoVolumeMetrics(
        read_local_pevt=float(workflow.get("total_read_local_mb_per_event", 0.0)),
        read_remote_pevt=float(workflow.get("total_read_remote_mb_per_event", 0.0)),
        write_local_pevt=float(workflow.get("total_write_local_mb_per_event", 0.0)),
        write_remote_pevt=float(workflow.get("total_write_remote_mb_per_event", 0.0)),
        total_read_local_mb=float(workflow.get("total_read_local_mb", 0.0)),
        total_read_remote_mb=float(workflow.get("total_read_remote_mb", 0.0)),
        total_write_local_mb=float(workflow.get("total_write_local_mb", 0.0)),
        total_write_remote_mb=float(workflow.get("total_write_remote_mb", 0.0)),
    )


def _real_vs_sim_source_style() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Real: opaque fill + hatch; simulated: standard solid I/O colors (``plot_io_patterns``)."""
    return (
        {
            "label": "Real (normalized)",
            "alpha": 1.0,
            "hatch": "//",
            "edgecolor": "#444444",
            "linewidth": 0.6,
        },
        {"label": "Simulated", "edgecolor": "white", "linewidth": 0.6},
    )


def _clustered_per_event_bars(
    ax: plt.Axes,
    x_idx: int,
    real: IoVolumeMetrics,
    sim: IoVolumeMetrics,
    *,
    include_local_read: bool,
    cluster_sep: float,
    bar_w: float,
) -> None:
    """Draw real vs sim per-event I/O bars aligned with stacked bars below."""
    real_style, sim_style = _real_vs_sim_source_style()
    x_real, x_sim = _real_sim_x_centers(x_idx, cluster_sep)

    if include_local_read:
        series = (
            ("Local Read", real.read_local_pevt, sim.read_local_pevt),
            ("Remote Read", real.read_remote_pevt, sim.read_remote_pevt),
            ("Local Write", real.write_local_pevt, sim.write_local_pevt),
            ("Remote Write", real.write_remote_pevt, sim.write_remote_pevt),
        )
        offsets = (-1.5 * bar_w, -0.5 * bar_w, 0.5 * bar_w, 1.5 * bar_w)
    else:
        series = (
            ("Remote Read", real.read_remote_pevt, sim.read_remote_pevt),
            ("Local Write", real.write_local_pevt, sim.write_local_pevt),
            ("Remote Write", real.write_remote_pevt, sim.write_remote_pevt),
        )
        offsets = (-bar_w, 0.0, bar_w)

    for off, (name, rv, sv) in zip(offsets, series):
        color = IO_PATTERN_COLORS[name]
        ax.bar(x_real + off, rv, bar_w, color=color, **real_style)
        ax.bar(x_sim + off, sv, bar_w, color=color, **sim_style)


def _clustered_stacked_totals(
    ax: plt.Axes,
    x_idx: int,
    real: IoVolumeMetrics,
    sim: IoVolumeMetrics,
    *,
    include_local_read: bool,
    cluster_sep: float,
    stack_w: float,
    vol_scale: float,
) -> None:
    """Side-by-side stacked total-volume bars (real vs sim) at one construction."""
    real_style, sim_style = _real_vs_sim_source_style()
    x_real, x_sim = _real_sim_x_centers(x_idx, cluster_sep)

    if include_local_read:
        segments = (
            ("Local Read", real.total_read_local_mb, sim.total_read_local_mb),
            ("Remote Read", real.total_read_remote_mb, sim.total_read_remote_mb),
            ("Local Write", real.total_write_local_mb, sim.total_write_local_mb),
            ("Remote Write", real.total_write_remote_mb, sim.total_write_remote_mb),
        )
    else:
        segments = (
            ("Remote Read", real.total_read_remote_mb, sim.total_read_remote_mb),
            ("Local Write", real.total_write_local_mb, sim.total_write_local_mb),
            ("Remote Write", real.total_write_remote_mb, sim.total_write_remote_mb),
        )

    bottom_r = 0.0
    bottom_s = 0.0
    for name, rv, sv in segments:
        color = IO_PATTERN_COLORS[name]
        ax.bar(x_real, rv * vol_scale, stack_w, bottom=bottom_r, color=color, **real_style)
        ax.bar(x_sim, sv * vol_scale, stack_w, bottom=bottom_s, color=color, **sim_style)
        bottom_r += rv * vol_scale
        bottom_s += sv * vol_scale


def _io_legend_patch(name: str) -> Patch:
    return Patch(facecolor=IO_PATTERN_COLORS[name], label=name)


def _real_vs_sim_io_legend_below(fig: plt.Figure, *, include_local_read: bool) -> None:
    """Two legend rows below the bottom panel (matplotlib ncol fills by column, not row)."""
    legend_kw: Dict[str, Any] = {
        "frameon": True,
        "fancybox": True,
        "framealpha": 0.95,
        "loc": "outside lower center",
    }
    real_patch = Patch(
        facecolor="#bbbbbb",
        edgecolor="#444444",
        hatch="//",
        alpha=0.9,
        label="Real (normalized)",
    )
    sim_patch = Patch(facecolor="#bbbbbb", edgecolor="#444444", alpha=1.0, label="Simulated")

    if include_local_read:
        io_handles = [
            _io_legend_patch("Local Read"),
            _io_legend_patch("Remote Read"),
            _io_legend_patch("Local Write"),
            _io_legend_patch("Remote Write"),
        ]
        io_ncol = 4
    else:
        io_handles = [
            _io_legend_patch("Remote Read"),
            _io_legend_patch("Local Write"),
            _io_legend_patch("Remote Write"),
        ]
        io_ncol = 3

    source_handles = [real_patch, sim_patch]

    # Anchors in figure coordinates (below bottom axes); tight spacing between rows.
    leg_io = fig.legend(handles=io_handles, ncol=io_ncol, bbox_to_anchor=(0.5, -0.05), **legend_kw)
    fig.add_artist(leg_io)
    fig.legend(handles=source_handles, ncol=2, bbox_to_anchor=(0.5, -0.085), **legend_kw)


def _save_real_vs_sim_figure(fig: plt.Figure, path: str) -> None:
    """Save figure including outside legends (second row is below the axes box)."""
    fig.savefig(path, bbox_inches="tight", pad_inches=0.12)


def plot_real_vs_sim_io_comparison(
    real_by_comp: Dict[int, Dict[str, Any]],
    sim_by_comp: Dict[int, Dict[str, Any]],
    output_dir: str,
    *,
    constructions: Sequence[Tuple[int, str]] = REAL_VS_SIM_COMPOSITIONS,
) -> None:
    """Compare I/O for real (normalized) vs simulated data at StepChain and TaskChain.

    Writes:

    - ``io_patterns_real_vs_sim_local.png`` — per-event + stacked totals with local read
    - ``io_patterns_real_vs_sim_nonlocal.png`` — per-event + stacked totals without local read
    """
    comps = [c for c, _ in constructions if c in real_by_comp and c in sim_by_comp]
    if not comps:
        raise ValueError("No overlapping composition numbers between real and simulated data")

    labels = [lbl for c, lbl in constructions if c in comps]
    real_io = {c: io_volumes_from_workflow(real_by_comp[c]) for c in comps}
    sim_io = {c: io_volumes_from_workflow(sim_by_comp[c]) for c in comps}

    n_plot = len(comps)
    x = np.arange(n_plot)
    cluster_sep = REAL_SIM_CLUSTER_SEP
    stack_w = STACKED_BAR_WIDTH
    bar_w_local = _per_event_bar_width(4)
    bar_w_nonlocal = _per_event_bar_width(3)

    fig_w = STACKED_COMPARISON_FIG_W_IN
    fig_h = STACKED_COMPARISON_FIG_H_IN + REAL_VS_SIM_FIG_H_EXTRA_IN

    os.makedirs(output_dir, exist_ok=True)

    all_totals_local = [
        v.total_read_local_mb + v.total_read_remote_mb + v.total_write_local_mb + v.total_write_remote_mb
        for v in list(real_io.values()) + list(sim_io.values())
    ]
    vol_scale_l, vol_unit_l = _stacked_total_volume_scale_and_unit_from_max_mb(max(all_totals_local))

    fig1, (ax1, ax3) = plt.subplots(2, 1, figsize=(fig_w, fig_h), layout="constrained", sharex=True)
    for i, comp in enumerate(comps):
        _clustered_per_event_bars(
            ax1, i, real_io[comp], sim_io[comp], include_local_read=True,
            cluster_sep=cluster_sep, bar_w=bar_w_local,
        )
        _clustered_stacked_totals(
            ax3, i, real_io[comp], sim_io[comp], include_local_read=True,
            cluster_sep=cluster_sep, stack_w=stack_w, vol_scale=vol_scale_l,
        )

    ax1.set_ylabel("Data Volume per Event (MB)")
    ax1.set_title("Data Volume per Event — Real vs Simulated")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", labelbottom=False)

    ax3.set_xlabel("Workflow Construction")
    ax3.set_ylabel(f"Total Data Volume ({vol_unit_l})")
    ax3.set_title("Total Data Volume — Real vs Simulated")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=0, ha="center")
    _style_stacked_total_data_volume_axis(ax3)
    _real_vs_sim_io_legend_below(fig1, include_local_read=True)

    path_local = os.path.join(output_dir, "io_patterns_real_vs_sim_local.png")
    _save_real_vs_sim_figure(fig1, path_local)
    plt.close(fig1)
    print(f"  => Real vs sim I/O (local) saved to {path_local}")

    all_totals_nl = [
        v.total_read_remote_mb + v.total_write_local_mb + v.total_write_remote_mb
        for v in list(real_io.values()) + list(sim_io.values())
    ]
    vol_scale_nl, vol_unit_nl = _stacked_total_volume_scale_and_unit_from_max_mb(max(all_totals_nl))

    fig2, (ax2, ax4) = plt.subplots(2, 1, figsize=(fig_w, fig_h), layout="constrained", sharex=True)
    for i, comp in enumerate(comps):
        _clustered_per_event_bars(
            ax2, i, real_io[comp], sim_io[comp], include_local_read=False,
            cluster_sep=cluster_sep, bar_w=bar_w_nonlocal,
        )
        _clustered_stacked_totals(
            ax4, i, real_io[comp], sim_io[comp], include_local_read=False,
            cluster_sep=cluster_sep, stack_w=stack_w, vol_scale=vol_scale_nl,
        )

    ax2.set_ylabel("Data Volume per Event (MB)")
    ax2.set_title("Data Volume per Event — Real vs Simulated")
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="x", labelbottom=False)

    ax4.set_xlabel("Workflow Construction")
    ax4.set_ylabel(f"Total Data Volume ({vol_unit_nl})")
    ax4.set_title("Total Data Volume — Real vs Simulated")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=0, ha="center")
    _style_stacked_total_data_volume_axis(ax4)
    _real_vs_sim_io_legend_below(fig2, include_local_read=False)

    path_nl = os.path.join(output_dir, "io_patterns_real_vs_sim_nonlocal.png")
    _save_real_vs_sim_figure(fig2, path_nl)
    plt.close(fig2)
    print(f"  => Real vs sim I/O (non-local) saved to {path_nl}")


def _target_composition_numbers(
    constructions: Sequence[Tuple[int, str]],
) -> List[int]:
    return [comp for comp, _ in constructions]


def load_real_workflows_by_composition(
    real_dir: str,
    compositions: List[int],
) -> Dict[int, Dict[str, Any]]:
    """Load and transform real summary JSONs keyed by composition number."""
    directory = Path(real_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Real data directory not found: {real_dir}")

    by_comp: Dict[int, Dict[str, Any]] = {}
    for path in sorted(directory.glob("summary_const*.json")):
        comp = extract_construction_number(path.name)
        if comp not in compositions:
            continue
        with path.open() as handle:
            raw = json.load(handle)
        by_comp[comp] = transform_real_data_to_simulation_format(raw, path.name)

    missing = sorted(set(compositions) - set(by_comp))
    if missing:
        raise FileNotFoundError(
            f"Missing real summary files for compositions {missing} in {real_dir}"
        )
    return by_comp


def load_sim_workflows_by_composition(
    sim_dir: str,
    compositions: List[int],
) -> Dict[int, Dict[str, Any]]:
    """Load simulation JSONs and flatten metrics keyed by composition number."""
    by_comp: Dict[int, Dict[str, Any]] = {}
    for file_path in find_simulation_files(sim_dir):
        file_name = Path(file_path).name
        comp = extract_construction_number(file_name)
        if comp not in compositions:
            continue
        with open(file_path) as handle:
            simulation_data = json.load(handle)
        workflow: Dict[str, Any] = {"_file_name": file_name}
        metrics = simulation_data.get("metrics", {})
        workflow.update(metrics)
        comp = int(metrics.get("composition_number") or comp)
        sim_result = simulation_data.get("simulation_result", {})
        workflow["_overhead_enabled"] = sim_result.get("overhead_enabled", True)
        by_comp[comp] = workflow

    missing = sorted(set(compositions) - set(by_comp))
    if missing:
        raise FileNotFoundError(
            f"Missing simulation files for compositions {missing} in {sim_dir}"
        )
    return by_comp


def _scenario_label_from_sim_path(sim_dir: str) -> str:
    """Derive a short scenario label from a typical sim results path."""
    parts = Path(sim_dir).parts
    for anchor in ("others", "sim"):
        if anchor in parts:
            idx = parts.index(anchor)
            tail = parts[idx + 1 :]
            if tail:
                return "/".join(tail)
    return Path(sim_dir).name


def main() -> None:
    """CLI entry point for real vs simulated I/O comparison plots."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare I/O patterns for real (normalized) CMS data vs a simulated scenario "
            "at StepChain (const 1) and TaskChain (const 16)."
        ),
    )
    parser.add_argument(
        "--real-dir",
        default="results/real_norm",
        help="Directory with summary_const001.json and summary_const016.json (default: real_norm)",
    )
    parser.add_argument(
        "--sim-dir",
        default="results/sim/others/seq_real/12h/fr0/100MBps",
        help="Directory with seq_real_const_*.json simulation results",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: results/vis/comparison/<scenario>)",
    )
    args = parser.parse_args()

    compositions = _target_composition_numbers(REAL_VS_SIM_COMPOSITIONS)
    real_by_comp = load_real_workflows_by_composition(args.real_dir, compositions)
    sim_by_comp = load_sim_workflows_by_composition(args.sim_dir, compositions)

    scenario = _scenario_label_from_sim_path(args.sim_dir)
    output_dir = args.output_dir or f"results/vis/comparison/real_vs_{scenario.replace('/', '_')}"

    print(f"Real data: {args.real_dir}")
    print(f"Simulation: {args.sim_dir}")
    print(f"Output: {output_dir}")
    plot_real_vs_sim_io_comparison(
        real_by_comp,
        sim_by_comp,
        output_dir,
        constructions=REAL_VS_SIM_COMPOSITIONS,
    )


if __name__ == "__main__":
    main()
