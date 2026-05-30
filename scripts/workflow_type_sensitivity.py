#!/usr/bin/env python3
"""
Workflow Type Sensitivity Analysis Script

This script analyzes how different workflow types respond to hybrid workflow
constructions compared to extreme cases. It aggregates data across workflow
types to identify which types benefit most from hybrid compositions.

Analysis: Workflow Type Sensitivity (Comparison #2)
- Fixed: target_job_length + failure_rate
- Variable: workflow_type (seq_real, seq_homo, seq_hetero)
- Compare: most grouped, most ungrouped, and best hybrid across workflow types
  (see :mod:`composition_extremes`)
- Primary Metric: event_throughput
- Second Metric: network_transfer_mb_per_event
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from composition_extremes import composition_extremes_from_single_map


def _extremes(wf: Dict[int, Dict[str, Any]]) -> tuple:
    if not wf:
        return (1, 16)
    return composition_extremes_from_single_map(wf)


def _style_event_throughput_yaxis(ax: plt.Axes, throughput_values: List[float]) -> None:
    """Plain decimal y ticks for small event throughput (no sci notation or offset multiplier).

    Keeps units as events/s. Uses one extra decimal when the data range is below ~0.02 evt/s.
    """
    hi = max(throughput_values) if throughput_values else 0.0
    fmt = "%.4f" if 0.0 < hi < 0.02 else "%.3f"
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))


def _legend_label_with_extremes(name: str, comp_nums: List[int]) -> str:
    """Build legend as 'Const N (most grouped)'; many N: 'Const 1, 3 (most grouped)'."""
    if not comp_nums:
        return name
    unique = sorted(set(comp_nums))
    role = name.lower()
    if len(unique) == 1:
        return f"Const {unique[0]} ({role})"
    return f"Const {', '.join(str(n) for n in unique)} ({role})"


def _default_workflow_sensitivity_output_dir(
    target_job_length: str,
    failure_rate: str,
    workflow_types: List[str],
) -> str:
    """Under results/analysis/workflow_type_sensitivity, use family subdirs when clear.

    If ``seq_real`` is in *workflow_types*, use ``.../sequential/{target}/{fr}``.
    Else if ``fork_real`` is in *workflow_types*, use ``.../fork/{target}/{fr}``.
    Otherwise use ``.../{target}/{fr}`` (no extra segment).
    """
    root = "results/analysis/workflow_type_sensitivity"
    if "seq_real" in workflow_types:
        return f"{root}/sequential/{target_job_length}/{failure_rate}"
    if "fork_real" in workflow_types:
        return f"{root}/fork/{target_job_length}/{failure_rate}"
    return f"{root}/{target_job_length}/{failure_rate}"


def load_simulation_data(file_path: str) -> Optional[Dict[str, Any]]:
    """Load and extract key metrics from a simulation JSON file.

    Args:
        file_path: Path to simulation result JSON file

    Returns:
        Dictionary with extracted metrics, or None if loading fails
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        metrics = data.get('metrics', {})
        sim_result = data.get('simulation_result', {})

        return {
            'composition_number': metrics.get('composition_number', 0),
            'event_throughput': metrics.get('event_throughput', 0.0),
            'wall_time_per_event': metrics.get('wall_time_per_event', 0.0),
            'cpu_time_per_event': metrics.get('cpu_time_per_event', 0.0),
            'total_write_remote_mb_per_event': metrics.get('total_write_remote_mb_per_event', 0.0),
            'total_read_remote_mb_per_event': metrics.get('total_read_remote_mb_per_event', 0.0),
            'network_transfer_mb_per_event': metrics.get('network_transfer_mb_per_event', 0.0),
            'cpu_utilization': metrics.get('cpu_utilization', 0.0),
            'memory_occupancy': metrics.get('memory_occupancy', 0.0),
            'total_groups': metrics.get('total_groups', 0),
            'failure_rate': sim_result.get('job_failure_rate', 0.0),
            'overhead_enabled': sim_result.get('overhead_enabled', True),
            'file_path': file_path
        }
    except Exception as e:
        print(f"  Warning: Failed to load {file_path}: {e}")
        return None


def collect_data_from_workflow_types(base_path: str,
                                     workflow_types: List[str],
                                     target_job_length: str,
                                     failure_rate: str,
                                     data_rate: str = "100MBps") -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Collect simulation data from multiple workflow types.

    Reads simulation result JSON files (*.json) in each workflow type directory.

    Args:
        base_path: Base path to results directory (e.g., 'results/sim/others')
        workflow_types: List of workflow types (e.g., ['seq_real', 'seq_homo', 'seq_hetero'])
        target_job_length: Target job length (e.g., '12h')
        failure_rate: Failure rate directory (e.g., 'fr0')
        data_rate: Data transfer rate directory (e.g., '100MBps')

    Returns:
        Dictionary mapping workflow_type to composition_number to metrics. Key
        order follows *workflow_types*; types with no data are omitted.
    """
    # Dictionary: workflow_type -> composition_number -> metrics
    data_by_workflow: Dict[str, Dict[int, Dict[str, Any]]] = {}

    print(f"Collecting data from workflow types: {', '.join(workflow_types)}")
    print(f"Target job length: {target_job_length}, Failure rate: {failure_rate}, Data rate: {data_rate}")

    for workflow_type in workflow_types:
        workflow_dir = Path(base_path) / workflow_type / target_job_length / failure_rate / data_rate

        if not workflow_dir.exists():
            print(f"  Warning: Directory {workflow_dir} not found, skipping {workflow_type}")
            continue

        # Find simulation result JSON files
        json_files = list(workflow_dir.glob("*.json"))
        print(f"  Processing {workflow_type}: {len(json_files)} files found")

        data_by_composition: Dict[int, Dict[str, Any]] = {}
        
        for json_file in sorted(json_files):
            metrics = load_simulation_data(str(json_file))
            if metrics:
                comp_num = metrics['composition_number']
                data_by_composition[comp_num] = metrics

        if data_by_composition:
            data_by_workflow[workflow_type] = data_by_composition

    return data_by_workflow


def identify_best_hybrid(
    data_by_composition: Dict[int, Dict[str, Any]],
    grouped_comp: int,
    independent_comp: int,
    verbose: bool = False,
) -> Optional[int]:
    """Best hybrid (strictly between grouped and ungrouped extremes).

    Uses event_throughput as the primary metric, with network_transfer_mb_per_event
    as a tiebreaker (lower network transfer is preferred).

    Args:
        data_by_composition: Dictionary mapping composition_number to metrics
        grouped_composition: From :func:`composition_extremes_from_single_map`
        independent_comp: From :func:`composition_extremes_from_single_map`
        verbose: If True, print information about ties to stdout

    Returns:
        Composition number of best hybrid, or None if not found
    """
    # Collect all hybrid constructions with their metrics
    hybrid_candidates = []

    for comp_num in range(grouped_comp + 1, independent_comp):
        if comp_num not in data_by_composition:
            continue

        metrics = data_by_composition[comp_num]
        hybrid_candidates.append({
            'comp_num': comp_num,
            'throughput': metrics['event_throughput'],
            'network_transfer': metrics['network_transfer_mb_per_event']
        })

    if not hybrid_candidates:
        return None

    # Find maximum throughput
    max_throughput = max(c['throughput'] for c in hybrid_candidates)

    # Find all candidates with maximum throughput
    tied_candidates = [c for c in hybrid_candidates if abs(c['throughput'] - max_throughput) < 1e-10]

    # If there's a tie, use network transfer as tiebreaker (lower is better)
    if len(tied_candidates) > 1:
        if verbose:
            tied_names = [f"Const {c['comp_num']}" for c in tied_candidates]
            print(f"    Tie detected: {', '.join(tied_names)} "
                  f"(throughput: {max_throughput:.6f} evt/s)")
            print(f"      Using network transfer as tiebreaker (lower is better)")

        # Sort by network transfer (ascending - lower is better), then by comp_num for consistency
        tied_candidates.sort(key=lambda x: (x['network_transfer'], x['comp_num']))
        best_comp = tied_candidates[0]['comp_num']

        if verbose:
            print(f"      Selected: Const {best_comp} "
                  f"(network transfer: {tied_candidates[0]['network_transfer']:.6f} MB/evt)")
    else:
        best_comp = tied_candidates[0]['comp_num']

    return best_comp


# Shared layout for side-by-side IEEE-friendly figures.
_COMBINED_FIG_SIZE = (8.0, 4.0)
_COMBINED_TITLE_FS = 10
_COMBINED_LABEL_FS = 9
_COMBINED_TICK_FS = 8
_COMBINED_LEGEND_FS = 7
_COMBINED_BAR_WIDTH = 0.25
_IMPROVEMENT_BAR_WIDTH = 0.35
_GROUPED_BAR_COLOR = "#d62728"
_UNGROUPED_BAR_COLOR = "#2ca02c"
_HYBRID_BAR_COLOR = "#1f77b4"


@dataclass
class _WorkflowTypePlotData:
    """Per-workflow-type metrics for combined comparison and improvement plots."""

    workflow_types: List[str]
    x: np.ndarray
    grouped_throughput: List[float]
    indep_throughput: List[float]
    best_hybrid_throughput: List[float]
    grouped_network: List[float]
    indep_network: List[float]
    best_hybrid_network: List[float]
    best_hybrid_labels: List[str]
    grouped_comp_nums: List[int]
    indep_comp_nums: List[int]
    throughput_improvement_grouped: List[float]
    throughput_improvement_indep: List[float]
    network_reduction_grouped: List[float]
    network_reduction_indep: List[float]


def _pct_throughput_improvement(best: float, baseline: float) -> float:
    if baseline <= 0:
        return 0.0
    return ((best - baseline) / baseline) * 100


def _pct_network_reduction(best: float, baseline: float) -> float:
    if baseline <= 0:
        return 0.0
    return ((baseline - best) / baseline) * 100


def _collect_workflow_type_plot_data(
    data_by_workflow: Dict[str, Dict[int, Dict[str, Any]]],
) -> _WorkflowTypePlotData:
    """Aggregate extremes, best hybrid, and improvement percentages in one pass."""
    workflow_types = list(data_by_workflow.keys())
    x = np.arange(len(workflow_types))

    grouped_throughput: List[float] = []
    indep_throughput: List[float] = []
    best_hybrid_throughput: List[float] = []
    grouped_network: List[float] = []
    indep_network: List[float] = []
    best_hybrid_network: List[float] = []
    best_hybrid_labels: List[str] = []
    grouped_comp_nums: List[int] = []
    indep_comp_nums: List[int] = []
    throughput_improvement_grouped: List[float] = []
    throughput_improvement_indep: List[float] = []
    network_reduction_grouped: List[float] = []
    network_reduction_indep: List[float] = []

    for workflow_type in workflow_types:
        workflow_data = data_by_workflow[workflow_type]
        g_comp, indep_comp = _extremes(workflow_data)
        grouped_comp_nums.append(g_comp)
        indep_comp_nums.append(indep_comp)

        g_tp = workflow_data[g_comp]["event_throughput"] if g_comp in workflow_data else 0.0
        i_tp = (
            workflow_data[indep_comp]["event_throughput"]
            if indep_comp in workflow_data
            else 0.0
        )
        g_net = (
            workflow_data[g_comp]["network_transfer_mb_per_event"]
            if g_comp in workflow_data
            else 0.0
        )
        i_net = (
            workflow_data[indep_comp]["network_transfer_mb_per_event"]
            if indep_comp in workflow_data
            else 0.0
        )
        grouped_throughput.append(g_tp)
        indep_throughput.append(i_tp)
        grouped_network.append(g_net)
        indep_network.append(i_net)

        best_hybrid = identify_best_hybrid(
            workflow_data, g_comp, indep_comp, verbose=False
        )
        if best_hybrid and best_hybrid in workflow_data:
            bh = workflow_data[best_hybrid]
            b_tp = bh["event_throughput"]
            b_net = bh["network_transfer_mb_per_event"]
            best_hybrid_throughput.append(b_tp)
            best_hybrid_network.append(b_net)
            best_hybrid_labels.append(f"Const {best_hybrid}")
            throughput_improvement_grouped.append(
                _pct_throughput_improvement(b_tp, g_tp)
            )
            throughput_improvement_indep.append(
                _pct_throughput_improvement(b_tp, i_tp)
            )
            network_reduction_grouped.append(_pct_network_reduction(b_net, g_net))
            network_reduction_indep.append(_pct_network_reduction(b_net, i_net))
        else:
            best_hybrid_throughput.append(0.0)
            best_hybrid_network.append(0.0)
            best_hybrid_labels.append("N/A")
            throughput_improvement_grouped.append(0.0)
            throughput_improvement_indep.append(0.0)
            network_reduction_grouped.append(0.0)
            network_reduction_indep.append(0.0)

    return _WorkflowTypePlotData(
        workflow_types=workflow_types,
        x=x,
        grouped_throughput=grouped_throughput,
        indep_throughput=indep_throughput,
        best_hybrid_throughput=best_hybrid_throughput,
        grouped_network=grouped_network,
        indep_network=indep_network,
        best_hybrid_network=best_hybrid_network,
        best_hybrid_labels=best_hybrid_labels,
        grouped_comp_nums=grouped_comp_nums,
        indep_comp_nums=indep_comp_nums,
        throughput_improvement_grouped=throughput_improvement_grouped,
        throughput_improvement_indep=throughput_improvement_indep,
        network_reduction_grouped=network_reduction_grouped,
        network_reduction_indep=network_reduction_indep,
    )


def _hybrid_bar_labels(heights: List[float], labels: List[str], *, fmt: str) -> List[str]:
    """Two lines per hybrid bar: construction id first, metric below (centered on bar)."""
    out: List[str] = []
    for i, h in enumerate(heights):
        if h <= 0:
            out.append("")
            continue
        val = f"{h:{fmt}}"
        hl = labels[i]
        if hl == "N/A":
            out.append(val)
        else:
            short = hl.replace("Const ", "C")
            out.append(f"{short}\n{val}")
    return out


def _apply_combined_subplot_layout(fig: plt.Figure) -> None:
    fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.26, wspace=0.22)


def _add_combined_figure_legend(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    ncol: int,
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.14),
        ncol=ncol,
        fontsize=_COMBINED_LEGEND_FS,
        frameon=True,
        columnspacing=0.9,
        handletextpad=0.35,
    )


def _save_combined_figure(fig: plt.Figure, output_dir: str, filename: str) -> None:
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    print(f"  => Saved: {output_path}")


def _label_percentage_bars(ax: plt.Axes, bar_groups: List[Any], *, fontsize: float) -> None:
    """Add percentage labels on grouped improvement bars (handles near-zero heights)."""
    for bars in bar_groups:
        for bar in bars:
            height = bar.get_height()
            if abs(height) < 0.01:
                label_y = 0.3 if height >= 0 else -0.3
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    label_y,
                    "0.0%",
                    ha="center",
                    va="bottom" if height >= 0 else "top",
                    fontsize=fontsize,
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}%",
                    ha="center",
                    va="bottom" if height > 0 else "top",
                    fontsize=fontsize,
                )


def plot_throughput_and_network_combined(
    data_by_workflow: Dict[str, Dict[int, Dict[str, Any]]],
    output_dir: str,
) -> None:
    """Throughput (left) and network transfer (right) in one row.

    Saves: ``throughput_network_comparison.png``
    """
    print("==> Creating combined throughput + network efficiency figure (side-by-side)")

    data = _collect_workflow_type_plot_data(data_by_workflow)
    width = _COMBINED_BAR_WIDTH
    hybrid_lbl_fs = 6.5

    fig, (ax_t, ax_n) = plt.subplots(1, 2, figsize=_COMBINED_FIG_SIZE)
    _apply_combined_subplot_layout(fig)

    ax_t.bar(
        data.x - width,
        data.grouped_throughput,
        width,
        label=_legend_label_with_extremes("Most grouped", data.grouped_comp_nums),
        color=_GROUPED_BAR_COLOR,
        alpha=0.8,
    )
    ax_t.bar(
        data.x,
        data.indep_throughput,
        width,
        label=_legend_label_with_extremes("Most ungrouped", data.indep_comp_nums),
        color=_UNGROUPED_BAR_COLOR,
        alpha=0.8,
    )
    bars3 = ax_t.bar(
        data.x + width,
        data.best_hybrid_throughput,
        width,
        label="Best Hybrid",
        color=_HYBRID_BAR_COLOR,
        alpha=0.8,
    )
    ax_t.bar_label(
        bars3,
        labels=_hybrid_bar_labels(
            data.best_hybrid_throughput, data.best_hybrid_labels, fmt=".3f"
        ),
        padding=4,
        fontsize=hybrid_lbl_fs,
    )
    tp_vals = (
        data.grouped_throughput + data.indep_throughput + data.best_hybrid_throughput
    )
    tp_hi = max(tp_vals)
    if tp_hi > 0:
        ax_t.set_ylim(0, tp_hi * 1.20)
        _style_event_throughput_yaxis(ax_t, tp_vals)

    ax_t.set_xlabel("Workflow Type", fontsize=_COMBINED_LABEL_FS)
    ax_t.set_ylabel("Event Throughput (events/s)", fontsize=_COMBINED_LABEL_FS)
    ax_t.set_title("(a) Throughput", fontsize=_COMBINED_TITLE_FS)
    ax_t.set_xticks(data.x)
    ax_t.set_xticklabels(
        data.workflow_types, fontsize=_COMBINED_TICK_FS, rotation=0, ha="center"
    )
    ax_t.grid(True, alpha=0.3, axis="y")

    ax_n.bar(
        data.x - width,
        data.grouped_network,
        width,
        label=_legend_label_with_extremes("Most grouped", data.grouped_comp_nums),
        color=_GROUPED_BAR_COLOR,
        alpha=0.8,
    )
    ax_n.bar(
        data.x,
        data.indep_network,
        width,
        label=_legend_label_with_extremes("Most ungrouped", data.indep_comp_nums),
        color=_UNGROUPED_BAR_COLOR,
        alpha=0.8,
    )
    nb3 = ax_n.bar(
        data.x + width,
        data.best_hybrid_network,
        width,
        label="Best Hybrid",
        color=_HYBRID_BAR_COLOR,
        alpha=0.8,
    )
    ax_n.bar_label(
        nb3,
        labels=_hybrid_bar_labels(
            data.best_hybrid_network, data.best_hybrid_labels, fmt=".3f"
        ),
        padding=4,
        fontsize=hybrid_lbl_fs,
    )
    net_hi = max(
        data.grouped_network + data.indep_network + data.best_hybrid_network
    )
    if net_hi > 0:
        ax_n.set_ylim(0, net_hi * 1.20)

    ax_n.set_xlabel("Workflow Type", fontsize=_COMBINED_LABEL_FS)
    ax_n.set_ylabel("Network Transfer per Event (MB)", fontsize=_COMBINED_LABEL_FS)
    ax_n.set_title("(b) Network efficiency", fontsize=_COMBINED_TITLE_FS)
    ax_n.set_xticks(data.x)
    ax_n.set_xticklabels(
        data.workflow_types, fontsize=_COMBINED_TICK_FS, rotation=0, ha="center"
    )
    ax_n.grid(True, alpha=0.3, axis="y")

    _add_combined_figure_legend(fig, ax_t, ncol=3)
    _save_combined_figure(fig, output_dir, "throughput_network_comparison.png")


def plot_throughput_and_network_improvement_combined(
    data_by_workflow: Dict[str, Dict[int, Dict[str, Any]]],
    output_dir: str,
) -> None:
    """Throughput (left) and network reduction (right) improvement percentages in one row.

    Saves: ``throughput_network_improvement.png``
    """
    print("==> Creating combined throughput + network improvement figure (side-by-side)")

    data = _collect_workflow_type_plot_data(data_by_workflow)
    width = _IMPROVEMENT_BAR_WIDTH
    bar_lbl_fs = _COMBINED_LEGEND_FS

    fig, (ax_tp, ax_net) = plt.subplots(1, 2, figsize=_COMBINED_FIG_SIZE)
    _apply_combined_subplot_layout(fig)

    tp_bars1 = ax_tp.bar(
        data.x - width / 2,
        data.throughput_improvement_grouped,
        width,
        label="Over most grouped",
        color=_GROUPED_BAR_COLOR,
        alpha=0.8,
    )
    tp_bars2 = ax_tp.bar(
        data.x + width / 2,
        data.throughput_improvement_indep,
        width,
        label="Over most ungrouped",
        color=_UNGROUPED_BAR_COLOR,
        alpha=0.8,
    )
    _label_percentage_bars(ax_tp, [tp_bars1, tp_bars2], fontsize=bar_lbl_fs)

    ax_tp.set_xlabel("Workflow Type", fontsize=_COMBINED_LABEL_FS)
    ax_tp.set_ylabel("Throughput Improvement (%)", fontsize=_COMBINED_LABEL_FS)
    ax_tp.set_title("(a) Throughput improvement", fontsize=_COMBINED_TITLE_FS)
    ax_tp.set_xticks(data.x)
    ax_tp.set_xticklabels(
        data.workflow_types, fontsize=_COMBINED_TICK_FS, rotation=0, ha="center"
    )
    ax_tp.grid(True, alpha=0.3, axis="y")
    ax_tp.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    net_bars1 = ax_net.bar(
        data.x - width / 2,
        data.network_reduction_grouped,
        width,
        label="Over most grouped",
        color=_GROUPED_BAR_COLOR,
        alpha=0.8,
    )
    net_bars2 = ax_net.bar(
        data.x + width / 2,
        data.network_reduction_indep,
        width,
        label="Over most ungrouped",
        color=_UNGROUPED_BAR_COLOR,
        alpha=0.8,
    )
    _label_percentage_bars(ax_net, [net_bars1, net_bars2], fontsize=bar_lbl_fs)

    ax_net.set_xlabel("Workflow Type", fontsize=_COMBINED_LABEL_FS)
    ax_net.set_ylabel("Network Transfer Reduction (%)", fontsize=_COMBINED_LABEL_FS)
    ax_net.set_title("(b) Network improvement", fontsize=_COMBINED_TITLE_FS)
    ax_net.set_xticks(data.x)
    ax_net.set_xticklabels(
        data.workflow_types, fontsize=_COMBINED_TICK_FS, rotation=0, ha="center"
    )
    ax_net.grid(True, alpha=0.3, axis="y")
    ax_net.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    _add_combined_figure_legend(fig, ax_tp, ncol=2)
    _save_combined_figure(fig, output_dir, "throughput_network_improvement.png")


def generate_summary_table(data_by_workflow: Dict[str, Dict[int, Dict[str, Any]]],
                          output_dir: str) -> pd.DataFrame:
    """Generate summary table with metrics across workflow types.

    Args:
        data_by_workflow: Dictionary mapping workflow_type to composition metrics
        output_dir: Output directory for table

    Returns:
        DataFrame with summary metrics
    """
    print(f"\n==> Generating summary table")

    table_data = []

    for workflow_type in data_by_workflow:
        workflow_data = data_by_workflow[workflow_type]
        
        g_comp, indep_comp = _extremes(workflow_data)
        const_g_data = workflow_data.get(g_comp)
        const_i_data = workflow_data.get(indep_comp)
        best_hybrid = identify_best_hybrid(
            workflow_data, g_comp, indep_comp, verbose=False
        )
        best_hybrid_data = workflow_data.get(best_hybrid) if best_hybrid else None

        for comp_num, metrics in [
            (g_comp, const_g_data),
            (indep_comp, const_i_data),
            (best_hybrid, best_hybrid_data),
        ]:
            if metrics is None:
                continue

            row = {
                'Workflow_Type': workflow_type,
                'Construction': comp_num,
                'Event_Throughput': metrics['event_throughput'],
                'Wall_Time_Per_Event': metrics['wall_time_per_event'],
                'CPU_Time_Per_Event': metrics['cpu_time_per_event'],
                'Network_Transfer_MB_Per_Event': metrics['network_transfer_mb_per_event'],
                'CPU_Utilization': metrics['cpu_utilization'],
                'Memory_Occupancy': metrics['memory_occupancy'],
                'Total_Groups': metrics['total_groups']
            }
            table_data.append(row)

    df = pd.DataFrame(table_data)

    # Save as CSV
    csv_path = os.path.join(output_dir, "workflow_type_sensitivity_summary.csv")
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  => Saved: {csv_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Analyze workflow construction performance across workflow types'
    )
    parser.add_argument('base_path', type=str,
                       help='Base path to results directory (e.g., results/sim/others)')
    parser.add_argument('target_job_length', type=str,
                       help='Target job length (e.g., 12h)')
    parser.add_argument('failure_rate', type=str,
                       help='Failure rate directory (e.g., fr0)')
    parser.add_argument('--data-rate', type=str, default='100MBps',
                       help='Data transfer rate directory (default: 100MBps)')
    parser.add_argument('--workflow-types', type=str, nargs='+',
                       default=['seq_real', 'seq_homo', 'seq_hetero'],
                       help='Workflow types to analyze (default: seq_real seq_homo seq_hetero)')
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory (default: under results/analysis/workflow_type_sensitivity: "
            ".../sequential/.../ if seq_real is in --workflow-types, else "
            ".../fork/.../ if fork_real, else .../<target_job_length>/<failure_rate> without "
            "a family subdir)"
        ),
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = _default_workflow_sensitivity_output_dir(
            args.target_job_length,
            args.failure_rate,
            list(args.workflow_types),
        )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Workflow Type Sensitivity Analysis")
    print("="*70)
    print(f"Target Job Length: {args.target_job_length}")
    print(f"Failure Rate: {args.failure_rate}")
    print(f"Data Rate: {args.data_rate}")
    print(f"Workflow Types: {', '.join(args.workflow_types)}")
    print(f"Output Directory: {args.output_dir}")
    print("="*70)

    data_by_workflow = collect_data_from_workflow_types(
        args.base_path,
        args.workflow_types,
        args.target_job_length,
        args.failure_rate,
        data_rate=args.data_rate
    )

    if not data_by_workflow:
        print("Error: No data collected. Please check directory paths and file availability.")
        return

    print(f"\nCollected data for {len(data_by_workflow)} workflow types")

    print(f"\nIdentifying best hybrid for each workflow type:")
    for workflow_type in data_by_workflow:
        wd = data_by_workflow[workflow_type]
        g0, i0 = _extremes(wd)
        best_hybrid = identify_best_hybrid(wd, g0, i0, verbose=True)
        if best_hybrid:
            print(
                f"  {workflow_type}: extremes (grouped, ungrouped) = ({g0}, {i0}); "
                f"best hybrid Const {best_hybrid}"
            )

    plot_throughput_and_network_combined(data_by_workflow, args.output_dir)
    plot_throughput_and_network_improvement_combined(data_by_workflow, args.output_dir)
    generate_summary_table(data_by_workflow, args.output_dir)

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
