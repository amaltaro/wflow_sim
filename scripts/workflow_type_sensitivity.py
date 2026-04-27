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
from typing import Any, Dict, List, Optional
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from composition_extremes import composition_extremes_from_single_map


def _extremes(wf: Dict[int, Dict[str, Any]]) -> tuple:
    if not wf:
        return (1, 16)
    return composition_extremes_from_single_map(wf)


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


def plot_throughput_comparison(data_by_workflow: Dict[str, Dict[int, Dict[str, Any]]],
                               output_dir: str) -> None:
    """Plot throughput comparison across workflow types.

    Args:
        data_by_workflow: Dictionary mapping workflow_type to composition metrics
        output_dir: Output directory for plots
    """
    print(f"\n==> Creating throughput comparison plot")

    fig, ax = plt.subplots(figsize=(12, 7))

    workflow_types = list(data_by_workflow.keys())
    x = np.arange(len(workflow_types))
    width = 0.25

    grouped_t: List[float] = []
    indep_t: List[float] = []
    best_hybrid_throughput: List[float] = []
    best_hybrid_labels: List[str] = []
    grouped_comp_nums: List[int] = []
    indep_comp_nums: List[int] = []

    for workflow_type in workflow_types:
        workflow_data = data_by_workflow[workflow_type]
        g_comp, indep_comp = _extremes(workflow_data)
        grouped_comp_nums.append(g_comp)
        indep_comp_nums.append(indep_comp)
        if g_comp in workflow_data:
            grouped_t.append(workflow_data[g_comp]['event_throughput'])
        else:
            grouped_t.append(0.0)
        if indep_comp in workflow_data:
            indep_t.append(workflow_data[indep_comp]['event_throughput'])
        else:
            indep_t.append(0.0)
        best_hybrid = identify_best_hybrid(
            workflow_data, g_comp, indep_comp, verbose=False
        )
        if best_hybrid and best_hybrid in workflow_data:
            best_hybrid_throughput.append(workflow_data[best_hybrid]['event_throughput'])
            best_hybrid_labels.append(f"Const {best_hybrid}")
        else:
            best_hybrid_throughput.append(0.0)
            best_hybrid_labels.append("N/A")

    bars1 = ax.bar(
        x - width,
        grouped_t,
        width,
        label=_legend_label_with_extremes("Most grouped", grouped_comp_nums),
        color="#d62728",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x,
        indep_t,
        width,
        label=_legend_label_with_extremes("Most ungrouped", indep_comp_nums),
        color="#2ca02c",
        alpha=0.8,
    )
    bars3 = ax.bar(x + width, best_hybrid_throughput, width, label='Best Hybrid',
                  color='#1f77b4', alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    # Best hybrid construction labels only (extremes appear in the legend)
    for i, _ in enumerate(workflow_types):
        if best_hybrid_throughput[i] > 0:
            ax.text(
                i + width,
                best_hybrid_throughput[i] + max(best_hybrid_throughput) * 0.02,
                best_hybrid_labels[i],
                ha="center",
                va="bottom",
                fontsize=8,
                style="italic",
            )

    ax.set_xlabel("Workflow Type", fontsize=12)
    ax.set_ylabel("Event Throughput (events/second)", fontsize=12)
    ax.set_title("Throughput Comparison Across Workflow Types", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(workflow_types, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "throughput_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_path}")


def plot_improvement_percentage(data_by_workflow: Dict[str, Dict[int, Dict[str, Any]]],
                                output_dir: str) -> None:
    """Plot event throughput improvement percentage of best hybrid over extremes.

    Calculates percentage improvement as: ((best_hybrid_throughput - extreme_throughput) / extreme_throughput) × 100

    Args:
        data_by_workflow: Dictionary mapping workflow_type to composition metrics
        output_dir: Output directory for plots
    """
    print(f"==> Creating improvement percentage plot")

    fig, ax = plt.subplots(figsize=(12, 7))

    workflow_types = list(data_by_workflow.keys())
    x = np.arange(len(workflow_types))
    width = 0.35

    improvement_over_const1 = []
    improvement_over_const16 = []

    for workflow_type in workflow_types:
        workflow_data = data_by_workflow[workflow_type]
        g_comp, indep_comp = _extremes(workflow_data)
        # Get best hybrid
        best_hybrid = identify_best_hybrid(
            workflow_data, g_comp, indep_comp, verbose=False
        )
        if not best_hybrid or best_hybrid not in workflow_data:
            improvement_over_const1.append(0.0)
            improvement_over_const16.append(0.0)
            continue

        best_throughput = workflow_data[best_hybrid]['event_throughput']

        if g_comp in workflow_data and workflow_data[g_comp]['event_throughput'] > 0:
            grouped_tp = workflow_data[g_comp]['event_throughput']
            improvement = ((best_throughput - grouped_tp) / grouped_tp) * 100
            improvement_over_const1.append(improvement)
        else:
            improvement_over_const1.append(0.0)

        if indep_comp in workflow_data and workflow_data[indep_comp]['event_throughput'] > 0:
            indep_tp = workflow_data[indep_comp]['event_throughput']
            improvement = ((best_throughput - indep_tp) / indep_tp) * 100
            improvement_over_const16.append(improvement)
        else:
            improvement_over_const16.append(0.0)

    bars1 = ax.bar(
        x - width / 2, improvement_over_const1, width,
        label="Over most grouped", color="#d62728", alpha=0.8
    )
    bars2 = ax.bar(
        x + width / 2, improvement_over_const16, width,
        label="Over most ungrouped", color="#2ca02c", alpha=0.8
    )

    # Add value labels on bars
    # Always show labels, even for zero or very small values
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            # For zero or near-zero values, show label slightly offset from baseline
            if abs(height) < 0.01:
                # Position label at a small offset so it's visible
                label_y = 0.3 if height >= 0 else -0.3
                ax.text(bar.get_x() + bar.get_width()/2., label_y,
                       '0.0%', ha='center', va='bottom' if height >= 0 else 'top', 
                       fontsize=9)
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', 
                       va='bottom' if height > 0 else 'top', fontsize=9)

    ax.set_xlabel("Workflow Type", fontsize=12)
    ax.set_ylabel("Throughput Improvement (%)", fontsize=12)
    ax.set_title("Best Hybrid Throughput Improvement Over Extremes", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(workflow_types, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "throughput_improvement.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_path}")


def plot_network_efficiency_comparison(data_by_workflow: Dict[str, Dict[int, Dict[str, Any]]],
                                      output_dir: str) -> None:
    """Plot network efficiency comparison across workflow types.

    Args:
        data_by_workflow: Dictionary mapping workflow_type to composition metrics
        output_dir: Output directory for plots
    """
    print(f"==> Creating network efficiency comparison plot")

    fig, ax = plt.subplots(figsize=(12, 7))

    workflow_types = list(data_by_workflow.keys())
    x = np.arange(len(workflow_types))
    width = 0.25

    const1_network: List[float] = []
    const16_network: List[float] = []
    best_hybrid_network: List[float] = []
    best_hybrid_labels: List[str] = []
    grouped_comp_nums: List[int] = []
    indep_comp_nums: List[int] = []

    for workflow_type in workflow_types:
        workflow_data = data_by_workflow[workflow_type]
        g_comp, indep_comp = _extremes(workflow_data)
        grouped_comp_nums.append(g_comp)
        indep_comp_nums.append(indep_comp)
        if g_comp in workflow_data:
            const1_network.append(workflow_data[g_comp]['network_transfer_mb_per_event'])
        else:
            const1_network.append(0.0)
        if indep_comp in workflow_data:
            const16_network.append(
                workflow_data[indep_comp]['network_transfer_mb_per_event']
            )
        else:
            const16_network.append(0.0)
        best_hybrid = identify_best_hybrid(
            workflow_data, g_comp, indep_comp, verbose=False
        )
        if best_hybrid and best_hybrid in workflow_data:
            best_hybrid_network.append(
                workflow_data[best_hybrid]['network_transfer_mb_per_event']
            )
            best_hybrid_labels.append(f"Const {best_hybrid}")
        else:
            best_hybrid_network.append(0.0)
            best_hybrid_labels.append("N/A")

    bars1 = ax.bar(
        x - width,
        const1_network,
        width,
        label=_legend_label_with_extremes("Most grouped", grouped_comp_nums),
        color="#d62728",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x,
        const16_network,
        width,
        label=_legend_label_with_extremes("Most ungrouped", indep_comp_nums),
        color="#2ca02c",
        alpha=0.8,
    )
    bars3 = ax.bar(x + width, best_hybrid_network, width, label='Best Hybrid',
                  color='#1f77b4', alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # Best hybrid construction labels only (extremes appear in the legend)
    for i, _ in enumerate(workflow_types):
        if best_hybrid_network[i] > 0 and best_hybrid_labels[i] != "N/A":
            ax.text(
                i + width,
                best_hybrid_network[i] + max(best_hybrid_network) * 0.02,
                best_hybrid_labels[i],
                ha="center",
                va="bottom",
                fontsize=8,
                style="italic",
            )

    ax.set_xlabel("Workflow Type", fontsize=12)
    ax.set_ylabel("Network Transfer per Event (MB)", fontsize=12)
    ax.set_title("Network Efficiency Comparison Across Workflow Types", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(workflow_types, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "network_efficiency_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_path}")


def plot_network_improvement_percentage(data_by_workflow: Dict[str, Dict[int, Dict[str, Any]]],
                                       output_dir: str) -> None:
    """Plot network transfer reduction percentage of best hybrid over extremes.

    Since lower network transfer is better, this shows reduction percentage:
    ((extreme_network - best_hybrid_network) / extreme_network) × 100
    Positive values indicate the hybrid uses less network (better).

    Args:
        data_by_workflow: Dictionary mapping workflow_type to composition metrics
        output_dir: Output directory for plots
    """
    print(f"==> Creating network improvement percentage plot")

    fig, ax = plt.subplots(figsize=(12, 7))

    workflow_types = list(data_by_workflow.keys())
    x = np.arange(len(workflow_types))
    width = 0.35

    reduction_over_const1 = []
    reduction_over_const16 = []

    for workflow_type in workflow_types:
        workflow_data = data_by_workflow[workflow_type]
        g_comp, indep_comp = _extremes(workflow_data)
        best_hybrid = identify_best_hybrid(
            workflow_data, g_comp, indep_comp, verbose=False
        )
        if not best_hybrid or best_hybrid not in workflow_data:
            reduction_over_const1.append(0.0)
            reduction_over_const16.append(0.0)
            continue

        best_network = workflow_data[best_hybrid]['network_transfer_mb_per_event']

        if g_comp in workflow_data and workflow_data[g_comp]['network_transfer_mb_per_event'] > 0:
            gn = workflow_data[g_comp]['network_transfer_mb_per_event']
            reduction = ((gn - best_network) / gn) * 100
            reduction_over_const1.append(reduction)
        else:
            reduction_over_const1.append(0.0)

        if indep_comp in workflow_data and workflow_data[indep_comp]['network_transfer_mb_per_event'] > 0:
            inn = workflow_data[indep_comp]['network_transfer_mb_per_event']
            reduction = ((inn - best_network) / inn) * 100
            reduction_over_const16.append(reduction)
        else:
            reduction_over_const16.append(0.0)

    bars1 = ax.bar(
        x - width / 2, reduction_over_const1, width,
        label="Over most grouped", color="#d62728", alpha=0.8
    )
    bars2 = ax.bar(
        x + width / 2, reduction_over_const16, width,
        label="Over most ungrouped", color="#2ca02c", alpha=0.8
    )

    # Add value labels on bars
    # Always show labels, even for zero or very small values
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            # For zero or near-zero values, show label slightly offset from baseline
            if abs(height) < 0.01:
                # Position label at a small offset so it's visible
                label_y = 0.3 if height >= 0 else -0.3
                ax.text(bar.get_x() + bar.get_width()/2., label_y,
                       '0.0%', ha='center', va='bottom' if height >= 0 else 'top', 
                       fontsize=9)
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', 
                       va='bottom' if height > 0 else 'top', fontsize=9)

    ax.set_xlabel("Workflow Type", fontsize=12)
    ax.set_ylabel("Network Transfer Reduction (%)", fontsize=12)
    ax.set_title("Best Hybrid Network Transfer Reduction Over Extremes", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(workflow_types, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "network_improvement_percentage.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_path}")


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

    plot_throughput_comparison(data_by_workflow, args.output_dir)
    plot_improvement_percentage(data_by_workflow, args.output_dir)
    plot_network_efficiency_comparison(data_by_workflow, args.output_dir)
    plot_network_improvement_percentage(data_by_workflow, args.output_dir)
    generate_summary_table(data_by_workflow, args.output_dir)

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
