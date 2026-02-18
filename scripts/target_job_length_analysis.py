#!/usr/bin/env python3
"""
Target Job Length Optimization Analysis Script

This script analyzes how different workflow constructions (1-16) perform across
various target job lengths. It aggregates data from multiple directories to create
cross-dimensional comparisons.

Analysis: Target Job Length Optimization (Comparison #3)
- Fixed: workflow_type + failure_rate
- Variable: target_job_length (15m, 30m, 1h, 2h, 4h, 8h, 12h, 24h)
- Compare: all 16 constructions across target job lengths
- Primary Metric: event_throughput
- Second Metric: network_transfer_mb_per_event
"""

import argparse
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
        jobs = sim_result.get('jobs', [])

        # Authoritative job counts and actual failure rate from JSON
        total_jobs = metrics.get('total_jobs', 0)
        total_job_retries = sim_result.get('total_job_retries', 0)
        total_logical_jobs = total_jobs - total_job_retries
        total_failed_jobs = total_job_retries  # each retry corresponds to one failed first attempt
        failure_rate_actual = sim_result.get('actual_job_failure_rate')

        # Failure cost metrics from job sample (wasted resources; sample may be limited)
        first_attempt_jobs = [j for j in jobs if j.get('retry_count', 0) == 0]
        failed_jobs = [j for j in first_attempt_jobs if j.get('status') == 'failed']
        total_wasted_cpu = sum(j.get('total_cpu_used_time', 0.0) for j in failed_jobs)
        total_wasted_wall = sum(j.get('wallclock_time', 0.0) for j in failed_jobs)
        total_wasted_network = sum(j.get('total_network_transfer_mb', 0.0) for j in failed_jobs)
        sample_failed = len(failed_jobs)
        avg_cpu_per_failure = total_wasted_cpu / sample_failed if sample_failed > 0 else 0.0
        avg_wall_per_failure = total_wasted_wall / sample_failed if sample_failed > 0 else 0.0
        avg_network_per_failure = total_wasted_network / sample_failed if sample_failed > 0 else 0.0
        max_cpu_failure = max((j.get('total_cpu_used_time', 0.0) for j in failed_jobs), default=0.0)
        max_wall_failure = max((j.get('wallclock_time', 0.0) for j in failed_jobs), default=0.0)
        max_network_failure = max(
            (j.get('total_network_transfer_mb', 0.0) for j in failed_jobs), default=0.0
        )

        return {
            'composition_number': metrics.get('composition_number', 0),
            'total_events': metrics.get('total_events', 0),
            'total_turnaround_time': metrics.get('total_turnaround_time', 0.0),
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
            # Failure metrics (counts from metrics/simulation_result; waste from job sample)
            'total_jobs': total_jobs,
            'total_job_retries': total_job_retries,
            'total_logical_jobs': total_logical_jobs,
            'total_failed_jobs': total_failed_jobs,
            'failure_rate_actual': failure_rate_actual,
            'total_wasted_cpu_time': total_wasted_cpu,
            'total_wasted_wall_time': total_wasted_wall,
            'total_wasted_network_mb': total_wasted_network,
            'avg_cpu_per_failure': avg_cpu_per_failure,
            'avg_wall_per_failure': avg_wall_per_failure,
            'avg_network_per_failure': avg_network_per_failure,
            'max_cpu_per_failure': max_cpu_failure,
            'max_wall_per_failure': max_wall_failure,
            'max_network_per_failure': max_network_failure,
            'file_path': file_path
        }
    except Exception as e:
        print(f"  Warning: Failed to load {file_path}: {e}")
        return None


def collect_data_from_directories(base_path: str,
                                  workflow_type: str,
                                  failure_rate: str,
                                  data_rate: str = "100MBps") -> Dict[int, List[Dict[str, Any]]]:
    """Collect simulation data from multiple target job length directories.

    Reads simulation result JSON files (*.json) in each target-length directory.

    Args:
        base_path: Base path to results directory (e.g., 'results/sim/others')
        workflow_type: Workflow type (e.g., 'case1_real')
        failure_rate: Failure rate directory (e.g., 'fr0')
        data_rate: Data transfer rate directory (e.g., '100MBps')

    Returns:
        Dictionary mapping composition_number to list of metrics across target job lengths
    """
    base_dir = Path(base_path) / workflow_type

    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    target_job_lengths = ['15m', '30m', '1h', '2h', '4h', '8h', '12h', '24h']

    # Dictionary: composition_number -> list of metrics (one per target job length)
    data_by_composition: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    print(f"Collecting data from: {base_dir}")
    print(f"Failure rate: {failure_rate}, Data rate: {data_rate}")

    for target_length in target_job_lengths:
        target_path = base_dir / target_length / failure_rate / data_rate
        if not target_path.exists():
            print(f"  Warning: Directory {target_path} not found, skipping")
            continue

        # Find simulation result JSON files
        json_files = list(target_path.glob("*.json"))
        print(f"  Processing {target_length}: {len(json_files)} files found")

        for json_file in sorted(json_files):
            metrics = load_simulation_data(str(json_file))
            if metrics:
                comp_num = metrics['composition_number']
                # Add target_job_length to metrics for later use
                metrics['target_job_length'] = target_length
                data_by_composition[comp_num].append(metrics)

    # Sort by composition number
    return dict(sorted(data_by_composition.items()))


def target_length_to_hours(target_length: str) -> float:
    """Convert target job length string to hours.

    Args:
        target_length: Target job length string (e.g., '15m', '30m', '1h', '12h')

    Returns:
        Hours as float
    """
    if target_length.endswith('m'):
        return int(target_length.replace('m', '')) / 60.0
    return float(target_length.replace('h', ''))


def get_target_length_xconfig(
    target_lengths: List[str],
    include_zero: bool = False,
) -> Tuple[List[str], Dict[str, int]]:
    """Get x-axis labels and mapping for categorical (equal) spacing.

    Uses integer positions so 15m, 30m, 1h are clearly separated (unlike linear hours).

    Args:
        target_lengths: Sorted list of target length strings (e.g. ['15m', '30m', '1h', ...])
        include_zero: If True, prepend '0' as first category (for origin plots)

    Returns:
        (xtick_labels, tl_to_x): labels for xticks and dict mapping target_length -> x position
    """
    if include_zero:
        labels = ["0"] + list(target_lengths)
    else:
        labels = list(target_lengths)
    tl_to_x = {tl: i for i, tl in enumerate(labels)}
    return labels, tl_to_x


def get_best_hybrid_colors(best_hybrids: Dict[str, Optional[int]]) -> Dict[int, str]:
    """Get unique colors for each best hybrid construction.

    Uses a color palette that excludes red (#d62728) and green (#2ca02c) used for Const 1 and Const 16.

    Args:
        best_hybrids: Dictionary mapping target_length to best hybrid composition number

    Returns:
        Dictionary mapping composition_number to color hex code
    """
    # Color palette for best hybrids (distinct from Const 1 red and Const 16 green)
    # Exclude red (#d62728) and green (#2ca02c) which are reserved for Const 1 and Const 16
    hybrid_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf',  # Cyan
        '#ff9896',  # Light red
        '#98df8a',  # Light green
        '#c5b0d5',  # Light purple
        '#ffbb78',  # Light orange
        '#c49c94',  # Light brown
        '#aec7e8',  # Light blue
        '#ffdbac',  # Peach
    ]

    # Get unique best hybrid composition numbers
    unique_hybrids = sorted(set([h for h in best_hybrids.values() if h is not None]))

    # Assign colors (cycling through palette if needed)
    color_map = {}
    for i, comp_num in enumerate(unique_hybrids):
        color_map[comp_num] = hybrid_colors[i % len(hybrid_colors)]

    return color_map


def plot_throughput_vs_target_length(data_by_composition: Dict[int, List[Dict[str, Any]]],
                                    best_hybrids: Dict[str, Optional[int]],
                                    output_dir: str) -> None:
    """Plot event throughput vs. target job length for all constructions.

    Args:
        data_by_composition: Dictionary mapping composition_number to metrics list
        best_hybrids: Dictionary mapping target_length to best hybrid composition number
        output_dir: Output directory for plots
    """
    print(f"\n==> Creating throughput vs. target job length plot")

    fig, ax = plt.subplots(figsize=(14, 8))

    # Extract target lengths and throughput for each construction
    target_lengths = []
    for comp_num in sorted(data_by_composition.keys()):
        comp_data = data_by_composition[comp_num]
        comp_data_sorted = sorted(comp_data, key=lambda x: target_length_to_hours(x['target_job_length']))
        target_length_values = [d['target_job_length'] for d in comp_data_sorted]
        if not target_lengths:
            target_lengths = target_length_values

    hybrid_color_map = get_best_hybrid_colors(best_hybrids)
    xtick_labels, tl_to_x = get_target_length_xconfig(target_lengths, include_zero=False)

    # Plot lines for all constructions
    for comp_num in sorted(data_by_composition.keys()):
        comp_data = data_by_composition[comp_num]
        comp_data_sorted = sorted(comp_data, key=lambda x: target_length_to_hours(x['target_job_length']))
        target_length_values = [d['target_job_length'] for d in comp_data_sorted]
        throughput_values = [d['event_throughput'] for d in comp_data_sorted]
        x_positions = [tl_to_x[tl] for tl in target_length_values]
        y_values = throughput_values

        # Plot line for this construction
        label = f"Const {comp_num}"
        if comp_num == 1:
            ax.plot(x_positions, y_values, 'o-', label=label, linewidth=2.5,
                   color='#d62728', markersize=8, zorder=10)
        elif comp_num == 16:
            ax.plot(x_positions, y_values, 's-', label=label, linewidth=2.5,
                   color='#2ca02c', markersize=8, zorder=10)
        else:
            is_best_hybrid = any(best_hybrids[tl] == comp_num for tl in target_lengths
                                if best_hybrids[tl] is not None)
            if is_best_hybrid:
                hybrid_color = hybrid_color_map.get(comp_num, '#1f77b4')
                ax.plot(x_positions, y_values, '^-', label=f"Best Hybrid (C{comp_num})",
                       linewidth=2, color=hybrid_color, markersize=7, zorder=10, alpha=0.9)
            else:
                ax.plot(x_positions, y_values, '-', label=label, linewidth=1.5,
                       alpha=0.7, markersize=5)

    ax.set_xlabel("Target Job Length", fontsize=12)
    ax.set_ylabel("Event Throughput (events/second)", fontsize=12)
    ax.set_title("Event Throughput vs. Target Job Length", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "throughput_vs_target_length.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_path}")


def plot_throughput_improvement(data_by_composition: Dict[int, List[Dict[str, Any]]],
                                output_dir: str) -> None:
    """Plot throughput improvement (relative to shortest target length) vs. target job length.

    Args:
        data_by_composition: Dictionary mapping composition_number to metrics list
        output_dir: Output directory for plots
    """
    print(f"==> Creating throughput improvement plot")

    fig, ax = plt.subplots(figsize=(14, 8))

    all_target_lengths = set()
    for comp_data in data_by_composition.values():
        for d in comp_data:
            all_target_lengths.add(d['target_job_length'])
    target_lengths = sorted(all_target_lengths, key=target_length_to_hours)
    xtick_labels, tl_to_x = get_target_length_xconfig(target_lengths, include_zero=False)

    for comp_num in sorted(data_by_composition.keys()):
        comp_data = data_by_composition[comp_num]
        comp_data_sorted = sorted(comp_data, key=lambda x: target_length_to_hours(x['target_job_length']))
        baseline = comp_data_sorted[0] if comp_data_sorted else None
        if not baseline:
            continue

        baseline_throughput = baseline['event_throughput']
        if baseline_throughput == 0:
            continue

        target_length_values = [d['target_job_length'] for d in comp_data_sorted]
        improvement_values = [
            ((d['event_throughput'] - baseline_throughput) / baseline_throughput) * 100
            for d in comp_data_sorted
        ]
        x_positions = [tl_to_x[tl] for tl in target_length_values]
        y_values = improvement_values

        label = f"Const {comp_num}"
        if comp_num == 1:
            ax.plot(x_positions, y_values, 'o-', label=label, linewidth=2.5,
                   color='#d62728', markersize=8, zorder=10)
        elif comp_num == 16:
            ax.plot(x_positions, y_values, 's-', label=label, linewidth=2.5,
                   color='#2ca02c', markersize=8, zorder=10)
        else:
            ax.plot(x_positions, y_values, '-', label=label, linewidth=1.5,
                   alpha=0.7, markersize=5)

    baseline_label = target_lengths[0] if target_lengths else "shortest"
    ax.set_xlabel("Target Job Length", fontsize=12)
    ax.set_ylabel("Throughput Improvement (%)", fontsize=12)
    ax.set_title(f"Throughput Improvement vs. Target Job Length\n(Relative to {baseline_label})", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "throughput_improvement.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_path}")


def identify_best_hybrid(data_by_composition: Dict[int, List[Dict[str, Any]]],
                        target_length: str,
                        verbose: bool = False) -> Optional[int]:
    """Identify the best hybrid construction (2-15) for a given target job length.

    Uses event_throughput as the primary metric, with network_transfer_mb_per_event
    as a tiebreaker (lower network transfer is preferred).

    Args:
        data_by_composition: Dictionary mapping composition_number to metrics list
        target_length: Target job length (e.g., '12h')
        verbose: If True, print information about ties to stdout

    Returns:
        Composition number of best hybrid, or None if not found
    """
    # Collect all hybrid constructions with their metrics
    hybrid_candidates = []

    for comp_num in range(2, 16):  # Only hybrid constructions (2-15)
        if comp_num not in data_by_composition:
            continue

        comp_data = data_by_composition[comp_num]
        # Find data for this target length
        target_data = next((d for d in comp_data if d['target_job_length'] == target_length), None)
        if target_data:
            hybrid_candidates.append({
                'comp_num': comp_num,
                'throughput': target_data['event_throughput'],
                'network_transfer': target_data['network_transfer_mb_per_event']
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
            print(f"    Tie detected at {target_length}: {', '.join(tied_names)} "
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


def identify_best_hybrid_per_target_length(
    data_by_composition: Dict[int, List[Dict[str, Any]]],
    verbose: bool = False,
) -> Dict[str, Optional[int]]:
    """Identify the best hybrid construction (2-15) for each target job length.

    Target job lengths are derived from the data. Each target length can have a
    different best hybrid. Uses identify_best_hybrid per target length (throughput
    as primary metric, network transfer as tiebreaker).

    Args:
        data_by_composition: Dictionary mapping composition_number to metrics list
        verbose: If True, print best hybrid per target length and tie details

    Returns:
        Dictionary mapping target_length -> best composition number, or None if not found
    """
    all_target_lengths = set()
    for comp_data in data_by_composition.values():
        for d in comp_data:
            all_target_lengths.add(d['target_job_length'])
    target_lengths = sorted(all_target_lengths, key=target_length_to_hours)

    best_hybrids: Dict[str, Optional[int]] = {}
    for target_length in target_lengths:
        best_hybrid = identify_best_hybrid(data_by_composition, target_length, verbose=verbose)
        best_hybrids[target_length] = best_hybrid
        if verbose and best_hybrid is not None:
            print(f"  Best hybrid for {target_length}: Const {best_hybrid}")
    return best_hybrids


def plot_network_activity_vs_target_length(data_by_composition: Dict[int, List[Dict[str, Any]]],
                                           best_hybrids: Dict[str, Optional[int]],
                                           output_dir: str) -> None:
    """Plot network transfer vs. target job length for all constructions.

    This visualization shows how network activity (remote I/O) changes with
    target job lengths, which is important for understanding workflow efficiency
    under different time constraints.

    Args:
        data_by_composition: Dictionary mapping composition_number to metrics list
        best_hybrids: Dictionary mapping target_length to best hybrid composition number
        output_dir: Output directory for plots
    """
    print(f"==> Creating network activity vs. target job length plot")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    all_target_lengths = set()
    for comp_data in data_by_composition.values():
        for d in comp_data:
            all_target_lengths.add(d['target_job_length'])
    target_lengths = sorted(all_target_lengths, key=target_length_to_hours)
    xtick_labels, tl_to_x = get_target_length_xconfig(target_lengths, include_zero=False)

    # Plot 1: Network Transfer per Event vs. Target Job Length
    for comp_num in sorted(data_by_composition.keys()):
        comp_data = data_by_composition[comp_num]
        comp_data_sorted = sorted(comp_data, key=lambda x: target_length_to_hours(x['target_job_length']))
        target_length_values = [d['target_job_length'] for d in comp_data_sorted]
        network_values = [d['network_transfer_mb_per_event'] for d in comp_data_sorted]
        x_positions = [tl_to_x[tl] for tl in target_length_values]

        label = f"Const {comp_num}"
        if comp_num == 1:
            ax1.plot(x_positions, network_values, 'o-', label=label, linewidth=2.5,
                    color='#d62728', markersize=8, zorder=10)
        elif comp_num == 16:
            ax1.plot(x_positions, network_values, 's-', label=label, linewidth=2.5,
                    color='#2ca02c', markersize=8, zorder=10)
        else:
            ax1.plot(x_positions, network_values, '-', label=label, linewidth=1.5,
                    alpha=0.7, markersize=5)

    ax1.set_xlabel("Target Job Length", fontsize=12)
    ax1.set_ylabel("Network Transfer per Event (MB)", fontsize=12)
    ax1.set_title("Network Transfer vs. Target Job Length", fontsize=13)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(len(xtick_labels)))
    ax1.set_xticklabels(xtick_labels)

    # Plot 2: Remote Read vs. Remote Write breakdown (focus on extremes and best hybrid)
    hybrid_color_map = get_best_hybrid_colors(best_hybrids)

    # Plot only Const 1, Const 16, and best hybrid for each target length
    constructions_to_plot = {1, 16}
    for target_length in target_lengths:
        if best_hybrids[target_length] is not None:
            constructions_to_plot.add(best_hybrids[target_length])

    for comp_num in sorted(constructions_to_plot):
        if comp_num not in data_by_composition:
            continue

        comp_data = data_by_composition[comp_num]
        comp_data_sorted = sorted(comp_data, key=lambda x: target_length_to_hours(x['target_job_length']))
        target_length_values = [d['target_job_length'] for d in comp_data_sorted]
        read_remote = [d['total_read_remote_mb_per_event'] for d in comp_data_sorted]
        write_remote = [d['total_write_remote_mb_per_event'] for d in comp_data_sorted]
        x_positions = [tl_to_x[tl] for tl in target_length_values]

        label = f"Const {comp_num}"
        if comp_num == 1:
            ax2.plot(x_positions, read_remote, 'o--', label=f"{label} (Read)", linewidth=2.5,
                    color='#d62728', markersize=7, zorder=10, alpha=0.7, markerfacecolor='#d62728',
                    markeredgecolor='#d62728', markeredgewidth=1.5)
            ax2.plot(x_positions, write_remote, 's-', label=f"{label} (Write)", linewidth=2.5,
                    color='#d62728', markersize=7, zorder=10, alpha=0.9, markerfacecolor='#d62728',
                    markeredgecolor='#d62728', markeredgewidth=1.5)
        elif comp_num == 16:
            ax2.plot(x_positions, read_remote, 'o--', label=f"{label} (Read)", linewidth=2.5,
                    color='#2ca02c', markersize=7, zorder=10, alpha=0.9, markerfacecolor='#2ca02c',
                    markeredgecolor='#2ca02c', markeredgewidth=1.5)
            ax2.plot(x_positions, write_remote, 's-', label=f"{label} (Write)", linewidth=2.5,
                    color='#2ca02c', markersize=7, zorder=10, alpha=0.9, markerfacecolor='#2ca02c',
                    markeredgecolor='#2ca02c', markeredgewidth=1.5)
        else:
            hybrid_color = hybrid_color_map.get(comp_num, '#1f77b4')
            ax2.plot(x_positions, read_remote, 'o--', label=f"{label} (Read)", linewidth=2,
                    color=hybrid_color, markersize=6, zorder=9, alpha=0.8, markerfacecolor=hybrid_color,
                    markeredgecolor=hybrid_color, markeredgewidth=1.5)
            ax2.plot(x_positions, write_remote, 's-', label=f"{label} (Write)", linewidth=2,
                    color=hybrid_color, markersize=6, zorder=9, alpha=0.8, markerfacecolor=hybrid_color,
                    markeredgecolor=hybrid_color, markeredgewidth=1.5)

    ax2.set_xlabel("Target Job Length", fontsize=12)
    ax2.set_ylabel("Data Volume per Event (MB)", fontsize=12)
    ax2.set_title("Remote I/O Breakdown vs. Target Job Length", fontsize=13)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(len(xtick_labels)))
    ax2.set_xticklabels(xtick_labels)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "network_activity_vs_target_length.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_path}")


def plot_best_hybrid_comparison(data_by_composition: Dict[int, List[Dict[str, Any]]],
                                best_hybrids: Dict[str, Optional[int]],
                                output_dir: str) -> None:
    """Plot comparison of Const 1, Const 16, and best hybrid for each target job length.

    Args:
        data_by_composition: Dictionary mapping composition_number to metrics list
        best_hybrids: Dictionary mapping target_length to best hybrid composition number
        output_dir: Output directory for plots
    """
    print(f"==> Creating best hybrid comparison plot")

    # Get all target lengths from the data
    all_target_lengths = set()
    for comp_data in data_by_composition.values():
        for d in comp_data:
            all_target_lengths.add(d['target_job_length'])
    target_lengths = sorted(all_target_lengths, key=target_length_to_hours)
    target_hours = [target_length_to_hours(tl) for tl in target_lengths]

    # Extract throughput values
    const1_throughput = []
    const16_throughput = []
    best_hybrid_throughput = []
    best_hybrid_labels = []

    for target_length in target_lengths:
        # Const 1
        if 1 in data_by_composition:
            const1_data = next((d for d in data_by_composition[1]
                              if d['target_job_length'] == target_length), None)
            const1_throughput.append(const1_data['event_throughput'] if const1_data else 0.0)
        else:
            const1_throughput.append(0.0)

        # Const 16
        if 16 in data_by_composition:
            const16_data = next((d for d in data_by_composition[16]
                              if d['target_job_length'] == target_length), None)
            const16_throughput.append(const16_data['event_throughput'] if const16_data else 0.0)
        else:
            const16_throughput.append(0.0)

        # Best hybrid
        best_comp = best_hybrids[target_length]
        if best_comp and best_comp in data_by_composition:
            best_data = next((d for d in data_by_composition[best_comp]
                            if d['target_job_length'] == target_length), None)
            best_hybrid_throughput.append(best_data['event_throughput'] if best_data else 0.0)
            best_hybrid_labels.append(f"C{best_comp}")
        else:
            best_hybrid_throughput.append(0.0)
            best_hybrid_labels.append("N/A")

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(target_lengths))
    width = 0.25

    # Determine best hybrid label for legend (show unique best hybrids)
    unique_best_hybrids = sorted(set([best_hybrids[tl] for tl in target_lengths if best_hybrids[tl] is not None]))
    if len(unique_best_hybrids) == 1:
        best_hybrid_legend = f"Best Hybrid (Const {unique_best_hybrids[0]})"
    elif len(unique_best_hybrids) <= 3:
        best_hybrid_legend = f"Best Hybrid (Const {', '.join(map(str, unique_best_hybrids))})"
    else:
        best_hybrid_legend = f"Best Hybrid (Const {unique_best_hybrids[0]}-{unique_best_hybrids[-1]})"

    bars1 = ax.bar(x - width, const1_throughput, width, label='Const 1 (All Chained)',
                  color='#d62728', alpha=0.8)
    bars2 = ax.bar(x, const16_throughput, width, label='Const 16 (All Independent)',
                  color='#2ca02c', alpha=0.8)
    bars3 = ax.bar(x + width, best_hybrid_throughput, width, label=best_hybrid_legend,
                  color='#1f77b4', alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=8)

    # Add best hybrid construction labels above best hybrid bars
    for i, (target_length, label) in enumerate(zip(target_lengths, best_hybrid_labels)):
        if best_hybrid_throughput[i] > 0:
            ax.text(i + width, best_hybrid_throughput[i] + max(best_hybrid_throughput) * 0.02,
                   label, ha='center', va='bottom', fontsize=8, style='italic')

    ax.set_xlabel("Target Job Length", fontsize=12)
    ax.set_ylabel("Event Throughput (events/second)", fontsize=12)
    ax.set_title("Best Hybrid vs. Extremes Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(target_lengths)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "best_hybrid_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_path}")


def plot_total_jobs_comparison(data_by_composition: Dict[int, List[Dict[str, Any]]],
                               best_hybrids: Dict[str, Optional[int]],
                               output_dir: str) -> None:
    """Plot total job count for Const 1, Const 16, and best hybrid per target job length.

    Same grouped bar layout as best_hybrid_comparison, but for total_jobs.

    Args:
        data_by_composition: Dictionary mapping composition_number to metrics list
        best_hybrids: Dictionary mapping target_length to best hybrid composition number
        output_dir: Output directory for plots
    """
    print(f"==> Creating total jobs comparison plot")

    all_target_lengths = set()
    for comp_data in data_by_composition.values():
        for d in comp_data:
            all_target_lengths.add(d['target_job_length'])
    target_lengths = sorted(all_target_lengths, key=target_length_to_hours)

    const1_jobs = []
    const16_jobs = []
    best_hybrid_jobs = []
    best_hybrid_labels = []

    for target_length in target_lengths:
        if 1 in data_by_composition:
            const1_data = next((d for d in data_by_composition[1]
                              if d['target_job_length'] == target_length), None)
            const1_jobs.append(const1_data.get('total_jobs', 0) if const1_data else 0)
        else:
            const1_jobs.append(0)

        if 16 in data_by_composition:
            const16_data = next((d for d in data_by_composition[16]
                              if d['target_job_length'] == target_length), None)
            const16_jobs.append(const16_data.get('total_jobs', 0) if const16_data else 0)
        else:
            const16_jobs.append(0)

        best_comp = best_hybrids[target_length]
        if best_comp and best_comp in data_by_composition:
            best_data = next((d for d in data_by_composition[best_comp]
                            if d['target_job_length'] == target_length), None)
            best_hybrid_jobs.append(best_data.get('total_jobs', 0) if best_data else 0)
            best_hybrid_labels.append(f"C{best_comp}")
        else:
            best_hybrid_jobs.append(0)
            best_hybrid_labels.append("N/A")

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(target_lengths))
    width = 0.25

    unique_best_hybrids = sorted(set([best_hybrids[tl] for tl in target_lengths
                                     if best_hybrids[tl] is not None]))
    if len(unique_best_hybrids) == 1:
        best_hybrid_legend = f"Best Hybrid (Const {unique_best_hybrids[0]})"
    elif len(unique_best_hybrids) <= 3:
        best_hybrid_legend = f"Best Hybrid (Const {', '.join(map(str, unique_best_hybrids))})"
    else:
        best_hybrid_legend = f"Best Hybrid (Const {unique_best_hybrids[0]}-{unique_best_hybrids[-1]})"

    bars1 = ax.bar(x - width, const1_jobs, width, label='Const 1 (All Chained)',
                  color='#d62728', alpha=0.8)
    bars2 = ax.bar(x, const16_jobs, width, label='Const 16 (All Independent)',
                  color='#2ca02c', alpha=0.8)
    bars3 = ax.bar(x + width, best_hybrid_jobs, width, label=best_hybrid_legend,
                  color='#1f77b4', alpha=0.8)

    max_jobs = max(const1_jobs + const16_jobs + best_hybrid_jobs) or 1
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=8)

    for i, (target_length, label) in enumerate(zip(target_lengths, best_hybrid_labels)):
        if best_hybrid_jobs[i] > 0:
            ax.text(i + width, best_hybrid_jobs[i] + max_jobs * 0.02,
                   label, ha='center', va='bottom', fontsize=8, style='italic')

    # Total events is common across all constructions and target lengths
    total_events = 0
    for comp_data in data_by_composition.values():
        for d in comp_data:
            total_events = d.get('total_events', 0)
            if total_events > 0:
                break
        if total_events > 0:
            break
    events_str = f"{total_events:.2e}".replace('e+0', 'e').replace('e+', 'e') if total_events > 0 else "N/A"

    ax.set_xlabel("Target Job Length", fontsize=12)
    ax.set_ylabel("Total Jobs", fontsize=12)
    ax.set_title(f"Total Jobs: Best Hybrid vs. Extremes ({events_str} events)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(target_lengths)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "total_jobs_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_path}")


def plot_turnaround_time_comparison(data_by_composition: Dict[int, List[Dict[str, Any]]],
                                    best_hybrids: Dict[str, Optional[int]],
                                    output_dir: str) -> None:
    """Plot turnaround time for Const 1, Const 16, and best hybrid per target job length.

    Same grouped bar layout as best_hybrid_comparison. Time is converted from seconds
    to hours (or days if max >= 24h) for readability.

    Args:
        data_by_composition: Dictionary mapping composition_number to metrics list
        best_hybrids: Dictionary mapping target_length to best hybrid composition number
        output_dir: Output directory for plots
    """
    print(f"==> Creating turnaround time comparison plot")

    all_target_lengths = set()
    for comp_data in data_by_composition.values():
        for d in comp_data:
            all_target_lengths.add(d['target_job_length'])
    target_lengths = sorted(all_target_lengths, key=target_length_to_hours)

    const1_turnaround = []
    const16_turnaround = []
    best_hybrid_turnaround = []
    best_hybrid_labels = []

    for target_length in target_lengths:
        if 1 in data_by_composition:
            const1_data = next((d for d in data_by_composition[1]
                              if d['target_job_length'] == target_length), None)
            const1_turnaround.append(const1_data.get('total_turnaround_time', 0.0)
                                    if const1_data else 0.0)
        else:
            const1_turnaround.append(0.0)

        if 16 in data_by_composition:
            const16_data = next((d for d in data_by_composition[16]
                              if d['target_job_length'] == target_length), None)
            const16_turnaround.append(const16_data.get('total_turnaround_time', 0.0)
                                     if const16_data else 0.0)
        else:
            const16_turnaround.append(0.0)

        best_comp = best_hybrids[target_length]
        if best_comp and best_comp in data_by_composition:
            best_data = next((d for d in data_by_composition[best_comp]
                            if d['target_job_length'] == target_length), None)
            best_hybrid_turnaround.append(best_data.get('total_turnaround_time', 0.0)
                                         if best_data else 0.0)
            best_hybrid_labels.append(f"C{best_comp}")
        else:
            best_hybrid_turnaround.append(0.0)
            best_hybrid_labels.append("N/A")

    # Convert seconds to hours (or days if max >= 24h)
    all_turnaround = const1_turnaround + const16_turnaround + best_hybrid_turnaround
    max_sec = max(all_turnaround) if all_turnaround else 0
    use_days = max_sec >= 86400  # 24 hours
    div = 86400.0 if use_days else 3600.0
    unit = "days" if use_days else "hours"

    const1_vals = [v / div for v in const1_turnaround]
    const16_vals = [v / div for v in const16_turnaround]
    best_hybrid_vals = [v / div for v in best_hybrid_turnaround]

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(target_lengths))
    width = 0.25

    unique_best_hybrids = sorted(set([best_hybrids[tl] for tl in target_lengths
                                    if best_hybrids[tl] is not None]))
    if len(unique_best_hybrids) == 1:
        best_hybrid_legend = f"Best Hybrid (Const {unique_best_hybrids[0]})"
    elif len(unique_best_hybrids) <= 3:
        best_hybrid_legend = f"Best Hybrid (Const {', '.join(map(str, unique_best_hybrids))})"
    else:
        best_hybrid_legend = f"Best Hybrid (Const {unique_best_hybrids[0]}-{unique_best_hybrids[-1]})"

    bars1 = ax.bar(x - width, const1_vals, width, label='Const 1 (All Chained)',
                  color='#d62728', alpha=0.8)
    bars2 = ax.bar(x, const16_vals, width, label='Const 16 (All Independent)',
                  color='#2ca02c', alpha=0.8)
    bars3 = ax.bar(x + width, best_hybrid_vals, width, label=best_hybrid_legend,
                  color='#1f77b4', alpha=0.8)

    max_val = max(const1_vals + const16_vals + best_hybrid_vals) or 1
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                fmt = f'{height:.1f}' if use_days else f'{height:.2f}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       fmt, ha='center', va='bottom', fontsize=8)

    for i, (target_length, label) in enumerate(zip(target_lengths, best_hybrid_labels)):
        if best_hybrid_vals[i] > 0:
            ax.text(i + width, best_hybrid_vals[i] + max_val * 0.02,
                   label, ha='center', va='bottom', fontsize=8, style='italic')

    ax.set_xlabel("Target Job Length", fontsize=12)
    ax.set_ylabel(f"Turnaround Time ({unit})", fontsize=12)
    ax.set_title("Turnaround Time: Best Hybrid vs. Extremes", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(target_lengths)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "turnaround_time_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_path}")


def plot_failure_cost_analysis(data_by_composition: Dict[int, List[Dict[str, Any]]],
                               best_hybrids: Dict[str, Optional[int]],
                               output_dir: str) -> None:
    """Plot failure cost analysis: cost per failure vs target job length.

    Args:
        data_by_composition: Dictionary mapping composition_number to metrics list
        best_hybrids: Dictionary mapping target_length to best hybrid composition number
        output_dir: Output directory for plots
    """
    print(f"==> Creating failure cost analysis plot")

    all_target_lengths = set()
    for comp_data in data_by_composition.values():
        for d in comp_data:
            all_target_lengths.add(d['target_job_length'])
    target_lengths = sorted(all_target_lengths, key=target_length_to_hours)
    xtick_labels, tl_to_x = get_target_length_xconfig(target_lengths, include_zero=False)

    # Check if we have failure data (fr25)
    has_failures = False
    for comp_data in data_by_composition.values():
        for d in comp_data:
            if d.get('total_failed_jobs', 0) > 0:
                has_failures = True
                break
        if has_failures:
            break

    if not has_failures:
        print("  => Skipping (no failure data - likely fr0)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    hybrid_color_map = get_best_hybrid_colors(best_hybrids)

    # Plot 1: Average cost per failure (CPU time only - wall time matches target length and is redundant)
    # First pass: Const 1, Const 16, and non-best-hybrid constructions
    best_hybrid_comp_nums = [best_hybrids[tl] for tl in target_lengths if best_hybrids[tl] is not None]
    best_hybrid_comp_nums = sorted(set(best_hybrid_comp_nums))

    for comp_num in sorted(data_by_composition.keys()):
        if comp_num in best_hybrid_comp_nums:
            continue
        comp_data = data_by_composition[comp_num]
        comp_data_sorted = sorted(comp_data, key=lambda x: target_length_to_hours(x['target_job_length']))
        target_length_values = [d['target_job_length'] for d in comp_data_sorted]
        avg_cpu_per_failure = [d.get('avg_cpu_per_failure', 0.0) / 3600.0 for d in comp_data_sorted]
        x_positions = [tl_to_x[tl] for tl in target_length_values]
        label = f"Const {comp_num}"
        if comp_num == 1:
            ax1.plot(x_positions, avg_cpu_per_failure, 'o-', label=label,
                    linewidth=2.5, color='#d62728', markersize=8, zorder=10)
        elif comp_num == 16:
            ax1.plot(x_positions, avg_cpu_per_failure, 's-', label=label,
                    linewidth=2.5, color='#2ca02c', markersize=8, zorder=10)

    # Second pass: best hybrid(s) drawn last with dashed line for visibility
    for comp_num in best_hybrid_comp_nums:
        if comp_num not in data_by_composition:
            continue
        comp_data = data_by_composition[comp_num]
        comp_data_sorted = sorted(comp_data, key=lambda x: target_length_to_hours(x['target_job_length']))
        target_length_values = [d['target_job_length'] for d in comp_data_sorted]
        avg_cpu_per_failure = [d.get('avg_cpu_per_failure', 0.0) / 3600.0 for d in comp_data_sorted]
        x_positions = [tl_to_x[tl] for tl in target_length_values]
        hybrid_color = hybrid_color_map.get(comp_num, '#1f77b4')
        ax1.plot(x_positions, avg_cpu_per_failure, '^--', label=f"Best Hybrid (C{comp_num})",
                linewidth=2.5, color=hybrid_color, markersize=8, zorder=15, alpha=1.0)

    ax1.set_xlabel("Target Job Length", fontsize=12)
    ax1.set_ylabel("Average CPU Cost per Failure (CPU-hours)", fontsize=12)
    ax1.set_title("Average Cost per Failure vs. Target Job Length", fontsize=13)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(len(xtick_labels)))
    ax1.set_xticklabels(xtick_labels)

    # Plot 2: Risk profile (max single failure cost - CPU time only)
    # First pass: Const 1, Const 16, and non-best-hybrid constructions
    for comp_num in sorted(data_by_composition.keys()):
        if comp_num in best_hybrid_comp_nums:
            continue
        comp_data = data_by_composition[comp_num]
        comp_data_sorted = sorted(comp_data, key=lambda x: target_length_to_hours(x['target_job_length']))
        target_length_values = [d['target_job_length'] for d in comp_data_sorted]
        max_cpu_per_failure = [d.get('max_cpu_per_failure', 0.0) / 3600.0 for d in comp_data_sorted]
        x_positions = [tl_to_x[tl] for tl in target_length_values]
        label = f"Const {comp_num}"
        if comp_num == 1:
            ax2.plot(x_positions, max_cpu_per_failure, 'o-', label=label,
                    linewidth=2.5, color='#d62728', markersize=8, zorder=10)
        elif comp_num == 16:
            ax2.plot(x_positions, max_cpu_per_failure, 's-', label=label,
                    linewidth=2.5, color='#2ca02c', markersize=8, zorder=10)

    # Second pass: best hybrid(s) drawn last with dashed line for visibility
    for comp_num in best_hybrid_comp_nums:
        if comp_num not in data_by_composition:
            continue
        comp_data = data_by_composition[comp_num]
        comp_data_sorted = sorted(comp_data, key=lambda x: target_length_to_hours(x['target_job_length']))
        target_length_values = [d['target_job_length'] for d in comp_data_sorted]
        max_cpu_per_failure = [d.get('max_cpu_per_failure', 0.0) / 3600.0 for d in comp_data_sorted]
        x_positions = [tl_to_x[tl] for tl in target_length_values]
        hybrid_color = hybrid_color_map.get(comp_num, '#1f77b4')
        ax2.plot(x_positions, max_cpu_per_failure, '^--', label=f"Best Hybrid (C{comp_num})",
                linewidth=2.5, color=hybrid_color, markersize=8, zorder=15, alpha=1.0)

    ax2.set_xlabel("Target Job Length", fontsize=12)
    ax2.set_ylabel("Max Single Failure Cost (CPU-hours)", fontsize=12)
    ax2.set_title("Risk Profile: Max Single Failure Cost vs. Target Job Length", fontsize=13)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(len(xtick_labels)))
    ax2.set_xticklabels(xtick_labels)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "failure_cost_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_path}")


def plot_failure_count_analysis(data_by_composition: Dict[int, List[Dict[str, Any]]],
                                best_hybrids: Dict[str, Optional[int]],
                                output_dir: str) -> None:
    """Plot failure count distribution vs target job length.

    Args:
        data_by_composition: Dictionary mapping composition_number to metrics list
        best_hybrids: Dictionary mapping target_length to best hybrid composition number
        output_dir: Output directory for plots
    """
    print(f"==> Creating failure count analysis plot")

    all_target_lengths = set()
    for comp_data in data_by_composition.values():
        for d in comp_data:
            all_target_lengths.add(d['target_job_length'])
    target_lengths = sorted(all_target_lengths, key=target_length_to_hours)
    xtick_labels, tl_to_x = get_target_length_xconfig(target_lengths, include_zero=False)

    # Check if we have failure data
    has_failures = False
    for comp_data in data_by_composition.values():
        for d in comp_data:
            if d.get('total_failed_jobs', 0) > 0:
                has_failures = True
                break
        if has_failures:
            break

    if not has_failures:
        print("  => Skipping (no failure data - likely fr0)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    hybrid_color_map = get_best_hybrid_colors(best_hybrids)
    best_hybrid_comp_nums = sorted(set([best_hybrids[tl] for tl in target_lengths if best_hybrids[tl] is not None]))

    # Plot 1: Failure count vs target job length
    for comp_num in sorted(data_by_composition.keys()):
        if comp_num in best_hybrid_comp_nums:
            continue
        comp_data = data_by_composition[comp_num]
        comp_data_sorted = sorted(comp_data, key=lambda x: target_length_to_hours(x['target_job_length']))
        target_length_values = [d['target_job_length'] for d in comp_data_sorted]
        failed_counts = [d.get('total_failed_jobs', 0) for d in comp_data_sorted]
        x_positions = [tl_to_x[tl] for tl in target_length_values]
        label = f"Const {comp_num}"
        if comp_num == 1:
            ax1.plot(x_positions, failed_counts, 'o-', label=label,
                    linewidth=2.5, color='#d62728', markersize=8, zorder=10)
        elif comp_num == 16:
            ax1.plot(x_positions, failed_counts, 's-', label=label,
                    linewidth=2.5, color='#2ca02c', markersize=8, zorder=10)

    for comp_num in best_hybrid_comp_nums:
        if comp_num not in data_by_composition:
            continue
        comp_data = data_by_composition[comp_num]
        comp_data_sorted = sorted(comp_data, key=lambda x: target_length_to_hours(x['target_job_length']))
        target_length_values = [d['target_job_length'] for d in comp_data_sorted]
        failed_counts = [d.get('total_failed_jobs', 0) for d in comp_data_sorted]
        x_positions = [tl_to_x[tl] for tl in target_length_values]
        hybrid_color = hybrid_color_map.get(comp_num, '#1f77b4')
        ax1.plot(x_positions, failed_counts, '^--', label=f"Best Hybrid (C{comp_num})",
                linewidth=2.5, color=hybrid_color, markersize=8, zorder=15, alpha=1.0)

    ax1.set_xlabel("Target Job Length", fontsize=12)
    ax1.set_ylabel("Number of Failed Jobs", fontsize=12)
    ax1.set_title("Failure Count vs. Target Job Length", fontsize=13)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(len(xtick_labels)))
    ax1.set_xticklabels(xtick_labels)

    # Plot 2: Failure rate (actual) vs target job length
    for comp_num in sorted(data_by_composition.keys()):
        if comp_num in best_hybrid_comp_nums:
            continue
        comp_data = data_by_composition[comp_num]
        comp_data_sorted = sorted(comp_data, key=lambda x: target_length_to_hours(x['target_job_length']))
        target_length_values = [d['target_job_length'] for d in comp_data_sorted]
        failure_rate_actual = [d.get('failure_rate_actual', 0.0) for d in comp_data_sorted]
        x_positions = [tl_to_x[tl] for tl in target_length_values]
        label = f"Const {comp_num}"
        if comp_num == 1:
            ax2.plot(x_positions, failure_rate_actual, 'o-', label=label,
                    linewidth=2.5, color='#d62728', markersize=8, zorder=10)
        elif comp_num == 16:
            ax2.plot(x_positions, failure_rate_actual, 's-', label=label,
                    linewidth=2.5, color='#2ca02c', markersize=8, zorder=10)

    for comp_num in best_hybrid_comp_nums:
        if comp_num not in data_by_composition:
            continue
        comp_data = data_by_composition[comp_num]
        comp_data_sorted = sorted(comp_data, key=lambda x: target_length_to_hours(x['target_job_length']))
        target_length_values = [d['target_job_length'] for d in comp_data_sorted]
        failure_rate_actual = [d.get('failure_rate_actual', 0.0) for d in comp_data_sorted]
        x_positions = [tl_to_x[tl] for tl in target_length_values]
        hybrid_color = hybrid_color_map.get(comp_num, '#1f77b4')
        ax2.plot(x_positions, failure_rate_actual, '^--', label=f"Best Hybrid (C{comp_num})",
                linewidth=2.5, color=hybrid_color, markersize=8, zorder=15, alpha=1.0)

    ax2.set_xlabel("Target Job Length", fontsize=12)
    ax2.set_ylabel("Actual Failure Rate (%)", fontsize=12)
    ax2.set_title("Actual Failure Rate vs. Target Job Length", fontsize=13)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(len(xtick_labels)))
    ax2.set_xticklabels(xtick_labels)
    # Add horizontal line at expected failure rate if available
    if data_by_composition:
        first_data = next(iter(data_by_composition.values()))[0]
        expected_fr = first_data.get('failure_rate', 0.0)
        if expected_fr > 0:
            ax2.axhline(y=expected_fr, color='gray', linestyle='--', linewidth=1, alpha=0.5,
                       label=f'Expected ({expected_fr:.1f}%)')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=9)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "failure_count_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_path}")


def generate_summary_table(data_by_composition: Dict[int, List[Dict[str, Any]]],
                          output_dir: str) -> pd.DataFrame:
    """Generate summary table with metrics across target job lengths.

    Args:
        data_by_composition: Dictionary mapping composition_number to metrics list
        output_dir: Output directory for table

    Returns:
        DataFrame with summary metrics
    """
    print(f"==> Generating summary table")

    table_data = []

    for comp_num in sorted(data_by_composition.keys()):
        comp_data = sorted(data_by_composition[comp_num],
                          key=lambda x: target_length_to_hours(x['target_job_length']))

        for metrics in comp_data:
            row = {
                'Composition': comp_num,
                'Target_Job_Length': metrics['target_job_length'],
                'Event_Throughput': metrics['event_throughput'],
                'Wall_Time_Per_Event': metrics['wall_time_per_event'],
                'CPU_Time_Per_Event': metrics['cpu_time_per_event'],
                'Network_Transfer_MB_Per_Event': metrics['network_transfer_mb_per_event'],
                'CPU_Utilization': metrics['cpu_utilization'],
                'Memory_Occupancy': metrics['memory_occupancy'],
                'Total_Groups': metrics['total_groups'],
                # Failure metrics (counts from metrics/simulation_result)
                'Total_Jobs': metrics.get('total_jobs', 0),
                'Total_Job_Retries': metrics.get('total_job_retries', 0),
                'Total_Logical_Jobs': metrics.get('total_logical_jobs', 0),
                'Total_Failed_Jobs': metrics.get('total_failed_jobs', 0),
                'Failure_Rate_Actual_%': metrics.get('failure_rate_actual', 0.0),
                'Total_Wasted_CPU_Time_s': metrics.get('total_wasted_cpu_time', 0.0),
                'Total_Wasted_Wall_Time_s': metrics.get('total_wasted_wall_time', 0.0),
                'Total_Wasted_Network_MB': metrics.get('total_wasted_network_mb', 0.0),
                'Avg_CPU_Per_Failure_s': metrics.get('avg_cpu_per_failure', 0.0),
                'Avg_Wall_Per_Failure_s': metrics.get('avg_wall_per_failure', 0.0),
                'Avg_Network_Per_Failure_MB': metrics.get('avg_network_per_failure', 0.0),
                'Max_CPU_Per_Failure_s': metrics.get('max_cpu_per_failure', 0.0),
                'Max_Wall_Per_Failure_s': metrics.get('max_wall_per_failure', 0.0),
                'Max_Network_Per_Failure_MB': metrics.get('max_network_per_failure', 0.0)
            }
            table_data.append(row)

    df = pd.DataFrame(table_data)

    # Save as CSV
    csv_path = os.path.join(output_dir, "target_job_length_analysis_summary.csv")
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  => Saved: {csv_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Analyze workflow construction performance across target job lengths'
    )
    parser.add_argument('base_path', type=str,
                       help='Base path to results directory (e.g., results/sim/others)')
    parser.add_argument('workflow_type', type=str,
                       help='Workflow type (e.g., case1_real)')
    parser.add_argument('failure_rate', type=str,
                       help='Failure rate directory (e.g., fr0)')
    parser.add_argument('--data-rate', type=str, default='100MBps',
                       help='Data transfer rate directory (default: 100MBps)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: results/analysis/target_job_length/{workflow_type}/{failure_rate})')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"results/analysis/target_job_length/{args.workflow_type}/{args.failure_rate}"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Target Job Length Optimization Analysis")
    print("="*70)
    print(f"Workflow Type: {args.workflow_type}")
    print(f"Failure Rate: {args.failure_rate}")
    print(f"Data Rate: {args.data_rate}")
    print(f"Output Directory: {args.output_dir}")
    print("="*70)

    data_by_composition = collect_data_from_directories(
        args.base_path,
        args.workflow_type,
        args.failure_rate,
        data_rate=args.data_rate
    )

    if not data_by_composition:
        print("Error: No data collected. Please check directory paths and file availability.")
        return

    print(f"\nCollected data for {len(data_by_composition)} constructions")

    best_hybrids = identify_best_hybrid_per_target_length(data_by_composition, verbose=True)

    plot_throughput_vs_target_length(data_by_composition, best_hybrids, args.output_dir)
    plot_throughput_improvement(data_by_composition, args.output_dir)
    plot_network_activity_vs_target_length(data_by_composition, best_hybrids, args.output_dir)
    plot_best_hybrid_comparison(data_by_composition, best_hybrids, args.output_dir)
    plot_total_jobs_comparison(data_by_composition, best_hybrids, args.output_dir)
    plot_turnaround_time_comparison(data_by_composition, best_hybrids, args.output_dir)
    plot_failure_cost_analysis(data_by_composition, best_hybrids, args.output_dir)
    plot_failure_count_analysis(data_by_composition, best_hybrids, args.output_dir)
    generate_summary_table(data_by_composition, args.output_dir)

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
