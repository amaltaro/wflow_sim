#!/usr/bin/env python3
"""
Target Job Length Optimization Analysis Script

This script analyzes how different workflow constructions (1-16) perform across
various target job lengths. It aggregates data from multiple directories to create
cross-dimensional comparisons.

Analysis: Target Job Length Optimization (Comparison #3)
- Fixed: workflow_type + failure_rate
- Variable: target_job_length (1h, 6h, 12h, 18h, 24h)
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

        # Extract failure-related metrics
        first_attempt_jobs = [j for j in jobs if j.get('retry_count', 0) == 0]
        failed_jobs = [j for j in first_attempt_jobs if j.get('status') == 'failed']

        # Calculate failure cost metrics
        total_failed_jobs = len(failed_jobs)
        total_wasted_cpu = sum(j.get('total_cpu_used_time', 0.0) for j in failed_jobs)
        total_wasted_wall = sum(j.get('wallclock_time', 0.0) for j in failed_jobs)
        total_wasted_network = sum(j.get('total_network_transfer_mb', 0.0) for j in failed_jobs)

        avg_cpu_per_failure = total_wasted_cpu / total_failed_jobs if total_failed_jobs > 0 else 0.0
        avg_wall_per_failure = total_wasted_wall / total_failed_jobs if total_failed_jobs > 0 else 0.0
        avg_network_per_failure = total_wasted_network / total_failed_jobs if total_failed_jobs > 0 else 0.0

        # Risk profile: max single failure cost
        max_cpu_failure = max((j.get('total_cpu_used_time', 0.0) for j in failed_jobs), default=0.0)
        max_wall_failure = max((j.get('wallclock_time', 0.0) for j in failed_jobs), default=0.0)
        max_network_failure = max((j.get('total_network_transfer_mb', 0.0) for j in failed_jobs), default=0.0)

        # Total logical jobs (excluding retries)
        total_logical_jobs = len(first_attempt_jobs)
        failure_rate_actual = (total_failed_jobs / total_logical_jobs * 100.0) if total_logical_jobs > 0 else 0.0

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
            'failure_rate': sim_result.get('failure_rate', 0.0),
            'overhead_enabled': sim_result.get('overhead_enabled', True),
            # Failure metrics
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
                                  overhead_type: str = "overhead") -> Dict[int, List[Dict[str, Any]]]:
    """Collect simulation data from multiple target job length directories.

    Args:
        base_path: Base path to results directory (e.g., 'results/sim/others')
        workflow_type: Workflow type (e.g., 'case1_real')
        failure_rate: Failure rate directory (e.g., 'fr0')
        overhead_type: 'overhead' or 'nooverhead'

    Returns:
        Dictionary mapping composition_number to list of metrics across target job lengths
    """
    base_dir = Path(base_path) / workflow_type

    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    # Expected target job length directories
    target_job_lengths = ['1h', '6h', '12h', '18h', '24h']
    suffix = "_overhead" if overhead_type == "overhead" else "_nooverhead"

    # Dictionary: composition_number -> list of metrics (one per target job length)
    data_by_composition: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    print(f"Collecting data from: {base_dir}")
    print(f"Failure rate: {failure_rate}")
    print(f"Overhead type: {overhead_type}")

    for target_length in target_job_lengths:
        target_path = base_dir / target_length / failure_rate
        if not target_path.exists():
            print(f"  Warning: Directory {target_path} not found, skipping")
            continue

        # Find all JSON files for this target job length
        json_files = list(target_path.glob(f"*{suffix}.json"))
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
        target_length: Target job length string (e.g., '1h', '12h')

    Returns:
        Hours as float
    """
    return float(target_length.replace('h', ''))


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
                                     output_dir: str,
                                     overhead_type: str) -> None:
    """Plot event throughput vs. target job length for all constructions.

    Args:
        data_by_composition: Dictionary mapping composition_number to metrics list
        output_dir: Output directory for plots
        overhead_type: 'overhead' or 'nooverhead'
    """
    print(f"\n==> Creating throughput vs. target job length plot")

    fig, ax = plt.subplots(figsize=(14, 8))

    # Extract target lengths and throughput for each construction
    target_lengths = []
    for comp_num in sorted(data_by_composition.keys()):
        comp_data = data_by_composition[comp_num]
        # Sort by target job length (convert to hours for sorting)
        comp_data_sorted = sorted(comp_data, key=lambda x: target_length_to_hours(x['target_job_length']))
        target_length_values = [d['target_job_length'] for d in comp_data_sorted]
        throughput_values = [d['event_throughput'] for d in comp_data_sorted]

        if not target_lengths:
            target_lengths = target_length_values

    # Find best hybrid for each target length
    best_hybrids = {}
    for target_length in target_lengths:
        best_hybrids[target_length] = identify_best_hybrid(data_by_composition, target_length)

    # Get color mapping for best hybrids
    hybrid_color_map = get_best_hybrid_colors(best_hybrids)

    # Convert target lengths to hours for x-axis
    target_hours = [target_length_to_hours(tl) for tl in target_lengths]

    # Plot lines for all constructions
    for comp_num in sorted(data_by_composition.keys()):
        comp_data = data_by_composition[comp_num]
        comp_data_sorted = sorted(comp_data, key=lambda x: target_length_to_hours(x['target_job_length']))
        target_length_values = [d['target_job_length'] for d in comp_data_sorted]
        throughput_values = [d['event_throughput'] for d in comp_data_sorted]
        target_hours_values = [target_length_to_hours(tl) for tl in target_length_values]

        # Plot line for this construction
        label = f"Const {comp_num}"
        if comp_num == 1:
            # Highlight Const 1 (all chained)
            ax.plot(target_hours_values, throughput_values, 'o-', label=label, linewidth=2.5,
                   color='#d62728', markersize=8, zorder=10)
        elif comp_num == 16:
            # Highlight Const 16 (all independent)
            ax.plot(target_hours_values, throughput_values, 's-', label=label, linewidth=2.5,
                   color='#2ca02c', markersize=8, zorder=10)
        else:
            # Check if this is the best hybrid for any target length
            is_best_hybrid = any(best_hybrids[tl] == comp_num for tl in target_lengths if best_hybrids[tl] is not None)

            if is_best_hybrid:
                # Get unique color for this best hybrid
                hybrid_color = hybrid_color_map.get(comp_num, '#1f77b4')
                # Highlight best hybrid with triangle markers
                marker_targets = [tl for tl in target_length_values if best_hybrids.get(tl) == comp_num]
                marker_throughput = [th for tl, th in zip(target_length_values, throughput_values)
                                    if best_hybrids.get(tl) == comp_num]

                # Plot the full line
                ax.plot(target_hours_values, throughput_values, '-', label=label, linewidth=1.5,
                       alpha=0.7, markersize=5, zorder=5)
                # Add triangle markers at best hybrid points with unique color
                if marker_targets:
                    marker_hours = [target_length_to_hours(tl) for tl in marker_targets]
                    ax.plot(marker_hours, marker_throughput, '^', label=None, linewidth=0,
                           color=hybrid_color, markersize=10, zorder=11, alpha=0.9,
                           markerfacecolor=hybrid_color, markeredgecolor='white', markeredgewidth=1.5)
            else:
                # Regular hybrid constructions
                ax.plot(target_hours_values, throughput_values, '-', label=label, linewidth=1.5,
                       alpha=0.7, markersize=5)

    # Format overhead label for title
    overhead_label = "With overhead" if overhead_type == "overhead" else "No overhead"

    ax.set_xlabel("Target Job Length (hours)", fontsize=12)
    ax.set_ylabel("Event Throughput (events/second)", fontsize=12)
    ax.set_title(f"Event Throughput vs. Target Job Length\n({overhead_label})", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(target_hours)
    ax.set_xticklabels([f"{int(h)}h" for h in target_hours])

    plt.tight_layout()
    suffix = "_nooverhead" if overhead_type == "nooverhead" else "_overhead"
    filename = f"throughput_vs_target_length{suffix}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_path}")


def plot_throughput_improvement(data_by_composition: Dict[int, List[Dict[str, Any]]],
                                output_dir: str,
                                overhead_type: str) -> None:
    """Plot throughput improvement (relative to shortest target length) vs. target job length.

    Args:
        data_by_composition: Dictionary mapping composition_number to metrics list
        output_dir: Output directory for plots
        overhead_type: 'overhead' or 'nooverhead'
    """
    print(f"\n==> Creating throughput improvement plot")

    fig, ax = plt.subplots(figsize=(14, 8))

    # Get all target lengths and sort them
    all_target_lengths = set()
    for comp_data in data_by_composition.values():
        for d in comp_data:
            all_target_lengths.add(d['target_job_length'])
    target_lengths = sorted(all_target_lengths, key=target_length_to_hours)
    target_hours = [target_length_to_hours(tl) for tl in target_lengths]

    for comp_num in sorted(data_by_composition.keys()):
        comp_data = data_by_composition[comp_num]
        comp_data_sorted = sorted(comp_data, key=lambda x: target_length_to_hours(x['target_job_length']))

        # Find baseline (shortest target length) throughput
        baseline = comp_data_sorted[0] if comp_data_sorted else None
        if not baseline:
            continue

        baseline_throughput = baseline['event_throughput']
        if baseline_throughput == 0:
            continue

        # Calculate improvement percentage
        target_length_values = []
        improvement_values = []
        for d in comp_data_sorted:
            target_length_values.append(d['target_job_length'])
            improvement = ((d['event_throughput'] - baseline_throughput) / baseline_throughput) * 100
            improvement_values.append(improvement)

        target_hours_values = [target_length_to_hours(tl) for tl in target_length_values]

        # Plot line for this construction
        label = f"Const {comp_num}"
        if comp_num == 1:
            ax.plot(target_hours_values, improvement_values, 'o-', label=label, linewidth=2.5,
                   color='#d62728', markersize=8, zorder=10)
        elif comp_num == 16:
            ax.plot(target_hours_values, improvement_values, 's-', label=label, linewidth=2.5,
                   color='#2ca02c', markersize=8, zorder=10)
        else:
            ax.plot(target_hours_values, improvement_values, '-', label=label, linewidth=1.5,
                   alpha=0.7, markersize=5)

    # Format overhead label for title
    overhead_label = "With overhead" if overhead_type == "overhead" else "No overhead"

    ax.set_xlabel("Target Job Length (hours)", fontsize=12)
    ax.set_ylabel("Throughput Improvement (%)", fontsize=12)
    ax.set_title(f"Throughput Improvement vs. Target Job Length\n(Relative to 1h, {overhead_label})",
                fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(target_hours)
    ax.set_xticklabels([f"{int(h)}h" for h in target_hours])
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    suffix = "_nooverhead" if overhead_type == "nooverhead" else "_overhead"
    filename = f"throughput_improvement{suffix}.png"
    output_path = os.path.join(output_dir, filename)
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


def plot_network_activity_vs_target_length(data_by_composition: Dict[int, List[Dict[str, Any]]],
                                           output_dir: str,
                                           overhead_type: str) -> None:
    """Plot network transfer vs. target job length for all constructions.

    This visualization shows how network activity (remote I/O) changes with
    target job lengths, which is important for understanding workflow efficiency
    under different time constraints.

    Args:
        data_by_composition: Dictionary mapping composition_number to metrics list
        output_dir: Output directory for plots
        overhead_type: 'overhead' or 'nooverhead'
    """
    print(f"\n==> Creating network activity vs. target job length plot")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Format overhead label for title
    overhead_label = "With overhead" if overhead_type == "overhead" else "No overhead"

    # Get all target lengths and sort them
    all_target_lengths = set()
    for comp_data in data_by_composition.values():
        for d in comp_data:
            all_target_lengths.add(d['target_job_length'])
    target_lengths = sorted(all_target_lengths, key=target_length_to_hours)
    target_hours = [target_length_to_hours(tl) for tl in target_lengths]

    # Plot 1: Network Transfer per Event vs. Target Job Length
    for comp_num in sorted(data_by_composition.keys()):
        comp_data = data_by_composition[comp_num]
        comp_data_sorted = sorted(comp_data, key=lambda x: target_length_to_hours(x['target_job_length']))
        target_length_values = [d['target_job_length'] for d in comp_data_sorted]
        network_values = [d['network_transfer_mb_per_event'] for d in comp_data_sorted]
        target_hours_values = [target_length_to_hours(tl) for tl in target_length_values]

        # Plot line for this construction
        label = f"Const {comp_num}"
        if comp_num == 1:
            ax1.plot(target_hours_values, network_values, 'o-', label=label, linewidth=2.5,
                    color='#d62728', markersize=8, zorder=10)
        elif comp_num == 16:
            ax1.plot(target_hours_values, network_values, 's-', label=label, linewidth=2.5,
                    color='#2ca02c', markersize=8, zorder=10)
        else:
            ax1.plot(target_hours_values, network_values, '-', label=label, linewidth=1.5,
                    alpha=0.7, markersize=5)

    ax1.set_xlabel("Target Job Length (hours)", fontsize=12)
    ax1.set_ylabel("Network Transfer per Event (MB)", fontsize=12)
    ax1.set_title(f"Network Transfer vs. Target Job Length\n({overhead_label})", fontsize=13)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(target_hours)
    ax1.set_xticklabels([f"{int(h)}h" for h in target_hours])

    # Plot 2: Remote Read vs. Remote Write breakdown (focus on extremes and best hybrid)
    # Find best hybrid for each target length
    best_hybrids = {}
    for target_length in target_lengths:
        best_hybrids[target_length] = identify_best_hybrid(data_by_composition, target_length)

    # Get color mapping for best hybrids
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
        target_hours_values = [target_length_to_hours(tl) for tl in target_length_values]

        # Plot lines
        label = f"Const {comp_num}"
        if comp_num == 1:
            ax2.plot(target_hours_values, read_remote, 'o--', label=f"{label} (Read)", linewidth=2.5,
                    color='#d62728', markersize=7, zorder=10, alpha=0.7, markerfacecolor='#d62728',
                    markeredgecolor='#d62728', markeredgewidth=1.5)
            ax2.plot(target_hours_values, write_remote, 's-', label=f"{label} (Write)", linewidth=2.5,
                    color='#d62728', markersize=7, zorder=10, alpha=0.9, markerfacecolor='#d62728',
                    markeredgecolor='#d62728', markeredgewidth=1.5)
        elif comp_num == 16:
            ax2.plot(target_hours_values, read_remote, 'o--', label=f"{label} (Read)", linewidth=2.5,
                    color='#2ca02c', markersize=7, zorder=10, alpha=0.9, markerfacecolor='#2ca02c',
                    markeredgecolor='#2ca02c', markeredgewidth=1.5)
            ax2.plot(target_hours_values, write_remote, 's-', label=f"{label} (Write)", linewidth=2.5,
                    color='#2ca02c', markersize=7, zorder=10, alpha=0.9, markerfacecolor='#2ca02c',
                    markeredgecolor='#2ca02c', markeredgewidth=1.5)
        else:
            # Best hybrid with unique color
            hybrid_color = hybrid_color_map.get(comp_num, '#1f77b4')
            ax2.plot(target_hours_values, read_remote, 'o--', label=f"{label} (Read)", linewidth=2,
                    color=hybrid_color, markersize=6, zorder=9, alpha=0.8, markerfacecolor=hybrid_color,
                    markeredgecolor=hybrid_color, markeredgewidth=1.5)
            ax2.plot(target_hours_values, write_remote, 's-', label=f"{label} (Write)", linewidth=2,
                    color=hybrid_color, markersize=6, zorder=9, alpha=0.8, markerfacecolor=hybrid_color,
                    markeredgecolor=hybrid_color, markeredgewidth=1.5)

    ax2.set_xlabel("Target Job Length (hours)", fontsize=12)
    ax2.set_ylabel("Data Volume per Event (MB)", fontsize=12)
    ax2.set_title(f"Remote I/O Breakdown vs. Target Job Length\n({overhead_label})", fontsize=13)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(target_hours)
    ax2.set_xticklabels([f"{int(h)}h" for h in target_hours])

    plt.tight_layout()
    suffix = "_nooverhead" if overhead_type == "nooverhead" else "_overhead"
    filename = f"network_activity_vs_target_length{suffix}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_path}")


def plot_best_hybrid_comparison(data_by_composition: Dict[int, List[Dict[str, Any]]],
                                output_dir: str,
                                overhead_type: str) -> None:
    """Plot comparison of Const 1, Const 16, and best hybrid for each target job length.

    Args:
        data_by_composition: Dictionary mapping composition_number to metrics list
        output_dir: Output directory for plots
        overhead_type: 'overhead' or 'nooverhead'
    """
    print(f"\n==> Creating best hybrid comparison plot")

    # Get all target lengths from the data
    all_target_lengths = set()
    for comp_data in data_by_composition.values():
        for d in comp_data:
            all_target_lengths.add(d['target_job_length'])
    target_lengths = sorted(all_target_lengths, key=target_length_to_hours)
    target_hours = [target_length_to_hours(tl) for tl in target_lengths]

    # Find best hybrid for each target length (each target length can have a different best hybrid)
    best_hybrids = {}
    for target_length in target_lengths:
        best_hybrid = identify_best_hybrid(data_by_composition, target_length, verbose=True)
        best_hybrids[target_length] = best_hybrid
        if best_hybrid:
            print(f"  Best hybrid for {target_length}: Const {best_hybrid}")

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
            best_hybrid_labels.append(f"Const {best_comp}")
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

    # Format overhead label for title
    overhead_label = "With overhead" if overhead_type == "overhead" else "No overhead"

    ax.set_xlabel("Target Job Length", fontsize=12)
    ax.set_ylabel("Event Throughput (events/second)", fontsize=12)
    ax.set_title(f"Best Hybrid vs. Extremes Comparison\n({overhead_label})", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(target_lengths)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    suffix = "_nooverhead" if overhead_type == "nooverhead" else "_overhead"
    filename = f"best_hybrid_comparison{suffix}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_path}")


def plot_failure_cost_analysis(data_by_composition: Dict[int, List[Dict[str, Any]]],
                               output_dir: str,
                               overhead_type: str) -> None:
    """Plot failure cost analysis: cost per failure vs target job length.

    Args:
        data_by_composition: Dictionary mapping composition_number to metrics list
        output_dir: Output directory for plots
        overhead_type: 'overhead' or 'nooverhead'
    """
    print(f"\n==> Creating failure cost analysis plot")

    # Get all target lengths and sort them
    all_target_lengths = set()
    for comp_data in data_by_composition.values():
        for d in comp_data:
            all_target_lengths.add(d['target_job_length'])
    target_lengths = sorted(all_target_lengths, key=target_length_to_hours)
    target_hours = [target_length_to_hours(tl) for tl in target_lengths]

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

    # Format overhead label for title
    overhead_label = "With overhead" if overhead_type == "overhead" else "No overhead"

    # Get color mapping for best hybrids (each target length can have a different best hybrid)
    best_hybrids = {}
    for tl in target_lengths:
        best_hybrid = identify_best_hybrid(data_by_composition, tl, verbose=True)
        best_hybrids[tl] = best_hybrid
        if best_hybrid:
            print(f"  Best hybrid for {tl}: Const {best_hybrid}")
    hybrid_color_map = get_best_hybrid_colors(best_hybrids)

    # Plot 1: Average cost per failure (CPU time only - wall time matches target length and is redundant)
    for comp_num in sorted(data_by_composition.keys()):
        comp_data = data_by_composition[comp_num]
        comp_data_sorted = sorted(comp_data, key=lambda x: target_length_to_hours(x['target_job_length']))

        target_length_values = [d['target_job_length'] for d in comp_data_sorted]
        avg_cpu_per_failure = [d.get('avg_cpu_per_failure', 0.0) / 3600.0 for d in comp_data_sorted]  # Convert to hours
        target_hours_values = [target_length_to_hours(tl) for tl in target_length_values]

        label = f"Const {comp_num}"
        if comp_num == 1:
            ax1.plot(target_hours_values, avg_cpu_per_failure, 'o-', label=label,
                    linewidth=2.5, color='#d62728', markersize=8, zorder=10)
        elif comp_num == 16:
            ax1.plot(target_hours_values, avg_cpu_per_failure, 's-', label=label,
                    linewidth=2.5, color='#2ca02c', markersize=8, zorder=10)
        else:
            # Only show best hybrid with unique colors
            if any(best_hybrids[tl] == comp_num for tl in target_lengths if best_hybrids[tl] is not None):
                hybrid_color = hybrid_color_map.get(comp_num, '#1f77b4')
                ax1.plot(target_hours_values, avg_cpu_per_failure, '^-', label=label,
                        linewidth=2, color=hybrid_color, markersize=6, zorder=9, alpha=0.8)

    ax1.set_xlabel("Target Job Length (hours)", fontsize=12)
    ax1.set_ylabel("Average CPU Cost per Failure (CPU-hours)", fontsize=12)
    ax1.set_title(f"Average Cost per Failure vs. Target Job Length\n({overhead_label})", fontsize=13)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(target_hours)
    ax1.set_xticklabels([f"{int(h)}h" for h in target_hours])

    # Plot 2: Risk profile (max single failure cost - CPU time only)
    for comp_num in sorted(data_by_composition.keys()):
        comp_data = data_by_composition[comp_num]
        comp_data_sorted = sorted(comp_data, key=lambda x: target_length_to_hours(x['target_job_length']))

        target_length_values = [d['target_job_length'] for d in comp_data_sorted]
        max_cpu_per_failure = [d.get('max_cpu_per_failure', 0.0) / 3600.0 for d in comp_data_sorted]  # Convert to hours
        target_hours_values = [target_length_to_hours(tl) for tl in target_length_values]

        label = f"Const {comp_num}"
        if comp_num == 1:
            ax2.plot(target_hours_values, max_cpu_per_failure, 'o-', label=label,
                    linewidth=2.5, color='#d62728', markersize=8, zorder=10)
        elif comp_num == 16:
            ax2.plot(target_hours_values, max_cpu_per_failure, 's-', label=label,
                    linewidth=2.5, color='#2ca02c', markersize=8, zorder=10)
        else:
            # Only show best hybrid with unique colors
            if any(best_hybrids[tl] == comp_num for tl in target_lengths if best_hybrids[tl] is not None):
                hybrid_color = hybrid_color_map.get(comp_num, '#1f77b4')
                ax2.plot(target_hours_values, max_cpu_per_failure, '^-', label=label,
                        linewidth=2, color=hybrid_color, markersize=6, zorder=9, alpha=0.8)

    ax2.set_xlabel("Target Job Length (hours)", fontsize=12)
    ax2.set_ylabel("Max Single Failure Cost (CPU-hours)", fontsize=12)
    ax2.set_title(f"Risk Profile: Max Single Failure Cost vs. Target Job Length\n({overhead_label})", fontsize=13)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(target_hours)
    ax2.set_xticklabels([f"{int(h)}h" for h in target_hours])

    plt.tight_layout()
    suffix = "_nooverhead" if overhead_type == "nooverhead" else "_overhead"
    filename = f"failure_cost_analysis{suffix}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_path}")


def plot_failure_count_analysis(data_by_composition: Dict[int, List[Dict[str, Any]]],
                                output_dir: str,
                                overhead_type: str) -> None:
    """Plot failure count distribution vs target job length.

    Args:
        data_by_composition: Dictionary mapping composition_number to metrics list
        output_dir: Output directory for plots
        overhead_type: 'overhead' or 'nooverhead'
    """
    print(f"\n==> Creating failure count analysis plot")

    # Get all target lengths and sort them
    all_target_lengths = set()
    for comp_data in data_by_composition.values():
        for d in comp_data:
            all_target_lengths.add(d['target_job_length'])
    target_lengths = sorted(all_target_lengths, key=target_length_to_hours)
    target_hours = [target_length_to_hours(tl) for tl in target_lengths]

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

    # Format overhead label for title
    overhead_label = "With overhead" if overhead_type == "overhead" else "No overhead"

    # Get color mapping for best hybrids (each target length can have a different best hybrid)
    best_hybrids = {}
    for tl in target_lengths:
        best_hybrid = identify_best_hybrid(data_by_composition, tl, verbose=True)
        best_hybrids[tl] = best_hybrid
        if best_hybrid:
            print(f"  Best hybrid for {tl}: Const {best_hybrid}")
    hybrid_color_map = get_best_hybrid_colors(best_hybrids)

    # Plot 1: Failure count vs target job length
    for comp_num in sorted(data_by_composition.keys()):
        comp_data = data_by_composition[comp_num]
        comp_data_sorted = sorted(comp_data, key=lambda x: target_length_to_hours(x['target_job_length']))

        target_length_values = [d['target_job_length'] for d in comp_data_sorted]
        failed_counts = [d.get('total_failed_jobs', 0) for d in comp_data_sorted]
        total_jobs = [d.get('total_logical_jobs', 0) for d in comp_data_sorted]
        failure_rate_actual = [d.get('failure_rate_actual', 0.0) for d in comp_data_sorted]
        target_hours_values = [target_length_to_hours(tl) for tl in target_length_values]

        label = f"Const {comp_num}"
        if comp_num == 1:
            ax1.plot(target_hours_values, failed_counts, 'o-', label=label,
                    linewidth=2.5, color='#d62728', markersize=8, zorder=10)
        elif comp_num == 16:
            ax1.plot(target_hours_values, failed_counts, 's-', label=label,
                    linewidth=2.5, color='#2ca02c', markersize=8, zorder=10)
        else:
            # Only show best hybrid with unique colors
            if any(best_hybrids[tl] == comp_num for tl in target_lengths if best_hybrids[tl] is not None):
                hybrid_color = hybrid_color_map.get(comp_num, '#1f77b4')
                ax1.plot(target_hours_values, failed_counts, '^-', label=label,
                        linewidth=2, color=hybrid_color, markersize=6, zorder=9, alpha=0.8)

    ax1.set_xlabel("Target Job Length (hours)", fontsize=12)
    ax1.set_ylabel("Number of Failed Jobs", fontsize=12)
    ax1.set_title(f"Failure Count vs. Target Job Length\n({overhead_label})", fontsize=13)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(target_hours)
    ax1.set_xticklabels([f"{int(h)}h" for h in target_hours])

    # Plot 2: Failure rate (actual) vs target job length
    for comp_num in sorted(data_by_composition.keys()):
        comp_data = data_by_composition[comp_num]
        comp_data_sorted = sorted(comp_data, key=lambda x: target_length_to_hours(x['target_job_length']))

        target_length_values = [d['target_job_length'] for d in comp_data_sorted]
        failure_rate_actual = [d.get('failure_rate_actual', 0.0) for d in comp_data_sorted]
        target_hours_values = [target_length_to_hours(tl) for tl in target_length_values]

        label = f"Const {comp_num}"
        if comp_num == 1:
            ax2.plot(target_hours_values, failure_rate_actual, 'o-', label=label,
                    linewidth=2.5, color='#d62728', markersize=8, zorder=10)
        elif comp_num == 16:
            ax2.plot(target_hours_values, failure_rate_actual, 's-', label=label,
                    linewidth=2.5, color='#2ca02c', markersize=8, zorder=10)
        else:
            # Only show best hybrid with unique colors
            if any(best_hybrids[tl] == comp_num for tl in target_lengths if best_hybrids[tl] is not None):
                hybrid_color = hybrid_color_map.get(comp_num, '#1f77b4')
                ax2.plot(target_hours_values, failure_rate_actual, '^-', label=label,
                        linewidth=2, color=hybrid_color, markersize=6, zorder=9, alpha=0.8)

    ax2.set_xlabel("Target Job Length (hours)", fontsize=12)
    ax2.set_ylabel("Actual Failure Rate (%)", fontsize=12)
    ax2.set_title(f"Actual Failure Rate vs. Target Job Length\n({overhead_label})", fontsize=13)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(target_hours)
    ax2.set_xticklabels([f"{int(h)}h" for h in target_hours])
    # Add horizontal line at expected failure rate if available
    if data_by_composition:
        first_data = next(iter(data_by_composition.values()))[0]
        expected_fr = first_data.get('failure_rate', 0.0)
        if expected_fr > 0:
            ax2.axhline(y=expected_fr, color='gray', linestyle='--', linewidth=1, alpha=0.5,
                       label=f'Expected ({expected_fr:.1f}%)')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=9)

    plt.tight_layout()
    suffix = "_nooverhead" if overhead_type == "nooverhead" else "_overhead"
    filename = f"failure_count_analysis{suffix}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_path}")


def generate_summary_table(data_by_composition: Dict[int, List[Dict[str, Any]]],
                          output_dir: str,
                          overhead_type: str) -> pd.DataFrame:
    """Generate summary table with metrics across target job lengths.

    Args:
        data_by_composition: Dictionary mapping composition_number to metrics list
        output_dir: Output directory for table
        overhead_type: 'overhead' or 'nooverhead'

    Returns:
        DataFrame with summary metrics
    """
    print(f"\n==> Generating summary table")

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
                # Failure metrics
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
    suffix = "_nooverhead" if overhead_type == "nooverhead" else "_overhead"
    csv_filename = f"target_job_length_analysis_summary{suffix}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
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
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: results/analysis/target_job_length/{overhead_type}/{workflow_type}/{failure_rate})')
    parser.add_argument('--overhead-type', type=str, choices=['overhead', 'nooverhead'],
                       default='overhead', help='Process overhead or nooverhead files')

    args = parser.parse_args()

    # Set default output directory
    if args.output_dir is None:
        args.output_dir = f"results/analysis/target_job_length/{args.overhead_type}/{args.workflow_type}/{args.failure_rate}"

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Target Job Length Optimization Analysis")
    print("="*70)
    print(f"Workflow Type: {args.workflow_type}")
    print(f"Failure Rate: {args.failure_rate}")
    print(f"Overhead Type: {args.overhead_type}")
    print(f"Output Directory: {args.output_dir}")
    print("="*70)

    # Collect data from all target job length directories
    data_by_composition = collect_data_from_directories(
        args.base_path,
        args.workflow_type,
        args.failure_rate,
        args.overhead_type
    )

    if not data_by_composition:
        print("Error: No data collected. Please check directory paths and file availability.")
        return

    print(f"\nCollected data for {len(data_by_composition)} constructions")

    # Generate visualizations
    plot_throughput_vs_target_length(data_by_composition, args.output_dir, args.overhead_type)
    plot_throughput_improvement(data_by_composition, args.output_dir, args.overhead_type)
    plot_network_activity_vs_target_length(data_by_composition, args.output_dir, args.overhead_type)
    plot_best_hybrid_comparison(data_by_composition, args.output_dir, args.overhead_type)

    # Generate failure cost analysis (only if failures exist)
    plot_failure_cost_analysis(data_by_composition, args.output_dir, args.overhead_type)
    plot_failure_count_analysis(data_by_composition, args.output_dir, args.overhead_type)

    # Generate summary table
    generate_summary_table(data_by_composition, args.output_dir, args.overhead_type)

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
