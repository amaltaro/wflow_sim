#!/usr/bin/env python3
"""
Failure Rate Impact Analysis Script

This script analyzes how workflow constructions (composition_number) perform across
various failure rates. It aggregates data from multiple directories to create
cross-dimensional comparisons.

The most **grouped** and most **ungrouped** compositions use ``total_groups`` from
result metrics: grouped is the **smallest** count (tied: lowest composition id);
ungrouped the **largest** (tied: highest id).

Analysis: Failure Rate Impact (Comparison #1)
- Fixed: workflow_type + target_job_length
- Variable: failure_rate (fr0, fr1, fr5, fr10, fr25)
- Compare: all available constructions across failure rates
- Primary Metric: event_throughput
- Second Metric: network_transfer_mb_per_event
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from composition_extremes import composition_extremes
from plot_legend_truncate import apply_truncated_construction_legend


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
        total_jobs = metrics.get('total_jobs', 0)
        total_job_retries = sim_result.get('total_job_retries', 0)
        total_logical_jobs = total_jobs - total_job_retries
        failure_rate_actual = sim_result.get('actual_job_failure_rate')

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
            'total_jobs': total_jobs,
            'total_job_retries': total_job_retries,
            'total_logical_jobs': total_logical_jobs,
            'failure_rate_actual': failure_rate_actual,
            'overhead_enabled': sim_result.get('overhead_enabled', True),
            'file_path': file_path
        }
    except Exception as e:
        print(f"  Warning: Failed to load {file_path}: {e}")
        return None


def collect_data_from_directories(
    base_path: str,
    workflow_type: str,
    target_job_length: str,
    data_rate: str = "100MBps",
) -> Dict[int, List[Dict[str, Any]]]:
    """Collect simulation data from multiple failure rate directories.

    Reads simulation result JSON files (*.json) in each failure-rate/data-rate
    directory.

    Args:
        base_path: Base path to results directory (e.g., 'results/sim/others')
        workflow_type: Workflow type (e.g., 'seq_real')
        target_job_length: Target job length (e.g., '12h')
        data_rate: Data transfer rate directory (e.g., '100MBps')

    Returns:
        Dictionary mapping composition_number to list of metrics across failure
        rates
    """
    base_dir = Path(base_path) / workflow_type / target_job_length

    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    failure_rates = ['fr0', 'fr1', 'fr5', 'fr10', 'fr25']

    # Dictionary: composition_number -> list of metrics (one per failure rate)
    data_by_composition: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    print(f"Collecting data from: {base_dir} (data rate: {data_rate})")

    for fr_dir in failure_rates:
        fr_path = base_dir / fr_dir / data_rate
        if not fr_path.exists():
            print(f"  Warning: Directory {fr_path} not found, skipping")
            continue

        json_files = list(fr_path.glob("*.json"))
        print(f"  Processing {fr_dir}: {len(json_files)} files found")

        for json_file in sorted(json_files):
            metrics = load_simulation_data(str(json_file))
            if metrics:
                comp_num = metrics['composition_number']
                data_by_composition[comp_num].append(metrics)

    return dict(sorted(data_by_composition.items()))


def plot_throughput_vs_failure_rate(
    data_by_composition: Dict[int, List[Dict[str, Any]]],
    output_dir: str,
    grouped_comp: int,
    independent_comp: int,
) -> None:
    """Plot event throughput vs. failure rate for all constructions.

    Args:
        data_by_composition: Dictionary mapping composition_number to metrics list
        output_dir: Output directory for plots
        grouped_comp: Most grouped composition
        independent_comp: Most ungrouped composition
    """
    print(f"==> Creating throughput vs. failure rate plot")

    fig, ax = plt.subplots(figsize=(14, 8))

    comps_sorted = sorted(data_by_composition.keys())
    must_legend_idx = {
        i for i, c in enumerate(comps_sorted) if c in (grouped_comp, independent_comp)
    }

    # Extract failure rates and throughput for each construction
    failure_rates = []
    for comp_num in comps_sorted:
        comp_data = data_by_composition[comp_num]
        # Sort by failure rate
        comp_data_sorted = sorted(comp_data, key=lambda x: x['failure_rate'])
        fr_values = [d['failure_rate'] for d in comp_data_sorted]
        throughput_values = [d['event_throughput'] for d in comp_data_sorted]

        if not failure_rates:
            failure_rates = fr_values

    # Find best hybrid for each failure rate
    best_hybrids = {}
    for fr in failure_rates:
        best_hybrids[fr] = identify_best_hybrid(
            data_by_composition, fr, grouped_comp, independent_comp
        )

    # Plot lines for all constructions
    for comp_num in comps_sorted:
        comp_data = data_by_composition[comp_num]
        comp_data_sorted = sorted(comp_data, key=lambda x: x['failure_rate'])
        fr_values = [d['failure_rate'] for d in comp_data_sorted]
        throughput_values = [d['event_throughput'] for d in comp_data_sorted]

        # Plot line for this construction
        label = f"Const {comp_num}"
        if comp_num == grouped_comp and comp_num == independent_comp:
            ax.plot(fr_values, throughput_values, 'D-', label=label, linewidth=2.5,
                   color='#9467bd', markersize=7, zorder=10)
        elif comp_num == grouped_comp:
            ax.plot(fr_values, throughput_values, 'o-', label=label, linewidth=2.5,
                   color='#d62728', markersize=8, zorder=10)
        elif comp_num == independent_comp:
            ax.plot(fr_values, throughput_values, 's-', label=label, linewidth=2.5,
                   color='#2ca02c', markersize=8, zorder=10)
        else:
            # Check if this is the best hybrid for any failure rate
            is_best_hybrid = any(
                best_hybrids[fr] == comp_num
                for fr in failure_rates
                if best_hybrids[fr] is not None
            )
            if is_best_hybrid:
                # Highlight best hybrid with triangle markers
                # Mark only the points where this construction is the best hybrid
                marker_fr = [fr for fr in fr_values if best_hybrids.get(fr) == comp_num]
                marker_throughput = [th for fr, th in zip(fr_values, throughput_values)
                                    if best_hybrids.get(fr) == comp_num]

                # Plot the full line
                ax.plot(fr_values, throughput_values, '-', label=label, linewidth=1.5,
                       alpha=0.7, markersize=5, zorder=5)
                # Add triangle markers at best hybrid points
                if marker_fr:
                    ax.plot(marker_fr, marker_throughput, '^', label=None, linewidth=0,
                           color='#1f77b4', markersize=10, zorder=11, alpha=0.9,
                           markerfacecolor='#1f77b4', markeredgecolor='white', markeredgewidth=1.5)
            else:
                # Regular hybrid constructions
                ax.plot(fr_values, throughput_values, '-', label=label, linewidth=1.5,
                       alpha=0.7, markersize=5)

    ax.set_xlabel("Failure Rate (%)", fontsize=12)
    ax.set_ylabel("Event Throughput (events/second)", fontsize=12)
    ax.set_title("Event Throughput vs. Failure Rate", fontsize=14)
    apply_truncated_construction_legend(
        ax, len(comps_sorted), must_legend_idx, bbox=(1.05, 1), fontsize=9
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=-1, right=26)  # Slight padding around failure rates

    plt.tight_layout()
    output_path = os.path.join(output_dir, "throughput_vs_failure_rate.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_path}")


def plot_throughput_degradation(
    data_by_composition: Dict[int, List[Dict[str, Any]]],
    output_dir: str,
    grouped_comp: int,
    independent_comp: int,
) -> None:
    """Plot throughput degradation (relative to fr0) vs. failure rate.

    Args:
        data_by_composition: Dictionary mapping composition_number to metrics list
        output_dir: Output directory for plots
        grouped_comp: Most grouped composition
        independent_comp: Most ungrouped composition
    """
    print(f"==> Creating throughput degradation plot")

    fig, ax = plt.subplots(figsize=(14, 8))

    degradation_by_comp: Dict[int, tuple] = {}
    for comp_num in sorted(data_by_composition.keys()):
        comp_data = data_by_composition[comp_num]
        comp_data_sorted = sorted(comp_data, key=lambda x: x['failure_rate'])
        baseline = next((d for d in comp_data_sorted if d['failure_rate'] == 0.0), None)
        if not baseline:
            continue
        baseline_throughput = baseline['event_throughput']
        if baseline_throughput == 0:
            continue
        fr_values: List[float] = []
        degradation_values: List[float] = []
        for d in comp_data_sorted:
            fr_values.append(d['failure_rate'])
            deg = ((baseline_throughput - d['event_throughput']) / baseline_throughput) * 100
            degradation_values.append(deg)
        degradation_by_comp[comp_num] = (fr_values, degradation_values)

    plotted = list(degradation_by_comp.keys())
    must_legend_idx = {
        j for j, c in enumerate(plotted) if c in (grouped_comp, independent_comp)
    }

    for comp_num, (fr_values, degradation_values) in degradation_by_comp.items():
        label = f"Const {comp_num}"
        if comp_num == grouped_comp and comp_num == independent_comp:
            ax.plot(fr_values, degradation_values, 'D-', label=label, linewidth=2.5,
                   color='#9467bd', markersize=7, zorder=10)
        elif comp_num == grouped_comp:
            ax.plot(fr_values, degradation_values, 'o-', label=label, linewidth=2.5,
                   color='#d62728', markersize=8, zorder=10)
        elif comp_num == independent_comp:
            ax.plot(fr_values, degradation_values, 's-', label=label, linewidth=2.5,
                   color='#2ca02c', markersize=8, zorder=10)
        else:
            ax.plot(fr_values, degradation_values, '-', label=label, linewidth=1.5,
                   alpha=0.7, markersize=5)

    ax.set_xlabel("Failure Rate (%)", fontsize=12)
    ax.set_ylabel("Throughput Degradation (%)", fontsize=12)
    ax.set_title("Throughput Degradation vs. Failure Rate\n(Relative to fr0)", fontsize=14)
    apply_truncated_construction_legend(
        ax, len(plotted), must_legend_idx, bbox=(1.05, 1), fontsize=9
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=-1, right=26)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "throughput_degradation.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_path}")


def identify_best_hybrid(
    data_by_composition: Dict[int, List[Dict[str, Any]]],
    failure_rate: float,
    grouped_comp: int,
    independent_comp: int,
    verbose: bool = False,
) -> Optional[int]:
    """Identify the best hybrid construction for a given failure rate.

    Hybrids are all compositions strictly between ``grouped_comp`` and
    ``independent_comp`` (exclusive of both extremes).

    Uses event_throughput as the primary metric, with network_transfer_mb_per_event
    as a tiebreaker (lower network transfer is preferred).

    Args:
        data_by_composition: Dictionary mapping composition_number to metrics list
        failure_rate: Target failure rate
        grouped_comp: Most grouped composition (excluded from hybrid search)
        independent_comp: Most ungrouped composition (excluded from hybrid search)
        verbose: If True, print information about ties to stdout

    Returns:
        Composition number of best hybrid, or None if not found
    """
    if independent_comp <= grouped_comp + 1:
        return None

    # Collect all hybrid constructions with their metrics
    hybrid_candidates = []

    for comp_num in range(grouped_comp + 1, independent_comp):
        if comp_num not in data_by_composition:
            continue

        comp_data = data_by_composition[comp_num]
        # Find data for this failure rate
        fr_data = next((d for d in comp_data if abs(d['failure_rate'] - failure_rate) < 0.1), None)
        if fr_data:
            hybrid_candidates.append({
                'comp_num': comp_num,
                'throughput': fr_data['event_throughput'],
                'network_transfer': fr_data['network_transfer_mb_per_event']
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
            print(f"    Tie detected at fr{int(failure_rate)}%: {', '.join(tied_names)} "
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


def plot_network_activity_vs_failure_rate(
    data_by_composition: Dict[int, List[Dict[str, Any]]],
    output_dir: str,
    grouped_comp: int,
    independent_comp: int,
) -> None:
    """Plot network transfer vs. failure rate for all constructions.

    This visualization shows how network activity (remote I/O) changes with
    failure rates, which is important for understanding workflow efficiency
    under different failure scenarios.

    Args:
        data_by_composition: Dictionary mapping composition_number to metrics list
        output_dir: Output directory for plots
        grouped_comp: Most grouped composition
        independent_comp: Most ungrouped composition
    """
    print(f"==> Creating network activity vs. failure rate plot")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Network Transfer per Event vs. Failure Rate
    comps_sorted = sorted(data_by_composition.keys())
    must_legend_idx = {
        i for i, c in enumerate(comps_sorted) if c in (grouped_comp, independent_comp)
    }
    failure_rates = []
    for comp_num in comps_sorted:
        comp_data = data_by_composition[comp_num]
        comp_data_sorted = sorted(comp_data, key=lambda x: x['failure_rate'])
        fr_values = [d['failure_rate'] for d in comp_data_sorted]
        network_values = [d['network_transfer_mb_per_event'] for d in comp_data_sorted]

        if not failure_rates:
            failure_rates = fr_values

        # Plot line for this construction
        label = f"Const {comp_num}"
        if comp_num == grouped_comp and comp_num == independent_comp:
            ax1.plot(fr_values, network_values, 'D-', label=label, linewidth=2.5,
                    color='#9467bd', markersize=7, zorder=10)
        elif comp_num == grouped_comp:
            ax1.plot(fr_values, network_values, 'o-', label=label, linewidth=2.5,
                    color='#d62728', markersize=8, zorder=10)
        elif comp_num == independent_comp:
            ax1.plot(fr_values, network_values, 's-', label=label, linewidth=2.5,
                    color='#2ca02c', markersize=8, zorder=10)
        else:
            ax1.plot(fr_values, network_values, '-', label=label, linewidth=1.5,
                    alpha=0.7, markersize=5)

    ax1.set_xlabel("Failure Rate (%)", fontsize=12)
    ax1.set_ylabel("Network Transfer per Event (MB)", fontsize=12)
    ax1.set_title("Network Transfer vs. Failure Rate", fontsize=13)
    apply_truncated_construction_legend(
        ax1, len(comps_sorted), must_legend_idx, bbox=(1.05, 1), fontsize=9
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=-1, right=26)

    # Plot 2: Remote Read vs. Remote Write breakdown (focus on extremes and best hybrid)
    # Find best hybrid for each failure rate
    best_hybrids = {}
    for fr in failure_rates:
        best_hybrids[fr] = identify_best_hybrid(
            data_by_composition, fr, grouped_comp, independent_comp
        )

    constructions_to_plot = {grouped_comp, independent_comp}
    for fr in failure_rates:
        if best_hybrids[fr] is not None:
            constructions_to_plot.add(best_hybrids[fr])

    for comp_num in sorted(constructions_to_plot):
        if comp_num not in data_by_composition:
            continue

        comp_data = data_by_composition[comp_num]
        comp_data_sorted = sorted(comp_data, key=lambda x: x['failure_rate'])
        fr_values = [d['failure_rate'] for d in comp_data_sorted]
        read_remote = [d['total_read_remote_mb_per_event'] for d in comp_data_sorted]
        write_remote = [d['total_write_remote_mb_per_event'] for d in comp_data_sorted]

        # Plot lines
        # Use different marker shapes for Read vs Write to make legend clearer
        # Read: circles (o), Write: squares (s)
        label = f"Const {comp_num}"
        if comp_num == grouped_comp and comp_num == independent_comp:
            ax2.plot(fr_values, read_remote, 'o--', label=f"{label} (Read)", linewidth=2.5,
                    color='#9467bd', markersize=6, zorder=10, alpha=0.85, markerfacecolor='#9467bd',
                    markeredgecolor='#9467bd', markeredgewidth=1.5)
            ax2.plot(fr_values, write_remote, 's-', label=f"{label} (Write)", linewidth=2.5,
                    color='#9467bd', markersize=6, zorder=10, alpha=0.9, markerfacecolor='#9467bd',
                    markeredgecolor='#9467bd', markeredgewidth=1.5)
        elif comp_num == grouped_comp:
            ax2.plot(fr_values, read_remote, 'o--', label=f"{label} (Read)", linewidth=2.5,
                    color='#d62728', markersize=7, zorder=10, alpha=0.7, markerfacecolor='#d62728',
                    markeredgecolor='#d62728', markeredgewidth=1.5)
            ax2.plot(fr_values, write_remote, 's-', label=f"{label} (Write)", linewidth=2.5,
                    color='#d62728', markersize=7, zorder=10, alpha=0.9, markerfacecolor='#d62728',
                    markeredgecolor='#d62728', markeredgewidth=1.5)
        elif comp_num == independent_comp:
            ax2.plot(fr_values, read_remote, 'o--', label=f"{label} (Read)", linewidth=2.5,
                    color='#2ca02c', markersize=7, zorder=10, alpha=0.9, markerfacecolor='#2ca02c',
                    markeredgecolor='#2ca02c', markeredgewidth=1.5)
            ax2.plot(fr_values, write_remote, 's-', label=f"{label} (Write)", linewidth=2.5,
                    color='#2ca02c', markersize=7, zorder=10, alpha=0.9, markerfacecolor='#2ca02c',
                    markeredgecolor='#2ca02c', markeredgewidth=1.5)
        else:
            # Best hybrid
            ax2.plot(fr_values, read_remote, 'o--', label=f"{label} (Read)", linewidth=2,
                    color='#1f77b4', markersize=6, zorder=9, alpha=0.8, markerfacecolor='#1f77b4',
                    markeredgecolor='#1f77b4', markeredgewidth=1.5)
            ax2.plot(fr_values, write_remote, 's-', label=f"{label} (Write)", linewidth=2,
                    color='#1f77b4', markersize=6, zorder=9, alpha=0.8, markerfacecolor='#1f77b4',
                    markeredgecolor='#1f77b4', markeredgewidth=1.5)

    ax2.set_xlabel("Failure Rate (%)", fontsize=12)
    ax2.set_ylabel("Data Volume per Event (MB)", fontsize=12)
    ax2.set_title("Remote I/O Breakdown vs. Failure Rate", fontsize=13)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=-1, right=26)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "network_activity_vs_failure_rate.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_path}")


def plot_best_hybrid_comparison(
    data_by_composition: Dict[int, List[Dict[str, Any]]],
    output_dir: str,
    grouped_comp: int,
    independent_comp: int,
) -> None:
    """Plot comparison of the two extremes and the best hybrid for each failure rate.

    Args:
        data_by_composition: Dictionary mapping composition_number to metrics list
        output_dir: Output directory for plots
        grouped_comp: Most grouped composition
        independent_comp: Most ungrouped composition
    """
    print(f"==> Creating best hybrid comparison plot")

    # Get all failure rates from the data
    all_failure_rates = set()
    for comp_data in data_by_composition.values():
        for d in comp_data:
            all_failure_rates.add(d['failure_rate'])
    failure_rates = sorted(all_failure_rates)

    # Find best hybrid for each failure rate
    best_hybrids = {}
    for fr in failure_rates:
        best_hybrids[fr] = identify_best_hybrid(
            data_by_composition, fr, grouped_comp, independent_comp
        )

    def _throughput_at_fr(comp: int, fr: float) -> float:
        if comp not in data_by_composition:
            return 0.0
        rec = next(
            (d for d in data_by_composition[comp] if abs(d['failure_rate'] - fr) < 0.1), None
        )
        return rec['event_throughput'] if rec else 0.0

    # Extract throughput values
    const_grouped = [_throughput_at_fr(grouped_comp, fr) for fr in failure_rates]
    const_indep = [_throughput_at_fr(independent_comp, fr) for fr in failure_rates]
    best_hybrid_throughput = []
    for fr in failure_rates:
        best_comp = best_hybrids[fr]
        if best_comp and best_comp in data_by_composition:
            best_hybrid_throughput.append(_throughput_at_fr(best_comp, fr))
        else:
            best_hybrid_throughput.append(0.0)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(failure_rates))
    width = 0.25

    # Determine best hybrid label for legend (show unique best hybrids)
    unique_best_hybrids = sorted(
        {best_hybrids[fr] for fr in failure_rates if best_hybrids[fr] is not None}
    )
    if len(unique_best_hybrids) == 1:
        best_hybrid_legend = f"Best hybrid (Const {unique_best_hybrids[0]})"
    elif len(unique_best_hybrids) <= 3:
        best_hybrid_legend = f"Best hybrid (Const {', '.join(map(str, unique_best_hybrids))})"
    else:
        best_hybrid_legend = (
            f"Best hybrid (Const {unique_best_hybrids[0]}-{unique_best_hybrids[-1]})"
        )

    if grouped_comp == independent_comp:
        w = 0.35
        g_label = f"Const {grouped_comp} (grouped and ungrouped)"
        bars1 = ax.bar(
            x - w / 2, const_grouped, w, label=g_label, color="#9467bd", alpha=0.8
        )
        bars2 = ax.bar(
            x + w / 2,
            best_hybrid_throughput,
            w,
            label=best_hybrid_legend,
            color="#1f77b4",
            alpha=0.8,
        )
        all_bar_groups: List[Any] = [bars1, bars2]
    else:
        label_g = f"Const {grouped_comp} (most grouped)"
        label_i = f"Const {independent_comp} (most ungrouped)"
        bars1 = ax.bar(
            x - width, const_grouped, width, label=label_g, color='#d62728', alpha=0.8
        )
        bars2 = ax.bar(x, const_indep, width, label=label_i, color='#2ca02c', alpha=0.8)
        bars3 = ax.bar(
            x + width, best_hybrid_throughput, width, label=best_hybrid_legend, color='#1f77b4',
            alpha=0.8,
        )
        all_bar_groups = [bars1, bars2, bars3]

    # Add value labels on bars
    for bars in all_bar_groups:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, height, f"{height:.4f}",
                    ha="center", va="bottom", fontsize=8
                )

    ax.set_xlabel("Failure Rate (%)", fontsize=12)
    ax.set_ylabel("Event Throughput (events/second)", fontsize=12)
    ax.set_title("Best Hybrid vs. Extremes Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(fr)}%" for fr in failure_rates])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "best_hybrid_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_path}")


def generate_summary_table(data_by_composition: Dict[int, List[Dict[str, Any]]],
                          output_dir: str) -> pd.DataFrame:
    """Generate summary table with metrics across failure rates.

    Args:
        data_by_composition: Dictionary mapping composition_number to metrics list
        output_dir: Output directory for table
    Returns:
        DataFrame with summary metrics
    """
    print(f"==> Generating summary table")

    table_data = []

    for comp_num in sorted(data_by_composition.keys()):
        comp_data = sorted(data_by_composition[comp_num], key=lambda x: x['failure_rate'])

        for metrics in comp_data:
            row = {
                'Composition': comp_num,
                'Failure_Rate_%': metrics['failure_rate'],
                'Total_Jobs': metrics.get('total_jobs', 0),
                'Total_Job_Retries': metrics.get('total_job_retries', 0),
                'Total_Logical_Jobs': metrics.get('total_logical_jobs', 0),
                'Failure_Rate_Actual_%': metrics.get('failure_rate_actual', 0.0),
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
    csv_path = os.path.join(output_dir, "failure_rate_analysis_summary.csv")
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  => Saved: {csv_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Analyze workflow construction performance across failure rates'
    )
    parser.add_argument('base_path', type=str,
                       help='Base path to results directory (e.g., results/sim/others)')
    parser.add_argument('workflow_type', type=str,
                       help='Workflow type (e.g., seq_real)')
    parser.add_argument('target_job_length', type=str,
                       help='Target job length (e.g., 12h)')
    parser.add_argument('--data-rate', type=str, default='100MBps',
                       help='Data transfer rate directory (default: 100MBps)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: results/analysis/failure_rate/{workflow_type}/{target_job_length})')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"results/analysis/failure_rate/{args.workflow_type}/{args.target_job_length}"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Failure Rate Impact Analysis")
    print("="*70)
    print(f"Workflow Type: {args.workflow_type}")
    print(f"Target Job Length: {args.target_job_length}")
    print(f"Data Rate: {args.data_rate}")
    print(f"Output Directory: {args.output_dir}")
    print("="*70)

    data_by_composition = collect_data_from_directories(
        args.base_path,
        args.workflow_type,
        args.target_job_length,
        data_rate=args.data_rate,
    )

    if not data_by_composition:
        print("Error: No data collected. Please check directory paths and file availability.")
        return

    grouped_comp, independent_comp = composition_extremes(data_by_composition)
    print(
        f"\nCollected data for {len(data_by_composition)} constructions "
        f"(most grouped: Const {grouped_comp}, most ungrouped: "
        f"Const {independent_comp})\n"
    )

    plot_throughput_vs_failure_rate(
        data_by_composition, args.output_dir, grouped_comp, independent_comp
    )
    plot_throughput_degradation(
        data_by_composition, args.output_dir, grouped_comp, independent_comp
    )
    plot_network_activity_vs_failure_rate(
        data_by_composition, args.output_dir, grouped_comp, independent_comp
    )
    plot_best_hybrid_comparison(
        data_by_composition, args.output_dir, grouped_comp, independent_comp
    )
    generate_summary_table(data_by_composition, args.output_dir)

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
