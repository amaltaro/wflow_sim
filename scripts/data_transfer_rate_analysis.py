#!/usr/bin/env python3
"""
Data Transfer Rate Sensitivity Analysis Script

This script analyzes how different network data transfer rates (10 MB/s, 100 MB/s,
1 GB/s, 10 GB/s) affect workflow construction performance. Simulations must be run
separately for each rate with output organized by rate directory.

Analysis: Data Transfer Rate Sensitivity
- Fixed: 12h target job length, chosen failure rate, all 3 workflow types
- Variable: network data transfer rate (10, 100, 1000, 10000 MB/s)
- Compare: Const 1, Const 16, and best hybrid across data rates
- Primary Metric: event_throughput
- Job overhead: mean and std of job_overhead_secs from simulation_result.jobs
  (sample of up to 10 jobs per group); lower data rate increases overhead.
"""

import argparse
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Rate directory names (MBps/GBps = megabytes/gigabytes per second) and MB/s values
RATE_DIRS = ['10MBps', '100MBps', '1GBps', '10GBps']
RATE_MBPS = [10, 100, 1000, 10000]
TARGET_JOB_LENGTH = '12h'
FAILURE_RATE = 'fr0'
WORKFLOW_TYPES = ['case1_real', 'case2_homo', 'case3_hetero']


def _job_overhead_stats(jobs: List[Dict[str, Any]], key: str = 'job_overhead_secs') -> Tuple[float, float, int]:
    """Compute mean, std and count for a job overhead field from simulation_result.jobs."""
    overheads = [
        j.get(key, 0.0)
        for j in jobs
        if isinstance(j.get(key), (int, float))
    ]
    n = len(overheads)
    if n == 0:
        return 0.0, 0.0, 0
    mean = float(np.mean(overheads))
    std = float(np.std(overheads)) if n > 1 else 0.0
    return mean, std, n


def load_simulation_data(file_path: str) -> Optional[Dict[str, Any]]:
    """Load and extract key metrics from a simulation JSON file.

    Includes job-level overhead stats from simulation_result.jobs (sample of up to 10
    jobs per group): mean and std of job_overhead_secs and job_overhead_cpu_time.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        metrics = data.get('metrics', {})
        sim_result = data.get('simulation_result', {})
        jobs = sim_result.get('jobs', [])
        secs_mean, secs_std, overhead_n = _job_overhead_stats(jobs, 'job_overhead_secs')
        cpu_mean, cpu_std, _ = _job_overhead_stats(jobs, 'job_overhead_cpu_time')

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
            'job_overhead_secs_mean': secs_mean,
            'job_overhead_secs_std': secs_std,
            'job_overhead_cpu_time_mean': cpu_mean,
            'job_overhead_cpu_time_std': cpu_std,
            'job_overhead_sample_count': overhead_n,
            'file_path': file_path
        }
    except Exception as e:
        print(f"  Warning: Failed to load {file_path}: {e}")
        return None


def rate_dir_to_mbps(rate_dir: str) -> int:
    """Convert rate directory name to MB/s for ordering and display."""
    return dict(zip(RATE_DIRS, RATE_MBPS)).get(rate_dir, 0)


def collect_data_by_rate(base_path: str,
                         rate_dirs: List[str],
                         workflow_types: List[str],
                         failure_rate: str = FAILURE_RATE) -> Dict[str, Dict[str, Dict[int, Dict[str, Any]]]]:
    """Collect simulation data from each rate directory.

    Expects unified structure:
    {base_path}/{workflow_type}/12h/{failure_rate}/{rate_dir}/*.json
    (base_path is e.g. results/sim/others)

    Returns:
        Dictionary mapping rate_dir -> workflow_type -> composition_number -> metrics
    """
    data_by_rate: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}
    base = Path(base_path)

    for rate_dir in rate_dirs:
        data_by_workflow: Dict[str, Dict[int, Dict[str, Any]]] = {}
        for workflow_type in workflow_types:
            workflow_dir = base / workflow_type / TARGET_JOB_LENGTH / failure_rate / rate_dir
            if not workflow_dir.exists():
                print(f"  Warning: {workflow_type}/12h/{failure_rate}/{rate_dir} not found, skipping")
                continue

            json_files = list(workflow_dir.glob("*.json"))
            if not json_files:
                print(f"  Warning: No JSON files in {workflow_dir}")
                continue

            data_by_composition: Dict[int, Dict[str, Any]] = {}
            for json_file in sorted(json_files):
                metrics = load_simulation_data(str(json_file))
                if metrics:
                    comp_num = metrics['composition_number']
                    data_by_composition[comp_num] = metrics

            if data_by_composition:
                data_by_workflow[workflow_type] = data_by_composition

        if data_by_workflow:
            data_by_rate[rate_dir] = data_by_workflow
            print(f"  {rate_dir}: {len(data_by_workflow)} workflow types, "
                  f"{sum(len(d) for d in data_by_workflow.values())} compositions")

    return data_by_rate


def identify_best_hybrid(data_by_composition: Dict[int, Dict[str, Any]]) -> Optional[int]:
    """Identify best hybrid (2-15) by throughput, then lower network transfer."""
    hybrid_candidates = []
    for comp_num in range(2, 16):
        if comp_num not in data_by_composition:
            continue
        m = data_by_composition[comp_num]
        hybrid_candidates.append({
            'comp_num': comp_num,
            'throughput': m['event_throughput'],
            'network_transfer': m['network_transfer_mb_per_event']
        })
    if not hybrid_candidates:
        return None
    best = max(hybrid_candidates,
               key=lambda x: (x['throughput'], -x['network_transfer']))
    return best['comp_num']


def plot_throughput_vs_data_rate(data_by_rate: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]],
                                output_dir: str,
                                failure_rate: str = FAILURE_RATE) -> None:
    """Plot event throughput vs. data transfer rate for Const 1, Const 16, best hybrid."""
    print("\n==> Creating throughput vs. data transfer rate plot")

    rate_dirs_sorted = sorted(
        [r for r in RATE_DIRS if r in data_by_rate],
        key=rate_dir_to_mbps
    )
    if not rate_dirs_sorted:
        print("  No rate data to plot")
        return

    x_mbps = [rate_dir_to_mbps(r) for r in rate_dirs_sorted]
    x_labels = [f"{rate_dir_to_mbps(r)} MB/s" for r in rate_dirs_sorted]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    workflow_types = sorted(WORKFLOW_TYPES)

    for ax, workflow_type in zip(axes, workflow_types):
        const1_throughput = []
        const16_throughput = []
        best_hybrid_throughput = []
        best_hybrid_nums = []

        for rate_dir in rate_dirs_sorted:
            wf_data = data_by_rate.get(rate_dir, {}).get(workflow_type, {})
            const1_throughput.append(wf_data.get(1, {}).get('event_throughput', 0.0))
            const16_throughput.append(wf_data.get(16, {}).get('event_throughput', 0.0))
            best_hybrid = identify_best_hybrid(wf_data) if wf_data else None
            if best_hybrid and best_hybrid in wf_data:
                best_hybrid_throughput.append(wf_data[best_hybrid]['event_throughput'])
                best_hybrid_nums.append(best_hybrid)
            else:
                best_hybrid_throughput.append(0.0)
                best_hybrid_nums.append(None)

        ax.plot(x_mbps, const1_throughput, 'o-', label='Const 1', color='#d62728', linewidth=2)
        ax.plot(x_mbps, const16_throughput, 's-', label='Const 16', color='#2ca02c', linewidth=2)
        ax.plot(x_mbps, best_hybrid_throughput, '^-', label='Best Hybrid', color='#1f77b4', linewidth=2)
        for i, (xi, yi) in enumerate(zip(x_mbps, best_hybrid_throughput)):
            if best_hybrid_nums[i] and yi > 0:
                ax.annotate(f"C{best_hybrid_nums[i]}", (xi, yi), textcoords="offset points",
                            xytext=(0, 8), ha='center', fontsize=8)
        ax.set_xscale('log')
        ax.set_xticks(x_mbps)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Data Transfer Rate (MB/s)")
        ax.set_ylabel("Event Throughput (events/s)" if ax == axes[0] else "")
        ax.set_title(workflow_type)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Event Throughput vs. Network Data Transfer Rate (12h, {failure_rate})",
                 fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "throughput_vs_data_transfer_rate.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {out_path}")


def _plot_one_job_overhead_bar_chart(
    data_by_rate: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]],
    output_dir: str,
    failure_rate: str,
    mean_key: str,
    std_key: str,
    y_label: str,
    title_metric: str,
    filename: str,
) -> None:
    """Draw one grouped bar chart of job overhead (mean ± std) vs. data transfer rate."""
    rate_dirs_sorted = sorted(
        [r for r in RATE_DIRS if r in data_by_rate],
        key=rate_dir_to_mbps
    )
    if not rate_dirs_sorted:
        return

    n_rates = len(rate_dirs_sorted)
    x_labels = [f"{rate_dir_to_mbps(r)} MB/s" for r in rate_dirs_sorted]
    bar_width = 0.25
    group_width = bar_width * 3 + 0.15
    x_base = np.arange(n_rates) * group_width

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)

    for ax, workflow_type in zip(axes, sorted(WORKFLOW_TYPES)):
        const1_mean, const1_std = [], []
        const16_mean, const16_std = [], []
        best_mean, best_std = [], []

        for rate_dir in rate_dirs_sorted:
            wf_data = data_by_rate.get(rate_dir, {}).get(workflow_type, {})
            c1 = wf_data.get(1, {})
            c16 = wf_data.get(16, {})
            const1_mean.append(c1.get(mean_key, 0.0))
            const1_std.append(c1.get(std_key, 0.0))
            const16_mean.append(c16.get(mean_key, 0.0))
            const16_std.append(c16.get(std_key, 0.0))
            best_hybrid = identify_best_hybrid(wf_data) if wf_data else None
            if best_hybrid and best_hybrid in wf_data:
                b = wf_data[best_hybrid]
                best_mean.append(b.get(mean_key, 0.0))
                best_std.append(b.get(std_key, 0.0))
            else:
                best_mean.append(0.0)
                best_std.append(0.0)

        x1 = x_base - bar_width
        x2 = x_base
        x3 = x_base + bar_width
        ax.bar(x1, const1_mean, bar_width, yerr=const1_std, label='Const 1',
               color='#d62728', capsize=3, error_kw={'linewidth': 1.5})
        ax.bar(x2, const16_mean, bar_width, yerr=const16_std, label='Const 16',
               color='#2ca02c', capsize=3, error_kw={'linewidth': 1.5})
        ax.bar(x3, best_mean, bar_width, yerr=best_std, label='Best Hybrid',
               color='#1f77b4', capsize=3, error_kw={'linewidth': 1.5})
        ax.set_xticks(x_base)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Data Transfer Rate (MB/s)")
        ax.set_ylabel(y_label if ax == axes[0] else "")
        ax.set_title(workflow_type)
        ax.set_yscale('log')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        f"Mean Job Overhead ({title_metric}) vs. Data Transfer Rate (12h, {failure_rate}); ",
        fontsize=12
    )
    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {out_path}")


def plot_job_overhead_vs_data_rate(data_by_rate: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]],
                                   output_dir: str,
                                   failure_rate: str = FAILURE_RATE) -> None:
    """Plot mean job overhead vs. data transfer rate: wallclock (secs) and CPU time.

    Creates two grouped bar charts: job_overhead_secs and job_overhead_cpu_time.
    One group per data rate; three bars per group (Const 1, Const 16, Best Hybrid).
    """
    print("\n==> Creating job overhead vs. data transfer rate plots")
    _plot_one_job_overhead_bar_chart(
        data_by_rate, output_dir, failure_rate,
        mean_key='job_overhead_secs_mean',
        std_key='job_overhead_secs_std',
        y_label="Mean job overhead (seconds, log scale)",
        title_metric="wallclock",
        filename="job_overhead_secs_vs_data_transfer_rate.png",
    )
    _plot_one_job_overhead_bar_chart(
        data_by_rate, output_dir, failure_rate,
        mean_key='job_overhead_cpu_time_mean',
        std_key='job_overhead_cpu_time_std',
        y_label="Mean job overhead (CPU-seconds, log scale)",
        title_metric="CPU time",
        filename="job_overhead_cpu_time_vs_data_transfer_rate.png",
    )


def generate_summary_table(data_by_rate: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]],
                           output_dir: str) -> pd.DataFrame:
    """Generate CSV summary: rate, workflow_type, composition, throughput, network_per_event, etc."""
    print("\n==> Generating summary table")

    rows = []
    for rate_dir in sorted(data_by_rate.keys(), key=rate_dir_to_mbps):
        rate_mbps = rate_dir_to_mbps(rate_dir)
        for workflow_type in sorted(data_by_rate[rate_dir].keys()):
            for comp_num in sorted(data_by_rate[rate_dir][workflow_type].keys()):
                m = data_by_rate[rate_dir][workflow_type][comp_num]
                rows.append({
                    'data_transfer_rate_mbps': rate_mbps,
                    'rate_dir': rate_dir,
                    'workflow_type': workflow_type,
                    'composition_number': comp_num,
                    'event_throughput': m['event_throughput'],
                    'wall_time_per_event': m['wall_time_per_event'],
                    'cpu_time_per_event': m['cpu_time_per_event'],
                    'network_transfer_mb_per_event': m['network_transfer_mb_per_event'],
                    'cpu_utilization': m['cpu_utilization'],
                    'memory_occupancy': m['memory_occupancy'],
                    'total_groups': m['total_groups'],
                    'job_overhead_secs_mean': m.get('job_overhead_secs_mean', 0.0),
                    'job_overhead_secs_std': m.get('job_overhead_secs_std', 0.0),
                    'job_overhead_cpu_time_mean': m.get('job_overhead_cpu_time_mean', 0.0),
                    'job_overhead_cpu_time_std': m.get('job_overhead_cpu_time_std', 0.0),
                    'job_overhead_sample_count': m.get('job_overhead_sample_count', 0),
                })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "data_transfer_rate_analysis_summary.csv")
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  => Saved: {csv_path}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Analyze workflow performance across network data transfer rates'
    )
    parser.add_argument('base_path', type=str,
                        help='Base path to simulation results containing workflow types '
                             '(e.g. results/sim/others)')
    parser.add_argument('--rate-dirs', type=str, nargs='+', default=RATE_DIRS,
                        help=f'Rate directory names (default: {" ".join(RATE_DIRS)})')
    parser.add_argument('--workflow-types', type=str, nargs='+', default=WORKFLOW_TYPES,
                        help='Workflow types to analyze')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: results/analysis/data_transfer_rate)')
    parser.add_argument('--failure-rate', type=str, default=FAILURE_RATE,
                        help=f'Failure rate directory, e.g. fr0, fr5 (default: {FAILURE_RATE})')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = "results/analysis/data_transfer_rate"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Data Transfer Rate Sensitivity Analysis")
    print("=" * 70)
    print(f"Base path: {args.base_path}")
    print(f"Rate directories: {', '.join(args.rate_dirs)}")
    print(f"Workflow types: {', '.join(args.workflow_types)}")
    print(f"Target job length: {TARGET_JOB_LENGTH}, Failure rate: {args.failure_rate}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)

    data_by_rate = collect_data_by_rate(
        args.base_path,
        args.rate_dirs,
        args.workflow_types,
        args.failure_rate
    )

    if not data_by_rate:
        print("Error: No data collected. Run simulate-all or simulate-data-transfer-rate "
              f"so that base_path/<workflow_type>/12h/{args.failure_rate}/<rate_dir>/*.json exist.")
        return 1

    plot_throughput_vs_data_rate(data_by_rate, args.output_dir, args.failure_rate)
    plot_job_overhead_vs_data_rate(data_by_rate, args.output_dir, args.failure_rate)
    generate_summary_table(data_by_rate, args.output_dir)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
