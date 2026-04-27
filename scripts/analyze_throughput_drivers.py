#!/usr/bin/env python3
"""
Analyze which CPU-, memory-, and network-related metrics best relate to event_throughput (correlation only).

Compares ten metrics as potential drivers of event_throughput (including the one used in the score formula), grouped by resource:
- CPU: cpu_utilization, total_cpu_cores_used, cpu_cores_per_event
- Memory: memory_occupancy, total_memory_used_mb, memory_mb_per_event
- I/O: total_network_transfer_mb, network_transfer_mb_per_event, total_write_remote_mb_per_event, total_read_remote_mb_per_event

Uses one scenario directory (same as construction_metrics_analysis): loads all 16
construction JSONs and prints Pearson correlation of each of the ten with
event_throughput (output order follows the groups above). Higher |r| means stronger linear relationship; sign indicates
direction.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# All keys to load; predictors ordered by group (CPU, Memory, I/O) for output
METRIC_KEYS = [
    'event_throughput',
    'cpu_utilization',
    'total_cpu_cores_used',
    'cpu_cores_per_event',
    'memory_occupancy',
    'total_memory_used_mb',
    'memory_mb_per_event',
    'total_network_transfer_mb',
    'network_transfer_mb_per_event',
    'total_write_remote_mb_per_event',
    'total_read_remote_mb_per_event',
]
# Predictors in display order: CPU, then Memory, then I/O (total then per-event where applicable)
PREDICTORS_ORDERED = [
    'cpu_utilization', 'total_cpu_cores_used', 'cpu_cores_per_event',
    'memory_occupancy', 'total_memory_used_mb', 'memory_mb_per_event',
    'total_network_transfer_mb', 'network_transfer_mb_per_event',
    'total_write_remote_mb_per_event', 'total_read_remote_mb_per_event',
]


def load_scenario_metrics(simulation_dir: str) -> pd.DataFrame:
    """Load the eleven metrics for all construction JSONs in the scenario directory."""
    path = Path(simulation_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {simulation_dir}")
    json_files = sorted(path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files in {simulation_dir}")

    rows = []
    for fp in json_files:
        try:
            with open(fp, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  Warning: Skip {fp.name}: {e}")
            continue
        metrics = data.get('metrics', {})
        comp = metrics.get('composition_number')
        if comp is None:
            stem = fp.stem
            if '_const_' in stem:
                try:
                    comp = int(stem.split('_const_')[-1])
                except ValueError:
                    comp = len(rows) + 1
            else:
                comp = len(rows) + 1
        row = {'construction': comp}
        for key in METRIC_KEYS:
            row[key] = metrics.get(key, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values('construction').reset_index(drop=True)
    if df.empty or len(df) < 2:
        raise ValueError("Need at least 2 constructions for correlation")
    return df


def run_correlation(df: pd.DataFrame) -> None:
    """Print Pearson correlation of each CPU, memory, and I/O metric with event_throughput."""
    print("\n" + "=" * 60)
    print("CORRELATION WITH EVENT_THROUGHPUT")
    print("=" * 60)
    target = 'event_throughput'
    for p in PREDICTORS_ORDERED:
        r = df[target].corr(df[p])
        print(f"  {p:30s}  r = {r:+.4f}")
    print("\n  Pearson r: +1 = perfect positive, -1 = perfect negative, 0 = no linear relation.")
    print("  Higher |r| = stronger linear relationship with throughput.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correlation of CPU, memory, and I/O metrics with event_throughput (which best relates?)."
    )
    parser.add_argument(
        'simulation_dir',
        type=str,
        help="Scenario directory (e.g. results/sim/others/seq_real/12h/fr5/100MBps)",
    )
    args = parser.parse_args()

    print(f"Loading: {args.simulation_dir}")
    df = load_scenario_metrics(args.simulation_dir)
    print(f"  Loaded {len(df)} constructions.")
    run_correlation(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
