#!/usr/bin/env python3
"""
Construction Metrics Analysis – multi-metric comparison for one scenario.

For a given scenario (workflow type, target job length, failure rate, data rate),
loads all 16 workflow construction results from the scenario directory, normalizes
selected metrics to [0, 1] (higher = better), and produces:
- A normalized heatmap (constructions x metrics) for quick comparison
- A CSV of raw and normalized metrics (same metrics and order as the heatmap)

Metrics (9 total, same for heatmap and CSV): event_throughput, total_cpu_cores_used,
cpu_utilization, total_memory_used_mb, memory_occupancy, total_turnaround_time,
wall_time_per_event, network_transfer_mb_per_event, total_write_remote_mb.
Higher-is-better: throughput, CPU util, memory occupancy. Lower-is-better: the rest
(inverted so normalized score 1 = best).

Output directory follows the same schema as other analysis scripts: output goes to
results/analysis/construction_metrics and is suffixed by workflow type, target job
length, failure rate and data rate (e.g. results/analysis/construction_metrics/
case1_real/12h/fr5/100MBps). When --output-dir is omitted, this path is derived
from the simulation directory path when it matches the standard sim tree.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Single source of truth: (key, label, higher_is_better). Full list = heatmap order.
_ALL_SPECS = [
    ('event_throughput', 'Throughput', True),
    ('total_cpu_cores_used', 'Alloc CPU Cores', False),
    ('cpu_utilization', 'CPU Util', True),
    ('total_memory_used_mb', 'Alloc Memory', False),
    ('memory_occupancy', 'Memory Occ', True),
    ('total_turnaround_time', 'Turnaround', False),
    ('wall_time_per_event', 'Wall Time/Evt', False),
    ('network_transfer_mb_per_event', 'Net MB/Evt', False),
    ('total_write_remote_mb', 'Write Remote', False),
]
HEATMAP_METRIC_SPECS = _ALL_SPECS
KEYS_TO_LOAD = [s[0] for s in HEATMAP_METRIC_SPECS]

# Output base for analysis; suffix = workflow_type/time/failure_rate/data_rate
CONSTRUCTION_METRICS_OUTPUT_BASE = "results/analysis/construction_metrics"


def default_output_dir_from_simulation_dir(simulation_dir: str) -> str:
    """Derive default output dir from simulation path using the analysis schema.

    If simulation_dir looks like .../sim/others/<case>/<time>/<fr>/<rate>,
    returns results/analysis/construction_metrics/<case>/<time>/<fr>/<rate>.
    Otherwise returns simulation_dir (write next to input).
    """
    path = Path(simulation_dir).resolve()
    parts = path.parts
    for i, p in enumerate(parts):
        if p == "others" and i > 0 and parts[i - 1] == "sim":
            suffix_parts = parts[i + 1:]
            if len(suffix_parts) >= 4:
                return str(Path(CONSTRUCTION_METRICS_OUTPUT_BASE, *suffix_parts[:4]))
            break
    return simulation_dir


def load_scenario_metrics(simulation_dir: str) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Load metrics for all construction JSON files in the scenario directory.

    Args:
        simulation_dir: Path to directory containing *_const_*.json files.

    Returns:
        (list of metric dicts per construction, list of composition numbers).
        Sorted by composition_number; composition_number taken from metrics or filename.
    """
    path = Path(simulation_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Scenario directory not found: {simulation_dir}")

    json_files = sorted(path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files in {simulation_dir}")

    rows: List[Tuple[int, Dict[str, Any]]] = []
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
            # Fallback: case1_real_const_003.json -> 3
            stem = fp.stem
            if '_const_' in stem:
                try:
                    comp = int(stem.split('_const_')[-1])
                except ValueError:
                    comp = len(rows) + 1
            else:
                comp = len(rows) + 1
        row = {key: metrics.get(key, 0.0) for key in KEYS_TO_LOAD}
        row['composition_number'] = comp
        rows.append((comp, row))

    rows.sort(key=lambda x: x[0])
    composition_numbers = [r[0] for r in rows]
    all_metrics = [r[1] for r in rows]
    return all_metrics, composition_numbers


def normalize_metrics(
    all_metrics: List[Dict[str, Any]],
    metric_specs: List[Tuple[str, str, bool]] = None,
) -> np.ndarray:
    """Normalize each metric to [0, 1]; higher normalized value = better.

    For higher-is-better: (x - min) / (max - min).
    For lower-is-better: (max - x) / (max - min). Handles constant column (-> 0.5).

    Args:
        all_metrics: List of metric dicts per construction.
        metric_specs: List of (key, label, higher_is_better). Default: HEATMAP_METRIC_SPECS.
    """
    if metric_specs is None:
        metric_specs = HEATMAP_METRIC_SPECS
    n = len(all_metrics)
    keys = [m[0] for m in metric_specs]
    higher = {m[0]: m[2] for m in metric_specs}
    M = np.zeros((n, len(keys)))
    for j, key in enumerate(keys):
        vals = np.array([m[key] for m in all_metrics], dtype=float)
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            if higher[key]:
                M[:, j] = (vals - vmin) / (vmax - vmin)
            else:
                M[:, j] = (vmax - vals) / (vmax - vmin)
        else:
            M[:, j] = 0.5
    return M


def plot_normalized_heatmap(
    norm_matrix: np.ndarray,
    composition_numbers: List[int],
    output_path: str,
    scenario_label: str = "",
    metric_labels: List[str] = None,
) -> None:
    """Plot constructions x metrics heatmap (normalized; green = better)."""
    if metric_labels is None:
        metric_labels = [s[1] for s in HEATMAP_METRIC_SPECS]
    n_rows, n_cols = norm_matrix.shape
    fig, ax = plt.subplots(figsize=(max(6, n_cols * 1.2), max(4, n_rows * 0.4)))
    im = ax.imshow(norm_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(metric_labels, rotation=45, ha='right')
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([f"Const {c}" for c in composition_numbers])
    ax.set_xlabel("Metric (normalized: 1 = best)")
    ax.set_ylabel("Workflow construction")
    if scenario_label:
        ax.set_title(f"Normalized metrics by construction – {scenario_label}")
    else:
        ax.set_title("Normalized metrics by construction")
    plt.colorbar(im, ax=ax, label="Score (0–1)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  => Heatmap saved to {output_path}")


def export_metrics_csv(
    all_metrics: List[Dict[str, Any]],
    norm_matrix: np.ndarray,
    composition_numbers: List[int],
    output_path: str,
) -> None:
    """Write CSV with construction, raw metrics, and normalized scores for scoring.

    Exports the same metrics as the heatmap, in the same order.
    """
    rows = []
    for i, (comp, raw) in enumerate(zip(composition_numbers, all_metrics)):
        row = {'construction': comp}
        for j, (key, _, _) in enumerate(HEATMAP_METRIC_SPECS):
            row[f'{key}_raw'] = raw[key]
            row[f'{key}_normalized'] = (
                norm_matrix[i, j] if i < norm_matrix.shape[0] and j < norm_matrix.shape[1]
                else None
            )
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  => Metrics CSV saved to {output_path}")


def build_scenario_label(use_case: str, time_dir: str, failure_rate: str,
                         data_rate: str) -> str:
    """Build a short label from scenario path components."""
    return f"{use_case} / {time_dir} / {failure_rate} / {data_rate}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-metric construction analysis for one scenario (16 constructions)."
    )
    parser.add_argument(
        'simulation_dir',
        type=str,
        help="Path to scenario directory (e.g. results/sim/others/case1_real/1h/fr1/100MBps)",
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help="Output directory for plots and CSV (default: results/analysis/"
             "construction_metrics/<workflow_type>/<time>/<fr>/<data_rate> from path)",
    )
    parser.add_argument(
        '--scenario-label',
        type=str,
        default=None,
        help="Optional label for plot titles (e.g. 'case1_real 1h fr1 100MBps')",
    )
    args = parser.parse_args()

    sim_dir = args.simulation_dir
    out_dir = args.output_dir or default_output_dir_from_simulation_dir(sim_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    scenario_label = args.scenario_label or sim_dir

    print(f"Loading scenario: {sim_dir}")
    all_metrics, composition_numbers = load_scenario_metrics(sim_dir)
    if len(all_metrics) == 0:
        print("No valid construction data found.")
        return
    print(f"  Loaded {len(all_metrics)} constructions: {composition_numbers}")

    norm_matrix = normalize_metrics(all_metrics)  # same for heatmap and CSV

    plot_normalized_heatmap(
        norm_matrix,
        composition_numbers,
        os.path.join(out_dir, "construction_metrics_heatmap.png"),
        scenario_label=scenario_label,
    )
    export_metrics_csv(
        all_metrics,
        norm_matrix,
        composition_numbers,
        os.path.join(out_dir, "construction_metrics.csv"),
    )
    print("Done.")


if __name__ == "__main__":
    main()
