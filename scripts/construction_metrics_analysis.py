#!/usr/bin/env python3
"""
Construction Metrics Analysis – multi-metric comparison for one scenario.

For a given scenario (workflow type, target job length, failure rate, data rate),
loads all 16 workflow construction results from the scenario directory, normalizes
selected metrics to [0, 1] (higher = better), and produces:
- A normalized heatmap (constructions x metrics) for quick comparison
- A single weighted score per construction (sum of weights = 1.0) and two score plots
- A CSV of raw and normalized metrics plus weighted_score (same metrics as heatmap)

Metrics (11 total, same for heatmap and CSV): event_throughput, total_cpu_cores_used,
cpu_utilization, cpu_cores_per_event, total_memory_used_mb, memory_occupancy,
memory_mb_per_event, total_turnaround_time, wall_time_per_event,
network_transfer_mb_per_event, total_write_remote_mb.
Score uses: event_throughput, cpu_cores_per_event, memory_mb_per_event,
network_transfer_mb_per_event (lower is better for the last three; normalized so 1 = best).

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


# Weighted score: metric key -> weight (weights must sum to 1.0). Focus: throughput + per-event resource use.
SCORE_METRICS_WEIGHTS = {
    'event_throughput': 0.4,
    'cpu_cores_per_event': 0.2,
    'memory_mb_per_event': 0.2,
    'network_transfer_mb_per_event': 0.2,
}

# Single source of truth: (key, label, higher_is_better). Full list = heatmap order.
_ALL_SPECS = [
    ('event_throughput', 'Throughput', True),
    ('cpu_cores_per_event', 'CPU Cores/Evt', False),
    ('memory_mb_per_event', 'Memory MB/Evt', False),
    ('cpu_utilization', 'CPU Util', True),
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


def compute_weighted_score(
    norm_matrix: np.ndarray,
    metrics_weights: Dict[str, float] = None,
) -> np.ndarray:
    """Compute a single weighted score per construction (normalized 0--1).

    Score = sum(weight_i * normalized_metric_i) over the selected metrics.
    metrics_weights: dict of metric_key -> weight; weights must sum to 1.0.
    Uses column order from HEATMAP_METRIC_SPECS.
    """
    if metrics_weights is None:
        metrics_weights = SCORE_METRICS_WEIGHTS
    metric_keys = list(metrics_weights.keys())
    weights = np.asarray(list(metrics_weights.values()), dtype=float)
    if abs(weights.sum() - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1.0, got {weights.sum()}")
    key_to_col = {s[0]: j for j, s in enumerate(HEATMAP_METRIC_SPECS)}
    indices = [key_to_col[k] for k in metric_keys]
    # (n_const, n_metrics) * (n_metrics,) -> sum over axis=1
    scores = (norm_matrix[:, indices] * weights).sum(axis=1)
    return scores


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


def plot_score_bars(
    composition_numbers: List[int],
    scores: np.ndarray,
    output_path: str,
    scenario_label: str = "",
) -> None:
    """Bar chart: construction (x) vs weighted score (y). Color by score (green=high)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(composition_numbers))
    bars = ax.bar(x, scores, color=plt.cm.RdYlGn(scores), edgecolor='gray', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Const {c}" for c in composition_numbers], rotation=45, ha='right')
    ax.set_ylabel("Weighted score (0–1)")
    ax.set_xlabel("Workflow construction")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Weighted score by construction – {scenario_label}"
        if scenario_label else "Weighted score by construction"
    )
    ax.axhline(y=scores.mean(), color='gray', linestyle='--', alpha=0.7, label=f"Mean = {scores.mean():.3f}")
    ax.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  => Score bar chart saved to {output_path}")


def plot_score_ranked(
    composition_numbers: List[int],
    scores: np.ndarray,
    output_path: str,
    scenario_label: str = "",
) -> None:
    """Horizontal bar chart: constructions sorted by score (best at top)."""
    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    sorted_labels = [f"Const {composition_numbers[i]}" for i in order]
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(sorted_labels))
    colors = plt.cm.RdYlGn(sorted_scores)
    ax.barh(y_pos, sorted_scores, color=colors, edgecolor='gray', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_labels)
    ax.set_xlabel("Weighted score (0–1)")
    ax.set_xlim(0, 1.05)
    ax.xaxis.grid(True, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title(
        f"Constructions ranked by score – {scenario_label}"
        if scenario_label else "Constructions ranked by score"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  => Score ranking saved to {output_path}")


def export_metrics_csv(
    all_metrics: List[Dict[str, Any]],
    norm_matrix: np.ndarray,
    composition_numbers: List[int],
    output_path: str,
    weighted_scores: np.ndarray = None,
) -> None:
    """Write CSV with construction, raw metrics, normalized scores, and optional weighted score.

    Exports the same metrics as the heatmap, in the same order. If weighted_scores
    is provided, adds a 'weighted_score' column.
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
        if weighted_scores is not None and i < len(weighted_scores):
            row['weighted_score'] = weighted_scores[i]
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
    scores = compute_weighted_score(norm_matrix)

    plot_normalized_heatmap(
        norm_matrix,
        composition_numbers,
        os.path.join(out_dir, "construction_metrics_heatmap.png"),
        scenario_label=scenario_label,
    )
    plot_score_bars(
        composition_numbers,
        scores,
        os.path.join(out_dir, "construction_score_bars.png"),
        scenario_label=scenario_label,
    )
    plot_score_ranked(
        composition_numbers,
        scores,
        os.path.join(out_dir, "construction_score_ranked.png"),
        scenario_label=scenario_label,
    )
    export_metrics_csv(
        all_metrics,
        norm_matrix,
        composition_numbers,
        os.path.join(out_dir, "construction_metrics.csv"),
        weighted_scores=scores,
    )
    print("Done.")


if __name__ == "__main__":
    main()
