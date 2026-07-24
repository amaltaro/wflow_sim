#!/usr/bin/env python3
"""
Aggregate multi-seed simulation results and plot means with error bars.

Reads ``results/sim/rebuttal/seed*/...`` (or ``--input-root``), groups runs by
``composition_number``, and writes comparison PNGs under ``--output-dir``.

When job failure rate is > 0 and N > 1, error bars show SEM
(``std / sqrt(N)``). At failure rate 0, means are plotted without error bars.

Reuses layout helpers from ``workflow_visualization.py``; does not change that
script or the Makefile.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from workflow_visualization import (  # noqa: E402
    PERF_SCATTER_FIG_H_IN,
    PERF_SCATTER_FIG_W_IN,
    PROCESSING_EFFICIENCY_BAR_DATA_W,
    PROCESSING_EFFICIENCY_FIG_H_IN,
    RESOURCE_COST_FIG_H_IN,
    RESOURCE_COST_SINGLE_BAR_DATA_W,
    RESOURCE_UTIL_BAR_DATA_W,
    RESOURCE_UTIL_STACK_FIG_H_IN,
    TURNAROUND_BAR_DATA_W,
    TURNAROUND_TIME_FIG_H_IN,
    _annotate_construction_scatter_labels,
    _comparison_figure_width_inches,
    _comparison_xtick_labels,
    _legend_kwargs,
    _resource_util_panel_center_banner,
    _set_comparison_xlim,
    _tight_axis_limits,
)

# Metrics aggregated across seeds (workflow-level ``metrics`` keys).
METRIC_KEYS: Tuple[str, ...] = (
    "cpu_time_per_event",
    "cpu_utilization",
    "event_throughput",
    "total_write_remote_mb_per_event",
    "total_turnaround_time",
    "network_transfer_mb_per_event",
    "memory_occupancy",
    "total_cpu_cores_used",
    "total_memory_used_mb",
)


def load_campaign_manifest(input_root: Path) -> Optional[Dict[str, Any]]:
    """Load ``campaign.json`` if present under ``input_root``."""
    path = input_root / "campaign.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def discover_result_files(input_root: Path) -> List[Path]:
    """Find simulation result JSON files under ``seed*`` trees (skip campaign)."""
    files: List[Path] = []
    for seed_dir in sorted(input_root.glob("seed*")):
        if not seed_dir.is_dir():
            continue
        for path in sorted(seed_dir.glob("**/*.json")):
            if path.name == "campaign.json":
                continue
            files.append(path)
    return files


def _composition_number(data: Dict[str, Any], path: Path) -> int:
    """Prefer metrics.composition_number; fall back to filename const index."""
    metrics = data.get("metrics") or {}
    comp = metrics.get("composition_number")
    if isinstance(comp, int) and comp > 0:
        return comp
    stem = path.stem
    # e.g. seq_real_const_001
    if "_const_" in stem:
        try:
            return int(stem.rsplit("_const_", 1)[1])
        except ValueError:
            pass
    return 0


def load_run_metrics(path: Path) -> Optional[Dict[str, Any]]:
    """Load one simulation JSON into a flat metrics record, or None if invalid."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        return None
    sim_result = data.get("simulation_result") or {}
    record: Dict[str, Any] = {
        "composition_number": _composition_number(data, path),
        "random_seed": sim_result.get("random_seed"),
        "job_failure_rate": sim_result.get(
            "job_failure_rate", metrics.get("job_failure_rate", 0.0)
        ),
        "_path": str(path),
    }
    for key in METRIC_KEYS:
        record[key] = float(metrics.get(key, 0.0) or 0.0)
    # Network transfer fallback (same as workflow_visualization)
    if metrics.get("network_transfer_mb_per_event") is None:
        record["network_transfer_mb_per_event"] = float(
            metrics.get("total_read_remote_mb_per_event", 0.0) or 0.0
        ) + float(metrics.get("total_write_remote_mb_per_event", 0.0) or 0.0)
    return record


def mean_and_sem(values: Sequence[float]) -> Tuple[float, float]:
    """Return sample mean and SEM (0 if N < 2)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return 0.0, 0.0
    mean = float(np.mean(arr))
    if n < 2:
        return mean, 0.0
    sem = float(np.std(arr, ddof=1) / np.sqrt(n))
    return mean, sem


def aggregate_by_composition(
    records: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Group run records by composition; compute mean/SEM per metric.

    Returns a list sorted by ``composition_number``, each entry containing
    ``composition_number``, ``n``, and for every metric key ``{key}`` (mean)
    plus ``{key}_sem``.
    """
    by_comp: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_comp[int(rec["composition_number"])].append(rec)

    aggregated: List[Dict[str, Any]] = []
    for comp in sorted(by_comp):
        runs = by_comp[comp]
        row: Dict[str, Any] = {
            "composition_number": comp,
            "n": len(runs),
        }
        for key in METRIC_KEYS:
            mean, sem = mean_and_sem([float(r.get(key, 0.0)) for r in runs])
            row[key] = mean
            row[f"{key}_sem"] = sem
        aggregated.append(row)
    return aggregated


def resolve_failure_rate(
    aggregated: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    manifest: Optional[Dict[str, Any]],
) -> float:
    """Infer campaign failure rate from manifest or result files."""
    if manifest and "job_failure_rate" in manifest:
        return float(manifest["job_failure_rate"])
    rates = [
        float(r["job_failure_rate"])
        for r in records
        if r.get("job_failure_rate") is not None
    ]
    if rates:
        return float(np.median(rates))
    return 0.0


def should_draw_error_bars(failure_rate: float, n_runs: int) -> bool:
    """Error bars only when failures are stochastic and multiple runs exist."""
    return failure_rate > 0.0 and n_runs > 1


def _metric_arrays(
    aggregated: List[Dict[str, Any]], key: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Return mean and SEM arrays for ``key`` across constructions."""
    means = np.array([float(r[key]) for r in aggregated], dtype=float)
    sems = np.array([float(r[f"{key}_sem"]) for r in aggregated], dtype=float)
    return means, sems


def _yerr_or_none(sems: np.ndarray, draw: bool) -> Optional[np.ndarray]:
    """Return SEM array for matplotlib when error bars are enabled."""
    if not draw:
        return None
    return sems


def _ylim_from_mean_err(
    means: np.ndarray,
    sems: np.ndarray,
    *,
    draw_err: bool,
    clamp_non_negative: bool = True,
    pad_rel: float = 0.10,
) -> Tuple[float, float]:
    """Tight y-limits including error-bar extent when drawn."""
    lo_vals = means - sems if draw_err else means
    hi_vals = means + sems if draw_err else means
    combined = np.concatenate([lo_vals, hi_vals])
    return _tight_axis_limits(
        combined, clamp_non_negative=clamp_non_negative, pad_rel=pad_rel
    )


def plot_processing_efficiency(
    aggregated: List[Dict[str, Any]],
    output_dir: Path,
    *,
    draw_error_bars: bool,
    n_runs: int,
) -> Path:
    """CPU time/event bars + utilization line (mean ± SEM when enabled)."""
    cpu_mean, cpu_sem = _metric_arrays(aggregated, "cpu_time_per_event")
    util_mean, util_sem = _metric_arrays(aggregated, "cpu_utilization")
    n_plot = len(aggregated)
    x = np.arange(n_plot)
    width = PROCESSING_EFFICIENCY_BAR_DATA_W
    fig_w = _comparison_figure_width_inches(n_plot, bar_width=width)
    wc_xticks = _comparison_xtick_labels(n_plot)

    fig, ax = plt.subplots(
        figsize=(fig_w, PROCESSING_EFFICIENCY_FIG_H_IN),
        layout="constrained",
    )
    ax_twin = ax.twinx()
    yerr = _yerr_or_none(cpu_sem, draw_error_bars)
    ax.bar(
        x,
        cpu_mean,
        width,
        yerr=yerr,
        label="CPU Time per Event",
        color="#2ca02c",
        alpha=0.7,
        capsize=3 if draw_error_bars else 0,
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
    )
    ax.set_xlabel("Workflow Construction")
    ax.set_ylabel("CPU Time per Event (seconds)", color="#2ca02c")
    ax.tick_params(axis="y", labelcolor="#2ca02c")

    util_yerr = _yerr_or_none(util_sem, draw_error_bars)
    ax_twin.errorbar(
        x,
        util_mean,
        yerr=util_yerr,
        fmt="o-",
        color="#d62728",
        linewidth=1.5,
        markersize=4,
        capsize=3 if draw_error_bars else 0,
        label="CPU Utilization",
    )
    ax_twin.set_ylabel("CPU Utilization Ratio", color="#d62728")
    ax_twin.tick_params(axis="y", labelcolor="#d62728")
    ax_twin.set_ylim(0, 1)

    bottom, top = _ylim_from_mean_err(
        cpu_mean, cpu_sem, draw_err=draw_error_bars, clamp_non_negative=True
    )
    ax.set_ylim(bottom, top)

    title = "Processing Efficiency Analysis"
    if draw_error_bars:
        title += f" (mean ± SEM, N={n_runs})"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(wc_xticks, rotation=0, ha="center")
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    # Outside below axes: dual-axis series leave little interior free space
    # (tall left bars; utilization line high; mid bars ~48s).
    fig.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="outside lower center",
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
    )
    _set_comparison_xlim(ax, n_plot, width)

    out = output_dir / "processing_efficiency_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  => {out}")
    return out


def plot_performance_vs_remote_write(
    aggregated: List[Dict[str, Any]],
    output_dir: Path,
    *,
    n_runs: int,
) -> Path:
    """Throughput vs remote write scatter of per-construction means (no SEM)."""
    thr_mean, _thr_sem = _metric_arrays(aggregated, "event_throughput")
    wr_mean, _wr_sem = _metric_arrays(aggregated, "total_write_remote_mb_per_event")

    fig, ax = plt.subplots(
        figsize=(PERF_SCATTER_FIG_W_IN, PERF_SCATTER_FIG_H_IN),
        layout="constrained",
    )
    ax.scatter(
        thr_mean,
        wr_mean,
        s=72,
        c="#4c72b0",
        edgecolors="white",
        linewidths=1.0,
        alpha=0.88,
        zorder=3,
    )
    _annotate_construction_scatter_labels(ax, thr_mean, wr_mean)
    ax.set_xlabel("Throughput (events/second)")
    ax.set_ylabel("Remote Write per Event (MB)")
    ax.set_title(
        f"Performance vs Remote Write Efficiency (mean, N={n_runs})",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)

    x_lo, x_hi = _tight_axis_limits(thr_mean, clamp_non_negative=False, pad_rel=0.16)
    y_lo, y_hi = _tight_axis_limits(wr_mean, clamp_non_negative=True, pad_rel=0.16)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    out = output_dir / "performance_vs_remote_write_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  => {out}")
    return out


def plot_turnaround_time(
    aggregated: List[Dict[str, Any]],
    output_dir: Path,
    *,
    draw_error_bars: bool,
    n_runs: int,
) -> Path:
    """Turnaround time bars in hours with optional SEM error bars."""
    tt_s, tt_sem_s = _metric_arrays(aggregated, "total_turnaround_time")
    hours = tt_s / 3600.0
    hours_sem = tt_sem_s / 3600.0
    n_plot = len(aggregated)
    x = np.arange(n_plot)
    width = TURNAROUND_BAR_DATA_W
    fig_w = _comparison_figure_width_inches(n_plot, bar_width=width)
    wc_xticks = _comparison_xtick_labels(n_plot)

    fig, ax = plt.subplots(
        figsize=(fig_w, TURNAROUND_TIME_FIG_H_IN),
        layout="constrained",
    )
    ax.bar(
        x,
        hours,
        width,
        yerr=_yerr_or_none(hours_sem, draw_error_bars),
        color="#17becf",
        alpha=0.85,
        capsize=3 if draw_error_bars else 0,
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
    )
    ax.set_xlabel("Workflow Construction")
    ax.set_ylabel("Turnaround Time (hours)")
    title = "Workflow Turnaround Time by Composition"
    if draw_error_bars:
        title += f" (mean ± SEM, N={n_runs})"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(wc_xticks, rotation=0, ha="center")
    ax.grid(True, axis="y", alpha=0.3)
    y_lo, y_hi = _ylim_from_mean_err(
        hours, hours_sem, draw_err=draw_error_bars, clamp_non_negative=True
    )
    ax.set_ylim(max(0.0, y_lo), y_hi)
    _set_comparison_xlim(ax, n_plot, width)

    out = output_dir / "turnaround_time_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  => {out}")
    return out


def plot_resource_utilization(
    aggregated: List[Dict[str, Any]],
    output_dir: Path,
    *,
    draw_error_bars: bool,
    n_runs: int,
) -> Tuple[Path, Path]:
    """Resource util stack + cost dual-axis plots with optional SEM bars."""
    net_mean, net_sem = _metric_arrays(aggregated, "network_transfer_mb_per_event")
    mem_mean, mem_sem = _metric_arrays(aggregated, "memory_occupancy")
    cpu_mean, cpu_sem = _metric_arrays(aggregated, "cpu_utilization")
    cores_mean, cores_sem = _metric_arrays(aggregated, "total_cpu_cores_used")
    mem_mb_mean, mem_mb_sem = _metric_arrays(aggregated, "total_memory_used_mb")
    mem_gb_mean = mem_mb_mean / 1024.0
    mem_gb_sem = mem_mb_sem / 1024.0

    n_plot = len(aggregated)
    x = np.arange(n_plot)
    width = RESOURCE_UTIL_BAR_DATA_W
    fig_w = _comparison_figure_width_inches(n_plot, bar_width=width)
    wc_xticks = _comparison_xtick_labels(n_plot)

    fig_u, axes = plt.subplots(
        3,
        1,
        figsize=(fig_w, RESOURCE_UTIL_STACK_FIG_H_IN),
        sharex=True,
        layout="constrained",
    )
    panels = (
        (axes[0], net_mean, net_sem, "Network transfer per event (MB)", "Network", "#1f77b4"),
        (axes[1], mem_mean, mem_sem, "Memory occupancy ratio", "Memory", "#ff7f0e"),
        (axes[2], cpu_mean, cpu_sem, "CPU utilization ratio", "CPU", "#2ca02c"),
    )
    for ax, means, sems, ylabel, banner, color in panels:
        ax.bar(
            x,
            means,
            width,
            yerr=_yerr_or_none(sems, draw_error_bars),
            color=color,
            alpha=0.75,
            capsize=3 if draw_error_bars else 0,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
        ax.set_ylabel(ylabel)
        _resource_util_panel_center_banner(ax, banner, color)
        ax.grid(True, axis="y", alpha=0.3)
        if banner in ("Memory", "CPU"):
            ax.set_ylim(0, 1)
        else:
            lo, hi = _ylim_from_mean_err(
                means, sems, draw_err=draw_error_bars, clamp_non_negative=True
            )
            ax.set_ylim(lo, hi)
        _set_comparison_xlim(ax, n_plot, width)

    axes[2].set_xlabel("Workflow Construction")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(wc_xticks, rotation=0, ha="center")
    title = "Resource Utilization"
    if draw_error_bars:
        title += f" (mean ± SEM, N={n_runs})"
    fig_u.suptitle(title)
    fig_u.align_ylabels(axes)

    out_u = output_dir / "resource_utilization_comparison.png"
    fig_u.savefig(out_u)
    plt.close(fig_u)
    print(f"  => {out_u}")

    # Cost plot (colors/labels match workflow_visualization.plot_resource_utilization)
    fig_c, ax_c = plt.subplots(
        figsize=(fig_w, RESOURCE_COST_FIG_H_IN),
        layout="constrained",
    )
    ax_c2 = ax_c.twinx()
    w = RESOURCE_COST_SINGLE_BAR_DATA_W
    ax_c.bar(
        x - w / 2,
        cores_mean,
        w,
        yerr=_yerr_or_none(cores_sem, draw_error_bars),
        color="#8c564b",
        alpha=0.7,
        label="Total CPU Cores",
        capsize=3 if draw_error_bars else 0,
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
    )
    ax_c2.bar(
        x + w / 2,
        mem_gb_mean,
        w,
        yerr=_yerr_or_none(mem_gb_sem, draw_error_bars),
        color="#ff7f0e",
        alpha=0.7,
        label="Total Memory (GB)",
        capsize=3 if draw_error_bars else 0,
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
    )
    ax_c.set_xlabel("Workflow Construction")
    ax_c.set_ylabel("Total CPU Cores Used", color="#8c564b")
    ax_c2.set_ylabel("Total Memory Used (GB)", color="#ff7f0e")
    ax_c.tick_params(axis="y", labelcolor="#8c564b")
    ax_c2.tick_params(axis="y", labelcolor="#ff7f0e")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(wc_xticks, rotation=0, ha="center")
    cost_title = "Overall Resource Cost Analysis"
    if draw_error_bars:
        cost_title += f" (mean ± SEM, N={n_runs})"
    ax_c.set_title(cost_title)
    ax_c.grid(True, alpha=0.3)
    lines1, labels1 = ax_c.get_legend_handles_labels()
    lines2, labels2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(
        lines1 + lines2,
        labels1 + labels2,
        **_legend_kwargs(loc="upper right"),
    )
    _set_comparison_xlim(ax_c, n_plot, 2.0 * w)

    out_c = output_dir / "resource_cost_comparison.png"
    fig_c.savefig(out_c)
    plt.close(fig_c)
    print(f"  => {out_c}")
    return out_u, out_c


def write_aggregation_csv(
    aggregated: List[Dict[str, Any]], output_dir: Path
) -> Path:
    """Write mean/SEM table for paper methods / rebuttal."""
    import csv

    fieldnames = ["composition_number", "n"]
    for key in METRIC_KEYS:
        fieldnames.extend([key, f"{key}_sem"])

    out = output_dir / "multiseed_aggregation_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in aggregated:
            writer.writerow(row)
    print(f"  => {out}")
    return out


def default_output_dir(
    input_root: Path,
    manifest: Optional[Dict[str, Any]],
    records: List[Dict[str, Any]],
) -> Path:
    """Mirror scenario path under ``results/vis/rebuttal`` when possible."""
    use_case = (manifest or {}).get("use_case", "unknown")
    # Infer time/fr/rate from first result path if present
    time_dir, fr_dir, rate_dir = "12h", "fr5", "100MBps"
    if records:
        parts = Path(records[0]["_path"]).parts
        for i, part in enumerate(parts):
            if part.startswith("fr") and part[2:].isdigit():
                fr_dir = part
                if i >= 1:
                    time_dir = parts[i - 1]
                if i + 1 < len(parts):
                    rate_dir = parts[i + 1]
                break
    return Path("results/vis/rebuttal") / use_case / time_dir / fr_dir / rate_dir


def run_visualization(
    *,
    input_root: Path,
    output_dir: Optional[Path] = None,
) -> int:
    """Load multi-seed results, aggregate, and write plots."""
    if not input_root.is_dir():
        print(f"ERROR: input root not found: {input_root}", file=sys.stderr)
        return 1

    manifest = load_campaign_manifest(input_root)
    files = discover_result_files(input_root)
    if not files:
        print(f"ERROR: no seed*/ result JSON under {input_root}", file=sys.stderr)
        return 1

    records: List[Dict[str, Any]] = []
    for path in files:
        rec = load_run_metrics(path)
        if rec is None:
            print(f"  Warning: skipping invalid file {path}")
            continue
        records.append(rec)

    if not records:
        print("ERROR: no valid simulation metrics loaded", file=sys.stderr)
        return 1

    aggregated = aggregate_by_composition(records)
    failure_rate = resolve_failure_rate(aggregated, records, manifest)
    n_runs = int((manifest or {}).get("runs") or aggregated[0]["n"])
    draw_err = should_draw_error_bars(failure_rate, n_runs)

    out_dir = output_dir or default_output_dir(input_root, manifest, records)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Multi-seed visualization")
    print(f"  input:         {input_root}")
    print(f"  constructions: {len(aggregated)}")
    print(f"  runs/N:        {n_runs}")
    print(f"  failure rate:  {failure_rate}%")
    print(f"  error bars:    {'SEM' if draw_err else 'off'}")
    print(f"  output:        {out_dir}")
    print("")

    write_aggregation_csv(aggregated, out_dir)
    plot_processing_efficiency(
        aggregated, out_dir, draw_error_bars=draw_err, n_runs=n_runs
    )
    plot_performance_vs_remote_write(aggregated, out_dir, n_runs=n_runs)
    plot_turnaround_time(
        aggregated, out_dir, draw_error_bars=draw_err, n_runs=n_runs
    )
    plot_resource_utilization(
        aggregated, out_dir, draw_error_bars=draw_err, n_runs=n_runs
    )
    print("Done.")
    return 0


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate multi-seed simulation results and plot means with "
            "SEM error bars when failure rate > 0."
        )
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="results/sim/rebuttal",
        help="Root with seed*/ trees and optional campaign.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory for PNGs/CSV "
            "(default: results/vis/rebuttal/<use_case>/<time>/fr<fr>/<rate>/)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_arguments(argv)
    output_dir = Path(args.output_dir) if args.output_dir else None
    return run_visualization(
        input_root=Path(args.input_root),
        output_dir=output_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
