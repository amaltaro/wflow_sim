import argparse
import json
import os
from typing import Any, Dict, List
from collections import defaultdict
from math import ceil
from pprint import pformat
import matplotlib
# Set non-interactive backend to avoid display issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

from matplotlib.ticker import AutoMinorLocator


def _tight_axis_limits(
    values: np.ndarray,
    *,
    pad_rel: float = 0.12,
    clamp_non_negative: bool = False,
) -> tuple[float, float]:
    """Axis limits with padding around finite data (optionally floor at 0 for non-negative series)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = float(np.min(v)), float(np.max(v))
    span = hi - lo
    if span <= 0.0 or not np.isfinite(span):
        mag = abs(hi) if hi != 0.0 else 1.0
        span = max(mag * 0.05, 1e-12)
    pad = max(span * pad_rel, max(abs(lo), abs(hi)) * 0.02, 1e-15)
    bottom = lo - pad
    top = hi + pad
    if clamp_non_negative and lo >= 0.0:
        bottom = max(0.0, bottom)
    if bottom >= top:
        top = bottom + max(span, 1e-9)
    return bottom, top


def _legend_kwargs(**overrides: Any) -> Dict[str, Any]:
    """Defaults for legends drawn over bars/points: light frame so data stays visible."""
    kw: Dict[str, Any] = {"frameon": True, "fancybox": True, "framealpha": 0.38}
    kw.update(overrides)
    return kw


# Stacked 2×1 figure size for ``plot_io_patterns`` only
STACKED_COMPARISON_FIG_W_IN = 7.0
STACKED_COMPARISON_FIG_H_IN = 6.0
# Split performance outputs: wide processing panel, narrow scatter (tight axes)
PROCESSING_EFFICIENCY_FIG_H_IN = STACKED_COMPARISON_FIG_H_IN / 2.0
PERF_SCATTER_FIG_W_IN = 4.0
PERF_SCATTER_FIG_H_IN = 3.0
# Resource utilization: 3×1 stack vs standalone cost figure
RESOURCE_UTIL_STACK_FIG_H_IN = STACKED_COMPARISON_FIG_H_IN * 1.5
RESOURCE_COST_FIG_H_IN = 4.0


def _style_stacked_total_data_volume_axis(ax: plt.Axes) -> None:
    """Denser y-axis grid for stacked total-volume bars (no numeric labels on bars)."""
    ax.set_axisbelow(True)
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(True, axis="y", which="major", alpha=0.42)
    ax.grid(True, axis="y", which="minor", alpha=0.24, linestyle=":")


def _stacked_total_volume_scale_and_unit_from_max_mb(max_total_mb: float) -> tuple[float, str]:
    """Scale stacked totals from MB into display units (binary 1024 steps).

    Chooses among MB, GB, TB, PB so the largest full-stack height maps to a readable
    magnitude. ``max_total_mb`` is the maximum over constructions of the sum of
    stacked segment sizes in megabytes.

    Returns
    -------
    scale
        Multiply segment values in **MB** by this to get the plotted number.
    unit
        One of ``MB``, ``GB``, ``TB``, ``PB``.
    """
    labels = ("MB", "GB", "TB", "PB")
    if not np.isfinite(max_total_mb) or max_total_mb <= 0:
        return 1.0, "MB"
    exp = 0
    while exp < len(labels) - 1 and max_total_mb >= 1024.0 ** (exp + 1):
        exp += 1
    return 1.0 / (1024.0**exp), labels[exp]


def plot_io_patterns(all_simulation_data: List[Dict],
                     sim_groups: List[Dict],
                     jobs: List[Dict],
                     output_dir: str = "plots",
                     custom_labels: List[str] = None):
    """Create I/O pattern analysis as two figures (2×1 stacked subplots).

    Saves:

    1. **io_patterns_comparison_local.png**: per-event (4 metrics) above stacked totals
       including local read.
    2. **io_patterns_comparison_nonlocal.png**: per-event (3 metrics) above stacked totals
       without local read.

    Stacked **total** panels sum workflow totals in **MB** and pick **MB / GB / TB / PB**
    on the y-axis using binary **1024** steps from the tallest stack.
    """
    print(f"==> Creating I/O pattern analysis for {len(all_simulation_data)} workflows")

    # Extract metrics for comparison
    event_throughputs = []
    write_remote_pevt = []
    total_write_remote = []
    read_remote_pevt = []
    write_local_pevt = []
    read_local_pevt = []
    construction_metrics = []  # Build this for text output

    for i, sim_data in enumerate(all_simulation_data):
        file_name = sim_data.get('_file_name', f'simulation_{i}')
        sim_groups_for_file = [g for g in sim_groups if g.get('file_name') == file_name]

        # Extract basic metrics
        event_throughputs.append(sim_data.get('event_throughput', 0.0))
        write_remote_pevt.append(sim_data.get('total_write_remote_mb_per_event', 0.0))
        total_write_remote.append(sim_data.get('total_write_remote_mb', 0.0))
        read_remote_pevt.append(sim_data.get('total_read_remote_mb_per_event', 0.0))
        write_local_pevt.append(sim_data.get('total_write_local_mb_per_event', 0.0))
        read_local_pevt.append(sim_data.get('total_read_local_mb_per_event', 0.0))

        # Build construction metrics for text output
        construction_metric = {
            'groups': [g['group_id'] for g in sim_groups_for_file],
            'num_groups': len(sim_groups_for_file),
            'event_throughput': sim_data.get('event_throughput', 0.0),
            'write_remote_per_event_mb': sim_data.get('total_write_remote_mb_per_event', 0.0),
            'read_remote_per_event_mb': sim_data.get('total_read_remote_mb_per_event', 0.0),
            'write_local_per_event_mb': sim_data.get('total_write_local_mb_per_event', 0.0),
            'read_local_per_event_mb': sim_data.get('total_read_local_mb_per_event', 0.0),
            'total_read_remote_mb': sim_data.get('total_read_remote_mb', 0.0),
            'total_read_local_mb': sim_data.get('total_read_local_mb', 0.0),
            'total_write_local_mb': sim_data.get('total_write_local_mb', 0.0),
            'total_write_remote_mb': sim_data.get('total_write_remote_mb', 0.0),
            'total_wallclock_time': sim_data.get('total_wall_time', 0.0),
            'total_memory_mb': sim_data.get('total_memory_used_mb', 0.0),
            'total_network_transfer_mb': sim_data.get('total_network_transfer_mb', 0.0),
            'network_transfer_per_event_mb': sim_data.get('network_transfer_mb_per_event', 0.0),
        }
        construction_metrics.append(construction_metric)

    # Convert lists to numpy arrays for numerical operations
    event_throughputs = np.array(event_throughputs)
    write_remote_pevt = np.array(write_remote_pevt)
    total_write_remote = np.array(total_write_remote)
    read_remote_pevt = np.array(read_remote_pevt)
    write_local_pevt = np.array(write_local_pevt)
    read_local_pevt = np.array(read_local_pevt)

    fig_w = STACKED_COMPARISON_FIG_W_IN
    fig_h = STACKED_COMPARISON_FIG_H_IN
    n_plot = len(all_simulation_data)
    wc_xticks = [str(i + 1) for i in range(n_plot)]
    x = np.arange(n_plot)

    colors = {
        "Local Read": "#1f77b4",
        "Remote Read": "#ff7f0e",
        "Local Write": "#2ca02c",
        "Remote Write": "#d62728",
    }

    # --- Figure 1: local read included (per-event on top, stacked totals below) ---
    fig1, (ax1, ax3) = plt.subplots(2, 1, figsize=(fig_w, fig_h), layout="constrained", sharex=True)

    width = 0.2
    ax1.bar(x - 1.5 * width, read_local_pevt, width, label="Local Read", color=colors["Local Read"])
    ax1.bar(x - 0.5 * width, read_remote_pevt, width, label="Remote Read", color=colors["Remote Read"])
    ax1.bar(x + 0.5 * width, write_local_pevt, width, label="Local Write", color=colors["Local Write"])
    ax1.bar(x + 1.5 * width, write_remote_pevt, width, label="Remote Write", color=colors["Remote Write"])
    ax1.set_xlabel("Workflow Construction")
    ax1.set_ylabel("Data Volume per Event (MB)")
    ax1.set_title("Data Volume Analysis Per Event (including local read)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(wc_xticks, rotation=0, ha="center")
    ax1.legend(**_legend_kwargs())
    ax1.grid(True)

    width_stack = 0.6
    bottom = np.zeros(n_plot)
    local_read_mb = np.array([m["total_read_local_mb"] for m in construction_metrics])
    remote_read_mb = np.array([m["total_read_remote_mb"] for m in construction_metrics])
    local_write_mb = np.array([m["total_write_local_mb"] for m in construction_metrics])
    remote_write_mb = np.array([m["total_write_remote_mb"] for m in construction_metrics])

    max_stack_mb = (
        float(np.max(local_read_mb + remote_read_mb + local_write_mb + remote_write_mb))
        if n_plot > 0
        else 0.0
    )
    vol_scale, vol_unit = _stacked_total_volume_scale_and_unit_from_max_mb(max_stack_mb)

    ax3.bar(
        x,
        local_read_mb * vol_scale,
        width_stack,
        label="Local Read",
        bottom=bottom,
        color=colors["Local Read"],
    )
    bottom = bottom + local_read_mb * vol_scale
    ax3.bar(
        x,
        remote_read_mb * vol_scale,
        width_stack,
        label="Remote Read",
        bottom=bottom,
        color=colors["Remote Read"],
    )
    bottom = bottom + remote_read_mb * vol_scale
    ax3.bar(
        x,
        local_write_mb * vol_scale,
        width_stack,
        label="Local Write",
        bottom=bottom,
        color=colors["Local Write"],
    )
    bottom = bottom + local_write_mb * vol_scale
    ax3.bar(
        x,
        remote_write_mb * vol_scale,
        width_stack,
        label="Remote Write",
        bottom=bottom,
        color=colors["Remote Write"],
    )

    ax3.set_xlabel("Workflow Construction")
    ax3.set_ylabel(f"Total Data Volume ({vol_unit})")
    ax3.set_title("Total Workflow Data Volume Analysis (including local read)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(wc_xticks, rotation=0, ha="center")
    ax3.legend(**_legend_kwargs())
    _style_stacked_total_data_volume_axis(ax3)

    fname_inc = "io_patterns_comparison_local.png"
    fig1.savefig(os.path.join(output_dir, fname_inc))
    plt.close(fig1)
    print(f"  => I/O patterns (local) saved to {output_dir}/{fname_inc}")

    # --- Figure 2: non-local only (per-event on top, stacked totals below) ---
    fig2, (ax2, ax4) = plt.subplots(2, 1, figsize=(fig_w, fig_h), layout="constrained", sharex=True)

    width3 = 0.25
    ax2.bar(x - width3, read_remote_pevt, width3, label="Remote Read", color=colors["Remote Read"])
    ax2.bar(x, write_local_pevt, width3, label="Local Write", color=colors["Local Write"])
    ax2.bar(x + width3, write_remote_pevt, width3, label="Remote Write", color=colors["Remote Write"])
    ax2.set_xlabel("Workflow Construction")
    ax2.set_ylabel("Data Volume per Event (MB)")
    ax2.set_title("Data Volume Analysis Per Event")
    ax2.set_xticks(x)
    ax2.set_xticklabels(wc_xticks, rotation=0, ha="center")
    ax2.legend(**_legend_kwargs())
    ax2.grid(True)

    bottom = np.zeros(n_plot)
    remote_read_mb2 = np.array([m["total_read_remote_mb"] for m in construction_metrics])
    local_write_mb2 = np.array([m["total_write_local_mb"] for m in construction_metrics])
    remote_write_mb2 = np.array([m["total_write_remote_mb"] for m in construction_metrics])

    max_stack_mb_nl = (
        float(np.max(remote_read_mb2 + local_write_mb2 + remote_write_mb2))
        if n_plot > 0
        else 0.0
    )
    vol_scale_nl, vol_unit_nl = _stacked_total_volume_scale_and_unit_from_max_mb(max_stack_mb_nl)

    ax4.bar(
        x,
        remote_read_mb2 * vol_scale_nl,
        width_stack,
        label="Remote Read",
        bottom=bottom,
        color=colors["Remote Read"],
    )
    bottom = bottom + remote_read_mb2 * vol_scale_nl
    ax4.bar(
        x,
        local_write_mb2 * vol_scale_nl,
        width_stack,
        label="Local Write",
        bottom=bottom,
        color=colors["Local Write"],
    )
    bottom = bottom + local_write_mb2 * vol_scale_nl
    ax4.bar(
        x,
        remote_write_mb2 * vol_scale_nl,
        width_stack,
        label="Remote Write",
        bottom=bottom,
        color=colors["Remote Write"],
    )

    ax4.set_xlabel("Workflow Construction")
    ax4.set_ylabel(f"Total Data Volume ({vol_unit_nl})")
    ax4.set_title("Total Workflow Data Volume Analysis")
    ax4.set_xticks(x)
    ax4.set_xticklabels(wc_xticks, rotation=0, ha="center")
    ax4.legend(**_legend_kwargs())
    _style_stacked_total_data_volume_axis(ax4)

    fname_exc = "io_patterns_comparison_nonlocal.png"
    fig2.savefig(os.path.join(output_dir, fname_exc))
    plt.close(fig2)
    print(f"  => I/O patterns (non-local) saved to {output_dir}/{fname_exc}")


def plot_resource_utilization(all_simulation_data: List[Dict],
                              sim_groups: List[Dict],
                              jobs: List[Dict],
                              output_dir: str = "plots",
                              custom_labels: List[str] = None):
    """Create resource utilization analysis plots for workflow comparison.

    Writes two PNGs:

    1. ``resource_utilization_comparison.png`` — network, memory, and CPU utilization
       bar charts in one column (top → bottom), shared x-axis, same width as I/O comparison
       plots.
    2. ``resource_cost_comparison.png`` — total CPU cores and total memory (GB), dual y-axis.
    """
    print(f"==> Creating resource utilization analysis for {len(all_simulation_data)} workflows")

    # Extract metrics for comparison
    num_groups = []
    cpu_utilization = []
    memory_utilization = []
    network_transfer = []
    total_cpu_cores = []
    total_memory_mb = []
    events_per_cpu_core = []
    construction_metrics = []  # Build this for text output

    for i, sim_data in enumerate(all_simulation_data):
        file_name = sim_data.get('_file_name', f'simulation_{i}')
        sim_groups_for_file = [g for g in sim_groups if g.get('file_name') == file_name]

        # Extract basic metrics
        num_groups.append(len(sim_groups_for_file))
        cpu_utilization.append(sim_data.get('cpu_utilization', 0.0))
        memory_utilization.append(sim_data.get('memory_occupancy', 0.0))

        # Network transfer calculation
        transfer = sim_data.get("network_transfer_mb_per_event")
        if transfer is None:
            transfer = sim_data.get("total_read_remote_mb_per_event", 0.0) + sim_data.get("total_write_remote_mb_per_event", 0.0)
        network_transfer.append(transfer)

        # Resource cost analysis
        total_cpu = sim_data.get('total_cpu_cores_used', 0.0)
        total_memory = sim_data.get('total_memory_used_mb', 0.0)
        total_events = sim_data.get('total_events_processed', 0.0)

        total_cpu_cores.append(total_cpu)
        total_memory_mb.append(total_memory)

        # Calculate events per CPU core (efficiency metric)
        if total_cpu > 0:
            events_per_cpu_core.append(total_events / total_cpu)
        else:
            events_per_cpu_core.append(0.0)

        # Build construction metrics for text output
        construction_metric = {
            'groups': [g['group_id'] for g in sim_groups_for_file],
            'num_groups': len(sim_groups_for_file),
            'cpu_utilization': sim_data.get('cpu_utilization', 0.0),
            'memory_occupancy': sim_data.get('memory_occupancy', 0.0),
            'network_transfer_per_event_mb': transfer,
            'total_cpu_cores_used': total_cpu,
            'total_memory_used_mb': total_memory,
            'events_per_cpu_core': total_events / total_cpu if total_cpu > 0 else 0.0,
        }
        construction_metrics.append(construction_metric)

    # Convert lists to numpy arrays for numerical operations
    num_groups = np.array(num_groups)
    cpu_utilization = np.array(cpu_utilization)
    memory_utilization = np.array(memory_utilization)
    network_transfer = np.array(network_transfer)
    total_cpu_cores = np.array(total_cpu_cores)
    total_memory_mb = np.array(total_memory_mb)
    events_per_cpu_core = np.array(events_per_cpu_core)

    n_plot = len(all_simulation_data)
    wc_xticks = [str(i + 1) for i in range(n_plot)]
    x = np.arange(n_plot)

    # --- 1) Network, memory, CPU utilization (3×1, shared x) ---
    fig_u, (ax_n, ax_m, ax_c) = plt.subplots(
        3,
        1,
        figsize=(STACKED_COMPARISON_FIG_W_IN, RESOURCE_UTIL_STACK_FIG_H_IN),
        layout="constrained",
        sharex=True,
    )

    ax_n.bar(x, network_transfer, color="#9467bd", alpha=0.7)
    ax_n.set_ylabel("Network Transfer per Event (MB)")
    ax_n.set_title("Network Transfer Analysis")
    ax_n.grid(True, alpha=0.3)
    ax_n.tick_params(axis="x", labelbottom=False)

    ax_m.bar(x, memory_utilization, color="#ff7f0e", alpha=0.7)
    ax_m.set_ylabel("Memory Utilization Ratio")
    ax_m.set_title("Memory Utilization Analysis")
    ax_m.set_ylim(bottom=0.0)
    ax_m.grid(True, alpha=0.3)
    ax_m.tick_params(axis="x", labelbottom=False)

    ax_c.bar(x, cpu_utilization, color="#8c564b", alpha=0.7)
    ax_c.set_xlabel("Workflow Construction")
    ax_c.set_ylabel("CPU Utilization Ratio")
    ax_c.set_title("CPU Utilization Analysis")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(wc_xticks, rotation=0, ha="center")
    ax_c.set_ylim(bottom=0.0)
    ax_c.grid(True, alpha=0.3)

    fname_u = "resource_utilization_comparison.png"
    fig_u.savefig(os.path.join(output_dir, fname_u))
    plt.close(fig_u)
    print(f"  => Resource utilization comparison saved to {output_dir}/{fname_u}")

    # --- 2) Resource cost (CPU cores + memory GB) ---
    fig_cost, ax_cost = plt.subplots(
        1,
        1,
        figsize=(STACKED_COMPARISON_FIG_W_IN, RESOURCE_COST_FIG_H_IN),
        layout="constrained",
    )
    width = 0.35
    total_memory_gb = total_memory_mb / 1024.0
    ax_cost_twin = ax_cost.twinx()

    ax_cost.bar(x - width / 2, total_cpu_cores, width, label="Total CPU Cores",
                color="#8c564b", alpha=0.7)
    ax_cost.set_xlabel("Workflow Construction")
    ax_cost.set_ylabel("Total CPU Cores Used", color="#8c564b")
    ax_cost.tick_params(axis="y", labelcolor="#8c564b")

    ax_cost_twin.bar(x + width / 2, total_memory_gb, width, label="Total Memory (GB)",
                     color="#ff7f0e", alpha=0.7)
    ax_cost_twin.set_ylabel("Total Memory Used (GB)", color="#ff7f0e")
    ax_cost_twin.tick_params(axis="y", labelcolor="#ff7f0e")

    ax_cost.set_title("Overall Resource Cost Analysis")
    ax_cost.set_xticks(x)
    ax_cost.set_xticklabels(wc_xticks, rotation=0, ha="center")
    ax_cost.grid(True, alpha=0.3)

    lines1, labels1 = ax_cost.get_legend_handles_labels()
    lines2, labels2 = ax_cost_twin.get_legend_handles_labels()
    ax_cost.legend(lines1 + lines2, labels1 + labels2, **_legend_kwargs())

    fname_cost = "resource_cost_comparison.png"
    fig_cost.savefig(os.path.join(output_dir, fname_cost))
    plt.close(fig_cost)
    print(f"  => Resource cost comparison saved to {output_dir}/{fname_cost}")


def plot_performance_metrics(all_simulation_data: List[Dict],
                             sim_groups: List[Dict],
                             jobs: List[Dict],
                             output_dir: str = "plots",
                             custom_labels: List[str] = None):
    """Write two performance figures: processing efficiency (wide) and scatter (narrow, tight axes).

    Outputs:

    1. ``processing_efficiency_comparison.png`` — CPU time per event + utilization
       (same width as I/O comparison plots, half stacked height).
    2. ``performance_vs_remote_write_comparison.png`` — scatter with **one point per workflow
       construction** (workflow-level ``event_throughput`` vs ``total_write_remote_mb_per_event``;
       not per-group). **Narrow** figure with **tight** x/y limits around the data.
    """
    print(f"==> Creating performance metrics analysis for {len(all_simulation_data)} workflows")

    # Extract metrics for comparison
    event_throughputs = []
    write_remote_pevt = []
    cpu_time_per_event = []
    cpu_utilization = []
    construction_metrics = []  # Build this for text output

    for i, sim_data in enumerate(all_simulation_data):
        file_name = sim_data.get('_file_name', f'simulation_{i}')
        sim_groups_for_file = [g for g in sim_groups if g.get('file_name') == file_name]

        # Extract basic metrics
        event_throughputs.append(sim_data.get('event_throughput', 0.0))
        write_remote_pevt.append(sim_data.get('total_write_remote_mb_per_event', 0.0))
        cpu_time_per_event.append(sim_data.get('cpu_time_per_event', 0.0))
        cpu_utilization.append(sim_data.get('cpu_utilization', 0.0))

        # Build construction metrics for text output
        construction_metric = {
            'groups': [g['group_id'] for g in sim_groups_for_file],
            'num_groups': len(sim_groups_for_file),
            'event_throughput': sim_data.get('event_throughput', 0.0),
            'write_remote_per_event_mb': sim_data.get('total_write_remote_mb_per_event', 0.0),
            'cpu_time_per_event': sim_data.get('cpu_time_per_event', 0.0),
            'cpu_utilization': sim_data.get('cpu_utilization', 0.0),
        }
        construction_metrics.append(construction_metric)

    # Convert lists to numpy arrays for numerical operations
    event_throughputs = np.array(event_throughputs)
    write_remote_pevt = np.array(write_remote_pevt)
    cpu_time_per_event = np.array(cpu_time_per_event)
    cpu_utilization = np.array(cpu_utilization)

    n_plot = len(all_simulation_data)
    wc_xticks = [str(i + 1) for i in range(n_plot)]

    # --- 1) Processing efficiency (wide single panel) ---
    fig_p, ax_p = plt.subplots(
        1,
        1,
        figsize=(STACKED_COMPARISON_FIG_W_IN, PROCESSING_EFFICIENCY_FIG_H_IN),
        layout="constrained",
    )
    x = np.arange(n_plot)
    width = 0.6
    ax_p_twin = ax_p.twinx()
    ax_p.bar(x, cpu_time_per_event, width, label="CPU Time per Event", color="#2ca02c", alpha=0.7)
    ax_p.set_xlabel("Workflow Construction")
    ax_p.set_ylabel("CPU Time per Event (seconds)", color="#2ca02c")
    ax_p.tick_params(axis="y", labelcolor="#2ca02c")
    ax_p_twin.plot(x, cpu_utilization, "o-", color="#d62728", linewidth=1.5, markersize=4,
                   label="CPU Utilization")
    ax_p_twin.set_ylabel("CPU Utilization Ratio", color="#d62728")
    ax_p_twin.tick_params(axis="y", labelcolor="#d62728")
    ax_p_twin.set_ylim(0, 1)

    _cpu = np.asarray(cpu_time_per_event, dtype=float)
    _cpu = _cpu[np.isfinite(_cpu)]
    if _cpu.size > 0:
        lo = float(np.min(_cpu))
        hi = float(np.max(_cpu))
        span = hi - lo
        if span <= 0.0 or not np.isfinite(span):
            mag = abs(hi) if hi != 0.0 else 1.0
            span = max(mag * 0.05, 1e-12)
        pad = max(span * 0.10, hi * 0.02, 1e-12)
        bottom = max(0.0, lo - pad) if lo >= 0.0 else lo - pad
        top = hi + pad
        if bottom >= top:
            top = bottom + max(span, 1e-9)
        ax_p.set_ylim(bottom, top)

    ax_p.set_title("Processing Efficiency Analysis")
    ax_p.set_xticks(x)
    ax_p.set_xticklabels(wc_xticks, rotation=0, ha="center")
    ax_p.grid(True, alpha=0.3)
    lines1, labels1 = ax_p.get_legend_handles_labels()
    lines2, labels2 = ax_p_twin.get_legend_handles_labels()
    ax_p.legend(lines1 + lines2, labels1 + labels2, **_legend_kwargs(loc="upper left"))

    fname_p = "processing_efficiency_comparison.png"
    fig_p.savefig(os.path.join(output_dir, fname_p))
    plt.close(fig_p)
    print(f"  => Processing efficiency plot saved to {output_dir}/{fname_p}")

    # --- 2) Throughput vs remote write (narrow, tight axis limits) ---
    fig_s, ax_s = plt.subplots(
        1,
        1,
        figsize=(PERF_SCATTER_FIG_W_IN, PERF_SCATTER_FIG_H_IN),
        layout="constrained",
    )
    ax_s.scatter(
        event_throughputs,
        write_remote_pevt,
        c=range(len(all_simulation_data)),
        cmap="viridis",
        s=100,
        alpha=0.7,
    )
    ax_s.set_xlabel("Event Throughput (events/second)")
    ax_s.set_ylabel("Remote Write Data per Event (MB)")
    ax_s.set_title("Performance vs Remote Write Efficiency\n", fontsize=10,
    )
    ax_s.grid(True, alpha=0.3)

    x_lo, x_hi = _tight_axis_limits(event_throughputs, clamp_non_negative=False)
    y_lo, y_hi = _tight_axis_limits(write_remote_pevt, clamp_non_negative=True)
    ax_s.set_xlim(x_lo, x_hi)
    ax_s.set_ylim(y_lo, y_hi)

    fname_s = "performance_vs_remote_write_comparison.png"
    fig_s.savefig(os.path.join(output_dir, fname_s))
    plt.close(fig_s)
    print(f"  => Performance vs remote write plot saved to {output_dir}/{fname_s}")


def plot_turnaround_time_comparison(all_simulation_data: List[Dict],
                                    sim_groups: List[Dict],
                                    jobs: List[Dict],
                                    output_dir: str = "plots",
                                    custom_labels: List[str] = None):
    """Create a vertical bar chart of workflow turnaround time per composition.

    Uses ``total_turnaround_time`` from simulation metrics (seconds) and plots
    values in hours on the y-axis. Compositions are ordered by
    ``composition_number``.
    """
    print(f"==> Creating turnaround time comparison for {len(all_simulation_data)} workflows")

    rows: List[tuple] = []
    for idx, sim_data in enumerate(all_simulation_data):
        comp = sim_data.get("composition_number", idx + 1)
        tt = sim_data.get("total_turnaround_time")
        if tt is None:
            tt = 0.0
        rows.append((comp, float(tt), idx))

    rows.sort(key=lambda r: r[0])
    turnaround_hours = np.array([r[1] / 3600.0 for r in rows])
    x = np.arange(len(rows))
    wc_xticks = [str(i + 1) for i in range(len(rows))]

    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.65
    ax.bar(x, turnaround_hours, width, color="#17becf", alpha=0.85)
    ax.set_xlabel("Workflow Construction")
    ax.set_ylabel("Turnaround Time (hours)")
    ax.set_title("Workflow Turnaround Time by Composition")
    ax.set_xticks(x)
    ax.set_xticklabels(wc_xticks, rotation=0, ha="center")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    ymax = float(np.max(turnaround_hours)) if len(turnaround_hours) else 0.0
    if ymax > 0:
        pad = ymax * 0.05
        ax.set_ylim(top=ymax + pad)

    plt.tight_layout()
    filename = "turnaround_time_comparison.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"  => Turnaround time comparison saved to {output_dir}/{filename}")


def plot_workflow_comparison(all_simulation_data: List[Dict],
                             sim_groups: List[Dict],
                             jobs: List[Dict],
                             output_dir: str = "plots",
                             custom_labels: List[str] = None,
                             use_aggregate_metrics: bool = False):
    """Create a comprehensive comparison of workflow constructions.

    This function creates multiple visualizations to help identify trade-offs
    between different workflow constructions.
    """
    print(f"Creating comprehensive workflow construction comparison for {len(all_simulation_data)} constructions")

    # Extract metrics for comparison directly from simulation data
    num_groups = []
    event_throughputs = []
    total_cpu_times = []
    write_remote_pevt = []
    total_write_remote = []
    read_remote_pevt = []
    write_local_pevt = []
    read_local_pevt = []
    group_combinations = []
    construction_metrics = []  # Build this for text output

    for i, sim_data in enumerate(all_simulation_data):
        # sim_data now contains metrics directly (not nested under 'metrics')
        file_name = sim_data.get('_file_name', f'simulation_{i}')

        # Get groups for this simulation from the groups parameter
        sim_groups = [g for g in groups if g.get('file_name') == file_name]

        # Extract basic metrics
        num_groups.append(len(sim_groups))
        event_throughputs.append(sim_data.get('event_throughput', 0.0))
        total_cpu_times.append(sim_data.get('total_cpu_allocated_time', 0.0))
        write_remote_pevt.append(sim_data.get('total_write_remote_mb_per_event', 0.0))
        total_write_remote.append(sim_data.get('total_write_remote_mb', 0.0))
        read_remote_pevt.append(sim_data.get('total_read_remote_mb_per_event', 0.0))
        write_local_pevt.append(sim_data.get('total_write_local_mb_per_event', 0.0))
        read_local_pevt.append(sim_data.get('total_read_local_mb_per_event', 0.0))

        # Build group combinations
        group_ids = [g['group_id'] for g in sim_groups]
        group_combinations.append(" + ".join(group_ids))

        # Build construction metrics for text output
        construction_metric = {
            'groups': group_ids,
            'num_groups': len(sim_groups),
            'event_throughput': sim_data.get('event_throughput', 0.0),
            'total_cpu_time': sim_data.get('total_cpu_allocated_time', 0.0),
            'write_remote_per_event_mb': sim_data.get('total_write_remote_mb_per_event', 0.0),
            'total_write_remote_mb': sim_data.get('total_write_remote_mb', 0.0),
            'read_remote_per_event_mb': sim_data.get('total_read_remote_mb_per_event', 0.0),
            'write_local_per_event_mb': sim_data.get('total_write_local_mb_per_event', 0.0),
            'read_local_per_event_mb': sim_data.get('total_read_local_mb_per_event', 0.0),
            'total_read_remote_mb': sim_data.get('total_read_remote_mb', 0.0),
            'total_write_local_mb': sim_data.get('total_write_local_mb', 0.0),
            'total_wallclock_time': sim_data.get('total_wall_time', 0.0),
            'total_memory_mb': sim_data.get('total_memory_used_mb', 0.0),
            'total_network_transfer_mb': sim_data.get('total_network_transfer_mb', 0.0),
            'network_transfer_per_event_mb': sim_data.get('network_transfer_mb_per_event', 0.0),
            'cpu_utilization': sim_data.get('cpu_utilization', 0.0),
            'memory_occupancy': sim_data.get('memory_occupancy', 0.0),
            'group_details': []  # Will be populated below
        }

        # Build group_details for text output
        for group in sim_groups:
            # Get taskset IDs for this group (from the extracted group metrics)
            tasks = group.get('group_taskset_count', 0)  # This is the count, not the actual IDs

            # Calculate events per task (using first job of this group)
            group_jobs = [j for j in jobs if j['group_id'] == group['group_id'] and j.get('file_name') == file_name]
            events_per_task = group_jobs[0]['batch_size'] if group_jobs else group.get('group_input_events', 0)

            # Calculate per-event data metrics from first job
            first_job = group_jobs[0] if group_jobs else None
            total_events = group.get('group_input_events', 0) * group.get('group_job_count', 1)

            read_local_per_event = 0.0
            read_remote_per_event = 0.0
            write_local_per_event = 0.0
            write_remote_per_event = 0.0

            if first_job and total_events > 0:
                read_local_per_event = first_job.get('total_read_local_mb', 0.0) / events_per_task
                read_remote_per_event = first_job.get('total_read_remote_mb', 0.0) / events_per_task
                write_local_per_event = first_job.get('total_write_local_mb', 0.0) / events_per_task
                write_remote_per_event = first_job.get('total_write_remote_mb', 0.0) / events_per_task

            # Calculate CPU seconds for the group (using extracted metrics)
            cpu_seconds = group.get('group_time_per_event', 0.0) * events_per_task

            construction_metric['group_details'].append({
                'group_id': group['group_id'],
                'tasks': [f"taskset_{i+1}" for i in range(tasks)],  # Generate task IDs
                'events_per_task': events_per_task,
                'cpu_seconds': cpu_seconds,
                'read_local_per_event_mb': read_local_per_event,
                'read_remote_per_event_mb': read_remote_per_event,
                'write_local_per_event_mb': write_local_per_event,
                'write_remote_per_event_mb': write_remote_per_event,
                'total_events': total_events
            })

        construction_metrics.append(construction_metric)

    # Convert lists to numpy arrays for numerical operations
    num_groups = np.array(num_groups)
    event_throughputs = np.array(event_throughputs)
    total_cpu_times = np.array(total_cpu_times)
    write_remote_pevt = np.array(write_remote_pevt)
    total_write_remote = np.array(total_write_remote)
    read_remote_pevt = np.array(read_remote_pevt)
    write_local_pevt = np.array(write_local_pevt)
    read_local_pevt = np.array(read_local_pevt)

    # Create a figure with multiple subplots - now with fixed, professional proportions
    if len(construction_metrics) <= 2:
        fig = plt.figure(figsize=(12, 16))
    else:
        fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1])  # Equal height ratios for all rows
    n_plot = len(all_simulation_data)
    wc_xticks = [str(i + 1) for i in range(n_plot)]

    # Define consistent colors for each metric type
    colors = {
        'Local Read': '#1f77b4',    # Blue
        'Remote Read': '#ff7f0e',   # Orange
        'Local Write': '#2ca02c',   # Green
        'Remote Write': '#d62728'   # Red
    }

    # 1. Data Volume Analysis Per Event (with Local Read)
    ax3 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(all_simulation_data))
    width = 0.2
    ax3.bar(x - 1.5*width, read_local_pevt, width, label='Local Read', color=colors['Local Read'])
    ax3.bar(x - 0.5*width, read_remote_pevt, width, label='Remote Read', color=colors['Remote Read'])
    ax3.bar(x + 0.5*width, write_local_pevt, width, label='Local Write', color=colors['Local Write'])
    ax3.bar(x + 1.5*width, write_remote_pevt, width, label='Remote Write', color=colors['Remote Write'])
    ax3.set_xlabel("Workflow Construction")
    ax3.set_ylabel("Data Volume per Event (MB)")
    ax3.set_title("Data Volume Analysis Per Event")
    ax3.set_xticks(x)
    ax3.set_xticklabels(wc_xticks, rotation=0, ha="center")
    ax3.legend(**_legend_kwargs())
    ax3.grid(True)

    # 2. Data Flow Analysis (Updated to use per-event metrics)
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(all_simulation_data))
    width = 0.25
    ax2.bar(x - width, read_remote_pevt, width, label='Remote Read', color=colors['Remote Read'])
    ax2.bar(x, write_local_pevt, width, label='Local Write', color=colors['Local Write'])
    ax2.bar(x + width, write_remote_pevt, width, label='Remote Write', color=colors['Remote Write'])
    ax2.set_xlabel("Workflow Construction")
    ax2.set_ylabel("Data Volume per Event (MB)")
    ax2.set_title("Data Volume Analysis Per Event")
    ax2.set_xticks(x)
    ax2.set_xticklabels(wc_xticks, rotation=0, ha="center")
    ax2.legend(**_legend_kwargs())
    ax2.grid(True)

    # 3. Total Data Volume Analysis (Stacked Bar)
    ax10 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(all_simulation_data))
    width = 0.6
    bottom = np.zeros(len(all_simulation_data))

    # Convert MB to GB for better readability
    remote_read_gb = [m["total_read_remote_mb"] / 1024.0 for m in construction_metrics]
    local_write_gb = [m["total_write_local_mb"] / 1024.0 for m in construction_metrics]
    remote_write_gb = [m["total_write_remote_mb"] / 1024.0 for m in construction_metrics]

    # Plot each data type as a layer in the stack
    ax10.bar(x, remote_read_gb, width, label='Remote Read', bottom=bottom)
    bottom += remote_read_gb

    ax10.bar(x, local_write_gb, width, label='Local Write', bottom=bottom)
    bottom += local_write_gb

    ax10.bar(x, remote_write_gb, width, label='Remote Write', bottom=bottom)

    # Add total value labels on top of each bar
    totals_gb = [rr + lw + rw for rr, lw, rw in zip(remote_read_gb, local_write_gb, remote_write_gb)]
    for i, total in enumerate(totals_gb):
        ax10.text(i, total, f'{total:.1f}', ha='center', va='bottom')

    ax10.set_xlabel("Workflow Construction")
    ax10.set_ylabel("Total Data Volume (GB)")
    ax10.set_title("Total Workflow Data Volume Analysis")
    ax10.set_xticks(x)
    ax10.set_xticklabels(wc_xticks, rotation=0, ha="center")
    ax10.legend(**_legend_kwargs())
    ax10.grid(True)

    # 4. Performance vs Remote Write Efficiency (simplified scatter plot)
    ax1 = fig.add_subplot(gs[1, 1])

    # Create a simple scatter plot
    scatter = ax1.scatter(event_throughputs, write_remote_pevt,
                         c=num_groups, cmap='viridis', s=100, alpha=0.7)

    # Add colorbar with discrete integer values
    cbar = plt.colorbar(scatter, ax=ax1, label="Number of Groups")

    # Get unique values and set discrete ticks
    unique_groups = np.unique(num_groups)
    cbar.set_ticks(unique_groups)
    cbar.set_ticklabels([f"{int(x)}" for x in unique_groups])

    ax1.set_xlabel("Event Throughput (events/second)")
    ax1.set_ylabel("Remote Write Data per Event (MB)")
    ax1.set_title("Performance vs Remote Write Efficiency")
    ax1.grid(True, alpha=0.3)

    # set x-axis to start at 0 and add 10% padding to the right
    max_throughput = np.max(event_throughputs)
    if max_throughput > 0:
        ax1.set_xlim(left=0, right=max_throughput * 1.1)
    else:
        # If all throughputs are 0, set a small range to avoid the warning
        ax1.set_xlim(left=0, right=1.0)
    # set y-axis to start at 0
    ax1.set_ylim(bottom=0)

    # 5. Network Transfer Analysis
    ax7 = fig.add_subplot(gs[2, 0])
    network_transfer = []
    for sim_data in all_simulation_data:
        # sim_data now contains metrics directly
        # Network transfer = remote read + remote write (only remote operations)
        # Use pre-calculated field if available, otherwise calculate from per-event metrics
        transfer = sim_data.get("network_transfer_mb_per_event")
        if transfer is None:
            # Fallback: calculate from read and write remote per event
            transfer = sim_data.get("total_read_remote_mb_per_event", 0.0) + sim_data.get("total_write_remote_mb_per_event", 0.0)
        network_transfer.append(transfer)

    ax7.bar(range(len(all_simulation_data)), network_transfer)
    ax7.set_xlabel("Workflow Construction")
    ax7.set_ylabel("Network Transfer per Event (MB)")
    ax7.set_title("Network Transfer Analysis")
    ax7.set_xticks(range(len(all_simulation_data)))
    ax7.set_xticklabels(wc_xticks, rotation=0, ha="center")
    ax7.grid(True)

    # 6. CPU Utilization Analysis
    ax4 = fig.add_subplot(gs[2, 1])
    cpu_utilization = []
    cpu_std = []  # Store standard deviations

    # Use metrics data from simulation
    for sim_data in all_simulation_data:
        # sim_data now contains metrics directly
        cpu_util = sim_data.get("cpu_utilization", 0.0)
        cpu_utilization.append(cpu_util)
        cpu_std.append(0)  # No std dev available from aggregated metrics

    # Create bar plot with error bars (only if we have std data)
    x = range(len(all_simulation_data))
    if any(std > 0 for std in cpu_std):
        ax4.bar(x, cpu_utilization, yerr=cpu_std, capsize=5)
    else:
        ax4.bar(x, cpu_utilization)
    ax4.set_xlabel("Workflow Construction")
    ax4.set_ylabel("CPU Utilization Ratio")
    ax4.set_title("CPU Utilization Analysis\n(Average CPU Usage / Allocated CPU ± Std Dev)")
    ax4.set_xticks(x)
    ax4.set_xticklabels(wc_xticks, rotation=0, ha="center")
    ax4.set_ylim(bottom=0)  # Set Y-axis to start at 0
    ax4.grid(True)

    # 7. Memory Utilization Analysis
    ax6 = fig.add_subplot(gs[3, 0])
    memory_utilization = []
    memory_std = []  # Store standard deviations

    # Use metrics data from simulation
    for sim_data in all_simulation_data:
        # sim_data now contains metrics directly
        mem_util = sim_data.get("memory_occupancy", 0.0)
        memory_utilization.append(mem_util)
        memory_std.append(0)  # No std dev available from aggregated metrics

    # Create bar plot with error bars (only if we have std data)
    x = range(len(all_simulation_data))
    if any(std > 0 for std in memory_std):
        ax6.bar(x, memory_utilization, yerr=memory_std, capsize=5)
    else:
        ax6.bar(x, memory_utilization)
    ax6.set_xlabel("Workflow Construction")
    ax6.set_ylabel("Memory Utilization Ratio")
    ax6.set_title("Memory Utilization Analysis\n(Average Memory Occupancy ± Std Dev)")
    ax6.set_xticks(x)
    ax6.set_xticklabels(wc_xticks, rotation=0, ha="center")
    ax6.grid(True)

    # 8. Event Processing Analysis
    ax5 = fig.add_subplot(gs[3, 1])
    events_per_group = []
    for i, sim_data in enumerate(all_simulation_data):
        file_name = sim_data.get('_file_name', f'simulation_{i}')
        # Get groups for this simulation from the groups parameter
        sim_groups = [g for g in groups if g.get('file_name') == file_name]
        events = [g.get('group_input_events', 0) * g.get('group_job_count', 1) for g in sim_groups]
        events_per_group.append(events)

    ax5.boxplot(events_per_group, tick_labels=wc_xticks)
    ax5.set_xlabel("Workflow Construction")
    ax5.set_ylabel("Events per Group")
    ax5.set_title("Event Processing Distribution")
    ax5.set_xticklabels(wc_xticks, rotation=0, ha="center")
    ax5.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "workflow_comparison.png"))
    plt.close()

    # Create a detailed comparison table
    with open(os.path.join(output_dir, "workflow_comparison.txt"), "w") as f:
        f.write("Workflow Construction Comparison\n")
        f.write("==============================\n\n")

        for i, metrics in enumerate(construction_metrics, 1):
            # Use custom label if provided, otherwise use default "Construction" label
            if custom_labels and i <= len(custom_labels):
                construction_label = custom_labels[i-1]
            else:
                construction_label = f"Construction {i}"
            f.write(f"{construction_label}:\n")
            f.write(f"  Groups: {metrics['groups']}\n")
            f.write(f"  Number of Groups: {metrics['num_groups']}\n")
            f.write(f"  Event Throughput: {metrics['event_throughput']:.4f} events/second\n")
            f.write(f"  Total CPU Time: {metrics['total_cpu_time']:.2f} seconds\n")
            f.write("  Total Data Volumes for one job of each group:\n")
            f.write(f"    Remote Read Data: {metrics['total_read_remote_mb']:.2f} MB\n")
            f.write(f"    Local Write Data: {metrics['total_write_local_mb']:.2f} MB\n")
            f.write(f"    Remote Write Data: {metrics['total_write_remote_mb']:.2f} MB\n")
            f.write("  Data Flow Metrics (per event):\n")
            f.write(f"    Local Read Data: {metrics['read_local_per_event_mb']:.3f} MB/event\n")
            f.write(f"    Remote Read Data: {metrics['read_remote_per_event_mb']:.3f} MB/event\n")
            f.write(f"    Local Write Data: {metrics['write_local_per_event_mb']:.3f} MB/event\n")
            f.write(f"    Remote Write Data: {metrics['write_remote_per_event_mb']:.3f} MB/event\n")
            if i <= len(memory_utilization):
                f.write(f"  Memory Utilization: {memory_utilization[i-1]:.2f}\n")
            if i <= len(network_transfer):
                f.write(f"  Network Transfer: {network_transfer[i-1]:.2f} MB\n")
            f.write("  Workflow Performance Metrics:\n")
            f.write(f"    Total CPU Time: {metrics['total_cpu_time']:.2f} seconds\n")
            f.write(f"    Total Wallclock Time: {metrics['total_wallclock_time']:.2f} seconds\n")
            f.write(f"    Total Memory: {metrics['total_memory_mb']:,.0f} MB\n")
            f.write(f"    Total Network Transfer: {metrics['total_network_transfer_mb']:,.0f} MB\n")
            f.write("  Group Details:\n")
            for group in metrics["group_details"]:
                f.write(f"    {group['group_id']}:\n")
                f.write(f"      Tasks: {group['tasks']}\n")
                f.write(f"      Events per Task: {group['events_per_task']}\n")
                f.write(f"      CPU Time: {group['cpu_seconds']:.2f} seconds\n")
                f.write("      Data Flow (per event):\n")
                f.write(f"        Local Read: {group['read_local_per_event_mb']:.3f} MB/event\n")
                f.write(f"        Remote Read: {group['read_remote_per_event_mb']:.3f} MB/event\n")
                f.write(f"        Local Write: {group['write_local_per_event_mb']:.3f} MB/event\n")
                f.write(f"        Remote Write: {group['write_remote_per_event_mb']:.3f} MB/event\n")
            f.write("\n")


def find_simulation_files(directory_path: str) -> List[str]:
    """Find all JSON simulation files in a directory.

    This function recursively searches for JSON files, including in subdirectories
    with wallclock time suffixes (e.g., seq_real_12h, seq_real_24h).
    """
    directory = Path(directory_path)
    if not directory.exists():
        raise FileNotFoundError(f"Directory '{directory_path}' not found")

    if not directory.is_dir():
        raise NotADirectoryError(f"'{directory_path}' is not a directory")

    # Find all JSON files recursively (handles directories with wallclock time suffixes)
    json_files = list(directory.glob("**/*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in directory '{directory_path}'")

    # Convert to strings and sort for consistent ordering
    return sorted([str(f) for f in json_files])


def extract_group_metrics(simulation_data: Dict[str, Any], file_name: str) -> List[Dict[str, Any]]:
    """Extract group-level metrics from simulation data."""
    groups = []

    for group in simulation_data.get('simulation_result', {}).get('groups', []):
        group_metrics = {
            'group_id': group['group_id'],
            'group_job_count': group['job_count'],
            'group_input_events': group['input_events'],
            'group_total_execution_time': group['total_execution_time'],
            'group_exact_job_count': group['exact_job_count'],
            'group_taskset_count': len(group['tasksets']),
            'group_dependencies': group.get('dependencies', []),
            'file_name': file_name,
            'composition_number': simulation_data.get('metrics', {}).get('composition_number', 0)
        }

        # Calculate aggregated metrics from tasksets
        group_size_per_event = sum(ts['size_per_event'] for ts in group['tasksets'])
        group_time_per_event = sum(ts['time_per_event'] for ts in group['tasksets'])
        group_memory = sum(ts['memory'] for ts in group['tasksets'])
        group_memory_max = max(ts['memory'] for ts in group['tasksets'])
        group_multicore = sum(ts['multicore'] for ts in group['tasksets'])
        group_multicore_max = max(ts['multicore'] for ts in group['tasksets'])

        group_metrics.update({
            'group_size_per_event': group_size_per_event,
            'group_time_per_event': group_time_per_event,
            'group_memory': group_memory,
            'group_memory_max': group_memory_max,
            'group_multicore': group_multicore,
            'group_multicore_max': group_multicore_max,
            'group_time_per_event_avg': group_time_per_event / len(group['tasksets']),
            'group_memory_avg': group_memory / len(group['tasksets']),
            'group_multicore_avg': group_multicore / len(group['tasksets']),
            'group_size_per_event_avg': group_size_per_event / len(group['tasksets'])
        })

        groups.append(group_metrics)

    return groups


def extract_job_metrics(simulation_data: Dict[str, Any], file_name: str) -> List[Dict[str, Any]]:
    """Extract job-level metrics from simulation data.

    Only extracts metrics for the first job of each group to minimize memory usage
    while still allowing analysis of group behavior. Aggregated data is available
    at the metrics level.
    """
    jobs = []
    processed_groups = set()

    for job in simulation_data.get('simulation_result', {}).get('jobs', []):
        group_id = job['group_id']

        # Only process the first job of each group
        if group_id in processed_groups:
            continue

        processed_groups.add(group_id)

        job_metrics = {
            'job_id': job['job_id'],
            'group_id': group_id,
            'batch_size': job['batch_size'],
            'wallclock_time': job['wallclock_time'],
            'start_time': job['start_time'],
            'end_time': job['end_time'],
            'status': job['status'],
            'total_cpu_used_time': job['total_cpu_used_time'],
            'total_cpu_allocated_time': job['total_cpu_allocated_time'],
            'total_write_local_mb': job['total_write_local_mb'],
            'total_write_remote_mb': job['total_write_remote_mb'],
            'total_read_local_mb': job['total_read_local_mb'],
            'total_read_remote_mb': job['total_read_remote_mb'],
            'total_network_transfer_mb': job['total_network_transfer_mb'],
            'file_name': file_name,
            'composition_number': simulation_data.get('metrics', {}).get('composition_number', 0)
        }

        # Calculate derived metrics
        cpu_utilization = (job['total_cpu_used_time'] / job['total_cpu_allocated_time']
                          if job['total_cpu_allocated_time'] > 0 else 0)

        job_metrics.update({
            'cpu_utilization': cpu_utilization,
            'throughput_eps': job['batch_size'] / job['wallclock_time'] if job['wallclock_time'] > 0 else 0,
            'cpu_efficiency': cpu_utilization,
            'data_io_ratio': (job['total_write_local_mb'] + job['total_write_remote_mb']) /
                           (job['total_read_local_mb'] + job['total_read_remote_mb'])
                           if (job['total_read_local_mb'] + job['total_read_remote_mb']) > 0 else 0
        })

        jobs.append(job_metrics)

    return jobs


def process_simulation_directory(directory_path: str, overhead_filter: str = None) -> tuple:
    """Process simulation files in a directory incrementally to minimize memory usage.

    Processes all simulation result JSON files (*.json) in the directory.

    Args:
        directory_path: Path to directory containing simulation files
        overhead_filter: Unused; kept for API compatibility.

    Returns:
        tuple: (all_groups, all_jobs, all_simulation_data) - aggregated metrics and
               all simulation data for comparison plots
    """
    simulation_files = find_simulation_files(directory_path)
    files_to_process = simulation_files
    print(f"Found {len(files_to_process)} simulation files")

    if not files_to_process:
        print("Warning: No simulation JSON files found")
        return [], [], []

    all_groups = []
    all_jobs = []
    all_simulation_data = []
    files_processed = 0

    print(f"Processing {len(files_to_process)} JSON files")

    for file_path in files_to_process:
        try:
            print(f"  Loading and processing: {Path(file_path).name}")

            # Load the file
            with open(file_path, 'r') as f:
                simulation_data = json.load(f)
            file_name = Path(file_path).name

            # Extract metrics immediately (this reduces the memory footprint)
            groups = extract_group_metrics(simulation_data, file_name)
            jobs = extract_job_metrics(simulation_data, file_name)

            # Accumulate the extracted metrics
            all_groups.extend(groups)
            all_jobs.extend(jobs)

            # Keep simulation data for comparison plots
            workflow_metrics = {'_file_name': file_name}
            workflow_metrics.update(simulation_data.get('metrics', {}))
            # Check for overhead_enabled in simulation_result
            sim_result = simulation_data.get('simulation_result', {})
            workflow_metrics['_overhead_enabled'] = sim_result.get('overhead_enabled', True)
            all_simulation_data.append(workflow_metrics)

            files_processed += 1

        except Exception as e:
            print(f"  Warning: Failed to process {Path(file_path).name}: {e}")
            continue

    if files_processed == 0:
        raise ValueError(f"No valid simulation data processed from directory '{directory_path}'")

    print(f"Successfully processed {files_processed} simulation files")
    print(f"Extracted {len(all_groups)} groups and {len(all_jobs)} jobs")

    return all_groups, all_jobs, all_simulation_data


def generate_summary_table(all_simulation_data: List[Dict],
                          sim_groups: List[Dict],
                          output_dir: str) -> pd.DataFrame:
    """Generate a summary table with key metrics for each workflow construction.

    Creates CSV and formatted text versions of the summary table.

    Args:
        all_simulation_data: List of simulation data dictionaries
        sim_groups: List of group metrics dictionaries
        output_dir: Output directory for the table file

    Returns:
        pandas DataFrame containing the summary table
    """
    print(f"==> Creating summary table for {len(all_simulation_data)} workflow constructions")

    # Prepare data for the table
    table_data = []

    for i, sim_data in enumerate(all_simulation_data):
        file_name = sim_data.get('_file_name', f'simulation_{i}')

        # Count total groups for this workflow
        groups_for_file = [g for g in sim_groups if g.get('file_name') == file_name]
        total_groups = len(groups_for_file)

        # Extract metrics
        composition_number = sim_data.get('composition_number', i + 1)
        wall_time_per_event = sim_data.get('wall_time_per_event', 0.0)
        cpu_time_per_event = sim_data.get('cpu_time_per_event', 0.0)
        event_throughput = sim_data.get('event_throughput', 0.0)
        total_write_remote_mb_per_event = sim_data.get('total_write_remote_mb_per_event', 0.0)
        total_read_remote_mb_per_event = sim_data.get('total_read_remote_mb_per_event', 0.0)
        network_transfer_mb_per_event = sim_data.get('network_transfer_mb_per_event', 0.0)

        # Calculate per-event metrics for CPU cores and memory
        total_events = sim_data.get('total_events_processed', 0.0)
        total_cpu_cores_used = sim_data.get('total_cpu_cores_used', 0.0)
        total_memory_used_mb = sim_data.get('total_memory_used_mb', 0.0)

        cpu_cores_per_event = sim_data.get('cpu_cores_per_event', 0.0)
        memory_mb_per_event = sim_data.get('memory_mb_per_event', 0.0)

        # Build row data with readable column names
        row_data = {
            'Comp': composition_number,
            'Groups': total_groups,
            'Wall Time/Evt (s)': wall_time_per_event,
            'CPU Time/Evt (s)': cpu_time_per_event,
            'Throughput (evt/s)': event_throughput,
            'Write Remote (MB/evt)': total_write_remote_mb_per_event,
            'Read Remote (MB/evt)': total_read_remote_mb_per_event,
            'Net Transfer (MB/evt)': network_transfer_mb_per_event,
            'CPU Cores/Evt': cpu_cores_per_event,
            'Memory/Evt (MB)': memory_mb_per_event
        }

        table_data.append(row_data)

    # Create DataFrame
    df = pd.DataFrame(table_data)

    # Sort by composition_number if available
    if 'Comp' in df.columns:
        df = df.sort_values('Comp').reset_index(drop=True)

    # Save as CSV (with original column names for compatibility)
    csv_filename = "workflow_summary_table.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    # Create a version with original column names for CSV
    df_csv = df.copy()
    df_csv.columns = [
        'composition_number', 'total_groups', 'wall_time_per_event',
        'cpu_time_per_event', 'event_throughput',
        'total_write_remote_mb_per_event', 'total_read_remote_mb_per_event',
        'network_transfer_mb_per_event', 'cpu_cores_per_event',
        'memory_mb_per_event'
    ]
    df_csv.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  => Summary table saved to {csv_path}")

    # Save as formatted text file with better formatting
    txt_filename = "workflow_summary_table.txt"
    txt_path = os.path.join(output_dir, txt_filename)
    with open(txt_path, 'w') as f:
        f.write("Workflow Construction Summary Table\n")
        f.write("=" * 120 + "\n\n")

        # Format with appropriate precision for different columns
        formatted_df = df.copy()
        time_cols = ['Wall Time/Evt (s)', 'CPU Time/Evt (s)']
        throughput_cols = ['Throughput (evt/s)']
        mb_cols = ['Write Remote (MB/evt)', 'Read Remote (MB/evt)',
                   'Net Transfer (MB/evt)', 'Memory/Evt (MB)']
        cores_cols = ['CPU Cores/Evt']

        for col in time_cols:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}")
        for col in throughput_cols:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}")
        for col in mb_cols:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}")
        for col in cores_cols:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}")

        # Save current pandas display options
        old_max_columns = pd.get_option('display.max_columns')
        old_width = pd.get_option('display.width')
        old_max_colwidth = pd.get_option('display.max_colwidth')

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        f.write(formatted_df.to_string(index=False))
        f.write("\n\n")

        # Add summary statistics if multiple workflows
        if len(df) > 1:
            f.write("Summary Statistics:\n")
            f.write("-" * 120 + "\n")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            summary = df[numeric_cols].describe()
            f.write(summary.to_string())
            f.write("\n")

        # Restore pandas display options
        pd.set_option('display.max_columns', old_max_columns)
        pd.set_option('display.width', old_width)
        pd.set_option('display.max_colwidth', old_max_colwidth)

    print(f"  => Formatted summary table saved to {txt_path}")

    return df


def generate_workflow_visualizations(all_simulation_data: List[Dict],
                                    sim_groups: List[Dict],
                                    jobs: List[Dict],
                                    output_dir: str) -> None:
    """Generate all workflow comparison visualizations.

    Args:
        all_simulation_data: List of simulation data dictionaries
        sim_groups: List of group metrics dictionaries
        jobs: List of job metrics dictionaries
        output_dir: Output directory for visualization files
    """
    if len(all_simulation_data) > 0:
        print(f"\nGenerating workflow comparison for {len(all_simulation_data)} workflow(s)...")
        try:
            generate_summary_table(
                all_simulation_data=all_simulation_data,
                sim_groups=sim_groups,
                output_dir=output_dir
            )

            if len(all_simulation_data) > 1:
                plot_io_patterns(
                    all_simulation_data=all_simulation_data,
                    sim_groups=sim_groups,
                    jobs=jobs,
                    output_dir=output_dir
                )

                plot_resource_utilization(
                    all_simulation_data=all_simulation_data,
                    sim_groups=sim_groups,
                    jobs=jobs,
                    output_dir=output_dir
                )

                plot_performance_metrics(
                    all_simulation_data=all_simulation_data,
                    sim_groups=sim_groups,
                    jobs=jobs,
                    output_dir=output_dir
                )

                plot_turnaround_time_comparison(
                    all_simulation_data=all_simulation_data,
                    sim_groups=sim_groups,
                    jobs=jobs,
                    output_dir=output_dir
                )
            else:
                print(f"  => Skipping comparison plots (only {len(all_simulation_data)} workflow found)")
        except Exception as e:
            print(f"Warning: Could not generate workflow comparison: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nNo simulation files found, skipping visualizations")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Create visualizations for workflow simulation results using pandas/matplotlib/seaborn'
    )
    parser.add_argument('simulation_directory', type=str,
                       help='Path to directory containing simulation result JSON files')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Base output directory (default: output)')
    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Processing simulation data from directory: {args.simulation_directory}")
    try:
        groups, jobs, all_simulation_data = process_simulation_directory(
            args.simulation_directory
        )

        generate_workflow_visualizations(
            all_simulation_data=all_simulation_data,
            sim_groups=groups,
            jobs=jobs,
            output_dir=args.output_dir
        )

    except Exception as e:
        print(f"Error processing simulation data: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

