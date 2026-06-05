# Workflow Simulation Visualization

This directory contains tooling to visualize workflow simulation results using pandas and
matplotlib. The visualizer reads completed simulation outputs from the `results/sim/` directory and
produces comparison plots across multiple workflow constructions (JSON files).

## Overview

The visualization tool analyzes simulation results and generates plots to help understand:

- **I/O Patterns**: Local/remote read/write volumes per event and in total
- **Resource Utilization**: CPU/memory utilization and resource cost
- **Performance Metrics**: Event throughput, CPU time per event, efficiency

## Files

- `workflow_visualization.py`: Main visualization script

## Requirements

The visualization tools require the following Python packages (included in `requirements.txt`):

```
matplotlib>=3.7.0
pandas>=2.0.0
numpy>=1.24.0
```

Install with `make setup` or `pip install -r requirements.txt`.

## Usage

### Usage

The script scans a directory (recursively) for simulation result JSON files and creates
comparison plots across all valid files it can process.

- Run on a directory of results (recursively):

```bash
python scripts/workflow_visualization.py results/sim/others/5tasks_fullsim/
```

- Specify a custom output directory (default: `output/`):

```bash
python scripts/workflow_visualization.py results/sim/others/5tasks_fullsim/ --output-dir results/vis/others/5tasks_fullsim
```

The current script always produces the full set of supported plots (no `--plots` selector).

### Real vs simulated I/O (StepChain / TaskChain)

Compare normalized CMS data (`results/real_norm`) against one simulated scenario (e.g.
`seq_real` at 12h, fr0, 100MBps) for **const 1 (StepChain)** and **const 16 (TaskChain)**:

```bash
python scripts/plot_real_vs_sim_io_comparison.py \
  --real-dir results/real_norm \
  --sim-dir results/sim/others/seq_real/12h/fr0/100MBps \
  --output-dir results/vis/comparison/real_vs_seq_real_12h_fr0_100MBps
```

Outputs (same metrics as `plot_io_patterns`):

- **io_patterns_real_vs_sim_local.png** — per-event + stacked totals (includes local read)
- **io_patterns_real_vs_sim_nonlocal.png** — per-event + stacked totals (remote read, local/remote write)

At each x position (StepChain, TaskChain), **Real (normalized)** and **Simulated** bar groups are
shown side by side. Figure size matches ``plot_io_patterns`` (**6×6 in**). Two legend rows sit in the
reserved bottom margin below the axes. All comparison logic lives in
`plot_real_vs_sim_io_comparison.py` (reuses stacked figure constants and volume-axis helpers from
`workflow_visualization.py`).

## Generated Output

The script generates the following files:

### Visualization Plots (PNG)

- **io_patterns_comparison_local.png** (2×1 stacked, width ≈ 2/3 of legacy 16 in):
  - Top: data volume per event — Local Read, Remote Read, Local Write, Remote Write
  - Bottom: total data volumes (stacked); y-axis **MB / GB / TB / PB** (binary 1024) from max stack
  - Legend: one horizontal row below the bottom x-axis (4 series), not overlaid on bars

- **io_patterns_comparison_nonlocal.png** (2×1 stacked, same width):
  - Top: data volume per event — Remote Read, Local Write, Remote Write
  - Bottom: total data volumes (stacked); same dynamic **MB / GB / TB / PB** y-axis
  - Legend: horizontal below the bottom panel (3 series)

- **resource_utilization_comparison.png** (3×1 stacked, same width as I/O comparison plots):
  - Top → bottom: network transfer per event (MB), memory utilization ratio, CPU utilization
    ratio; shared **Workflow Construction** x-axis (ticks **1 … n** on the bottom panel only)

- **resource_cost_comparison.png** (single panel, same width):
  - Total CPU cores used (left axis) and total memory used in GB (right axis)

- **processing_efficiency_comparison.png** (wide, same width as I/O comparison plots):
  - CPU time per event (bars) overlaid with CPU utilization (line); x ticks **1 … n**

- **performance_vs_remote_write_comparison.png** (narrower figure):
  - **One point per workflow construction** (each simulation / construction on the x–y axes is
    a single workflow-wide aggregate, not one point per taskset group). Scatter: event throughput
    vs remote write per event; each point is labeled **1 … n** (same order as other comparison
    plots). Axes use **tight limits** around the data (no large empty margins).

### Text Report

- If enabled in future, additional summary tables may be added. At present, only the PNG
  visualizations are written.

## Notes on Data Source

- All inputs are consumed directly from existing simulation outputs under `results/sim/`.
- Files are discovered recursively; any `*.json` in the directory tree is considered.
- Metrics are extracted from the `metrics` object and from the first job of each group under
  `simulation_result.jobs` to keep memory usage low while enabling group-level analysis.

## Data Structure

The visualization tool expects simulation results in the following JSON structure:

```json
{
  "metrics": {
    "workflow_id": "...",
    "total_events": 2666667,
    "total_tasksets": 5,
    "total_groups": 1,
    "total_jobs": 485,
    "cpu_utilization": 0.606,
    "memory_occupancy": 0.894,
    ...
  },
  "simulation_result": {
    "groups": [
      {
        "group_id": "group_14",
        "job_count": 485,
        "input_events": 5502,
        "total_execution_time": 43195.1016,
        "tasksets": [...]
      }
    ],
    "jobs": [
      {
        "job_id": "group_14_job_1",
        "group_id": "group_14",
        "batch_size": 5502,
        "wallclock_time": 43195.1016,
        "cpu_utilization": 0.606,
        ...
      }
    ]
  }
}
```

## Customization

The visualization script can be easily customized by:

1. **Modifying plot styles**: Update colors, fonts, and layouts inside plotting functions
2. **Adding new plots**: Add new plotting functions following existing patterns and call them
3. **Customizing metrics**: Extend metric extraction in `extract_group_metrics`/`extract_job_metrics`
4. **Changing output formats**: Adjust `plt.savefig()` to write different formats or resolutions

## Integration

The visualization tools integrate seamlessly with DAGFlowSim:

1. Run simulations to produce JSON under `results/sim/`
2. Generate visualizations using `scripts/workflow_visualization.py` (outputs to `results/vis/` by default)
3. Analyze results using the generated PNG plots

This provides a complete workflow analysis pipeline from simulation to visualization.

