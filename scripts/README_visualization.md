# Workflow Simulation Visualization

This directory contains tooling to visualize workflow simulation results using pandas, matplotlib,
and seaborn. The visualizer reads completed simulation outputs from the `results/` directory and
produces comparison plots across multiple workflow constructions (JSON files).

## Overview

The visualization tool analyzes simulation results and generates plots to help understand:

- **I/O Patterns**: Local/remote read/write volumes per event and in total
- **Resource Utilization**: CPU/memory utilization and resource cost
- **Performance Metrics**: Event throughput, CPU time per event, efficiency

## Files

- `workflow_visualization.py`: Main visualization script

## Requirements

The visualization tools require the following Python packages (already included in `requirements_visualization.txt`):

```
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
numpy>=1.21.0
```

## Usage

### Usage

The script scans a directory (recursively) for simulation result JSON files and creates
comparison plots across all valid files it can process.

- Run on a directory of results (recursively):

```bash
python scripts/workflow_visualization.py results/others/5tasks_fullsim/
```

- Specify a custom output directory (default: `output/`):

```bash
python scripts/workflow_visualization.py results/others/5tasks_fullsim/ --output-dir my_plots
```

The current script always produces the full set of supported plots (no `--plots` selector).

## Generated Output

The script generates the following files:

### Visualization Plots (PNG)

- **io_patterns_comparison.png**:
  - Data volume per event: Local Read, Remote Read, Local Write, Remote Write
  - Data flow per event: Remote Read, Local Write, Remote Write
  - Total data volumes (GB): Local/Remote Read/Write (stacked)
  - Total data volumes (GB): Remote Read, Local Write, Remote Write (stacked)

- **resource_utilization_comparison.png**:
  - Network transfer per event (MB)
  - CPU utilization ratio
  - Memory utilization ratio
  - Resource cost: total CPU cores (left axis) and total memory GB (right axis)

- **performance_metrics_comparison.png**:
  - Event throughput vs remote write per event (scatter)
  - CPU time per event (bars) overlaid with CPU utilization (line)

### Text Report

- If enabled in future, additional summary tables may be added. At present, only the PNG
  visualizations are written.

## Notes on Data Source

- All inputs are consumed directly from existing simulation outputs under `results/`.
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

The visualization tools integrate seamlessly with the workflow simulator:

1. Run simulations to produce JSON under `results/`
2. Generate visualizations using `scripts/workflow_visualization.py`
3. Analyze results using the generated PNG plots

This provides a complete workflow analysis pipeline from simulation to visualization.
