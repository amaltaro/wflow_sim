# Data Transfer Rate Sensitivity Analysis

This document describes the data transfer rate sensitivity analysis, which evaluates how different network data transfer rates (10 MB/s, 100 MB/s, 1 GB/s, 10 GB/s) affect workflow construction performance.

## Overview

**Analysis Type**: Data Transfer Rate Sensitivity

- **Fixed Dimensions**: 12h target job length, 0% failure rate (fr0), all 3 workflow types
- **Variable Dimension**: network data transfer rate (10, 100, 1000, 10000 MB/s)
- **Compare**: Const 1, Const 16, and best hybrid across data rates
- **Primary Metric**: Event throughput
- **Second Metric**: Network transfer per event

## Purpose

This analysis helps demonstrate:

1. How workflow constructions (1-16) perform under different network overhead assumptions
2. Whether hybrid constructions remain beneficial at low vs. high data transfer rates
3. Sensitivity of throughput and network efficiency to the data transfer rate parameter
4. Identification of the best hybrid construction at each rate (per workflow type)

## Simulation Setup

Simulations use the **unified** results structure (data transfer rate is a dimension alongside workflow type, target job length, and failure rate):

- **Target job length**: 12h
- **Failure rate**: 0% (fr0)
- **Workflow types**: case1_real, case2_homo, case3_hetero
- **Data transfer rates**: 10 MB/s, 100 MB/s, 1 GB/s, 10 GB/s
- **Output organization**: `results/sim/others/<workflow_type>/12h/fr0/<rate_dir>/*.json`  
  (`rate_dir` is one of `10MBps`, `100MBps`, `1GBps`, `10GBps`)

Rate directory names use uppercase B (MBps/GBps = bytes per second) to avoid confusion with bits (Mbps/Gbps).

## Running Simulations for This Analysis

Either run the full suite (all times, failure rates, and data rates) or only the 12h+fr0+4 rates subset:

- **Full suite**: `make simulate-all` — writes to `results/sim/others/<case>/<time>/fr<fr>/<data_rate>/`
- **Data transfer rate subset only (12h, fr0, 4 rates)**: `make simulate-data-transfer-rate` — same structure, fewer combinations

Example single run (10 MB/s):

```bash
python -m src.workflow_runner \
  --target-wallclock-time 43200 \
  --failure-rate 0 \
  --data-transfer-rate 10 \
  --input-workflow-path templates/others/case1_real/case1_real_const_001.json
# Result: results/sim/others/case1_real/12h/fr0/10MBps/case1_real_const_001.json
```

## Running the Analysis

After simulations exist under the unified structure, point the script at the **simulation base** (the directory that contains workflow-type subdirs, e.g. `results/sim/others`):

```bash
python scripts/data_transfer_rate_analysis.py results/sim/others [--output-dir RESULTS_DIR]
```

The script looks for `base_path/<workflow_type>/12h/fr0/<rate_dir>/*.json` for each rate directory.

### Arguments

- `base_path`: Base path to simulation results containing workflow-type subdirs (e.g. `results/sim/others`)
- `--rate-dirs`: Rate directory names (default: `10MBps 100MBps 1GBps 10GBps`)
- `--workflow-types`: Workflow types to analyze (default: `case1_real case2_homo case3_hetero`)
- `--output-dir`: Output directory (default: `results/analysis/data_transfer_rate`)

### Example

```bash
# Run analysis (after simulate-all or simulate-data-transfer-rate)
python scripts/data_transfer_rate_analysis.py results/sim/others

# Custom output directory
python scripts/data_transfer_rate_analysis.py results/sim/others \
  --output-dir results/analysis/data_transfer_rate/custom
```

## Using the Makefile

```bash
# Option A: Run only 12h+fr0+4 data rates (faster for this analysis)
make simulate-data-transfer-rate
make analyze-data-transfer-rate

# Option B: Run full suite (all times, failure rates, data rates), then analyze
make simulate-all
make analyze-data-transfer-rate   # reads 12h/fr0/<rate_dir> from unified tree
```

## Output Location

Results are saved under:

```
results/analysis/data_transfer_rate/
```

## Output Files

The script generates:

### Visualizations

1. **`throughput_vs_data_transfer_rate.png`**
   - Three panels (one per workflow type): event throughput vs. data transfer rate (log scale)
   - Const 1, Const 16, and best hybrid; best hybrid construction number annotated
   - Shows how throughput changes with network overhead assumption

2. **`network_efficiency_vs_data_transfer_rate.png`**
   - Three panels: network transfer per event (MB/event) vs. data transfer rate (log scale)
   - Const 1, Const 16, and best hybrid
   - Shows whether network efficiency is sensitive to the rate parameter

### Data Tables

3. **`data_transfer_rate_analysis_summary.csv`**
   - Rows: one per (rate, workflow_type, composition_number)
   - Columns: data_transfer_rate_mbps, rate_dir, workflow_type, composition_number, event_throughput, wall_time_per_event, cpu_time_per_event, network_transfer_mb_per_event, cpu_utilization, memory_occupancy, total_groups

## Requirements

- Python 3.x
- Required packages: `matplotlib`, `numpy`, `pandas`
- Simulation result JSON files in the **unified** structure (`base_path` = e.g. `results/sim/others`):
  ```
  {base_path}/
    case1_real/12h/fr0/10MBps/*.json
    case1_real/12h/fr0/100MBps/*.json
    case1_real/12h/fr0/1GBps/*.json
    case1_real/12h/fr0/10GBps/*.json
    case2_homo/12h/fr0/10MBps/
    ...
    case3_hetero/12h/fr0/10GBps/
  ```

## Notes

- The script expects fixed 12h and fr0; it reads `base_path/<workflow_type>/12h/fr0/<rate_dir>/*.json`.
- Missing rate or workflow directories are skipped with a warning.
- Best hybrid is identified per (rate, workflow_type) using event throughput with network transfer as tiebreaker.
- Data transfer rate is used in the simulator to compute network transfer overhead (time = data_mb / rate_mb_per_s).
