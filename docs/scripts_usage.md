# Scripts Usage Guide

This document provides brief documentation for utility scripts in the `scripts/` directory.

## Data Processing Scripts

### `normalize_real_metrics.py`

Normalizes real workflow execution metrics to a target event count (default: 1M events) for fair comparison with simulated data.

**Purpose**: Real workflow data often processes fewer events than simulated workflows due to job failures. This script scales whole-workflow total metrics (time, CPU, memory, I/O) to enable accurate comparison.

**Usage**:
```bash
python scripts/normalize_real_metrics.py input_file.json output_file.json
python scripts/normalize_real_metrics.py results/real/summary_const001.json results/real_norm/summary_const001.json
```

**What it does**:
- Scales total metrics (wallclock time, CPU time, memory, I/O volumes) by `target_events / actual_events`
- Sets ``event_metrics.total_events`` to the target count
- Preserves event-normalized metrics (per-event values, ratios, utilization) unchanged
- Preserves workflow turnaround time unchanged (reflects real-world execution)

**Options**:
- `--target-events`: Target number of events (default: 1,000,000)

### `condor_data_metrics.py`

Extracts high-level workflow statistics from Elasticsearch condor producer documents.

**Purpose**: Processes raw Elasticsearch job data to calculate workflow-level metrics including execution times, resource usage, I/O patterns, and event throughput.

**Usage**:
```bash
python scripts/condor_data_metrics.py input.json output.json
```

**Metrics calculated**:
- Document and job counts (by type and taskset)
- Time metrics: total wallclock time, workflow turnaround time, overhead
- CPU metrics: total CPU time (used/allocated), CPU utilization
- Memory metrics: total memory (used/allocated), memory utilization
- I/O metrics: total read/write (local/remote) volumes
- Event metrics: total events processed, throughput, time per event

**Output**: JSON file with structured metrics organized by category (time, CPU, memory, I/O, event).

### `explore_job_data.py`

Explores grid job information from Elasticsearch, extracting job-level metrics similar to those simulated.

**Purpose**: Analyzes individual job data to extract detailed metrics per job including events processed, timing, CPU usage, I/O operations, and resource allocation.

**Usage**:
```bash
python scripts/explore_job_data.py input.json [--output output.json]
```

**Metrics extracted**:
- Per-job: events processed, turnaround time, CPU time, I/O volumes
- Resource allocation: cores, memory
- CMSSW step information
- Event throughput per job

**Output**: Detailed job-level metrics in JSON format.

### `get_time_per_event.py`

Extracts time and size per event metrics from condor producer documents.

**Purpose**: Calculates per-event metrics (time and size) from the last CMSSW run in each job, useful for workflow characterization.

**Usage**:
```bash
python scripts/get_time_per_event.py input.json [--output output.json]
```

**Metrics calculated**:
- Time per event: `ChirpCMSSW_cmsRunXXX_Elapsed / ChirpCMSSW_cmsRunXXX_Events`
- Size per event: `ChirpCMSSW_cmsRunXXX_WriteBytes / ChirpCMSSW_cmsRunXXX_Events` (in KB)

**Note**: Only includes Production and Processing jobs. Skips internally restarted jobs.

## Visualization Scripts

### `workflow_visualization.py`

Generates comparison plots from simulation results.

**Purpose**: Creates visualization diagrams comparing multiple workflow compositions, showing I/O patterns, resource utilization, and performance metrics.

**Usage**:
```bash
python scripts/workflow_visualization.py results/sim/others/seq_real/ [--output-dir results/vis/others/seq_real]
```

**Output**: PNG plots for I/O patterns, resource utilization, and performance metrics comparison.

**See also**: [Visualization Usage Guide](visualization_usage.md) for detailed documentation.

### `plot_construction_groups_overview.py`

Plots an overview of how tasksets are grouped across all constructions in a compositions summary JSON.

**Purpose**: Produces a compact "barcode" figure where each row is a construction (composition), each
column is a taskset index, and contiguous blocks are colored and labeled by `group_id`.

**Usage**:
```bash
python scripts/plot_construction_groups_overview.py \
  --summary-json templates/others/seq_real/seq_real_compositions_summary.json \
  --output-dir results/vis/others/seq_real/
```

**Output**: Writes `<output-dir>/<template_name>_construction_groups_overview.png`.

**Options**:
- `--show-colorbar`: Show the colorbar legend (hidden by default because group ids are labeled inside blocks)

### `real_workflow_visualization.py`

Generates comparison plots from real workflow execution data.

**Purpose**: Transforms real workflow metrics (from `condor_data_metrics.py` output) into simulation format and generates the same visualization plots for comparison with simulated data.

**Usage**:
```bash
# Raw execution summaries (default: results/real)
python scripts/real_workflow_visualization.py --input-dir results/real

# Normalized to requested events (results/real_norm)
python scripts/real_workflow_visualization.py --input-dir results/real_norm

# Both raw and normalized in one run
python scripts/real_workflow_visualization.py --input-dir results/real --also-normalized
```

**What it does**:
- Reads real data summary JSON files (`summary_*.json` from `condor_data_metrics.py`)
- Transforms metrics to simulation format
- Generates comparison plots via `workflow_visualization.py` (same layout as simulated comparisons)
- Sorts workflows by composition number; x-axis labels use **StepChain** / **TaskChain** for const 1 / 16

**Output** (when at least two workflows are present), per output directory:
- `io_patterns_comparison_local.png`, `io_patterns_comparison_nonlocal.png` (2×1 stacked; legend below x-axis)
- `resource_utilization_comparison.png` (3×1 network / memory / CPU), `resource_cost_comparison.png`
- `processing_efficiency_comparison.png`, `performance_vs_remote_write_comparison.png`, `turnaround_time_comparison.png`
- `workflow_summary_table.csv` / `.txt`

### `construction_metrics_analysis.py`

Multi-metric comparison and weighted score for the 16 workflow constructions in a single scenario. Produces a heatmap, two score plots, and a CSV (raw + normalized metrics + `weighted_score`).

**See**: [Construction metrics analysis](construction_metrics_analysis.md) for full documentation (metrics, normalization, score formula, outputs, usage, and Makefile target).

### `run_multiseed_simulations.py`

Runs all `*_const_*.json` constructions for **one** scenario across **N** RNG
seeds (seed = run index `0 .. N-1`). Each seed is applied to every construction
before moving to the next seed. Invokes `python -m src.workflow_runner` (no
Makefile changes).

**Default scenario** (paper rebuttal figure): `seq_real`, 12h, fr5, 100 MB/s,
10 runs.

```bash
# Defaults: seq_real / 12h / fr5 / 100MBps / 10 seeds
python scripts/run_multiseed_simulations.py

# Smaller smoke run
python scripts/run_multiseed_simulations.py --runs 2
```

**Output**:
- `results/sim/rebuttal/campaign.json` — campaign parameters and seed list
- `results/sim/rebuttal/seed{S}/others/<use_case>/<time>/fr<fr>/<rate>/*.json`

Template paths must live under the repository so `--output-base` nesting works.

### `run_multiseed_visualization.py`

Aggregates multi-seed results under `results/sim/rebuttal/seed*/` by construction,
then writes comparison PNGs with **mean ± SEM** error bars when failure rate > 0
(and N > 1). The throughput vs remote-write scatter uses **mean** points only
(no SEM). Reuses layout helpers from `workflow_visualization.py`.

```bash
# Defaults: input results/sim/rebuttal → results/vis/rebuttal/<use_case>/...
python scripts/run_multiseed_visualization.py

python scripts/run_multiseed_visualization.py \
  --input-root results/sim/rebuttal \
  --output-dir results/vis/rebuttal/seq_real/12h/fr5/100MBps
```

**Outputs** (same names as the single-run visualizer where applicable):
- `processing_efficiency_comparison.png`
- `performance_vs_remote_write_comparison.png`
- `turnaround_time_comparison.png`
- `resource_utilization_comparison.png`, `resource_cost_comparison.png`
- `multiseed_aggregation_summary.csv`

## Workflow

Typical workflow for analyzing real vs simulated data:

1. **Extract real data metrics**:
   ```bash
   python scripts/condor_data_metrics.py data/const001.json results/real/summary_const001.json
   ```

2. **Normalize to 1M events** (optional, for comparison):
   ```bash
   python scripts/normalize_real_metrics.py results/real/summary_const001.json results/real_norm/summary_const001.json
   ```

3. **Visualize real data**:
   ```bash
   python scripts/real_workflow_visualization.py results/real/
   ```

4. **Compare with simulated data**:
   ```bash
   python scripts/workflow_visualization.py results/sim/others/seq_real/
   ```
