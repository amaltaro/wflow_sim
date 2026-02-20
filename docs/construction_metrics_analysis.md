# Construction Metrics Analysis

This document describes the **construction metrics analysis** implemented in
`scripts/construction_metrics_analysis.py`: multi-metric comparison and a single
weighted score for each of the 16 workflow constructions in a given scenario.

## Purpose

For a fixed scenario (workflow type, target job length, failure rate, data transfer rate),
the script:

- Loads all 16 workflow construction result JSONs from the scenario directory
- Normalizes selected metrics to [0, 1] (higher = better) for fair comparison
- Produces a **heatmap** (constructions × metrics) and a **weighted score** per construction
- Writes a **CSV** of raw and normalized metrics plus the score for downstream use

The score supports research goals such as maximizing event throughput and improving
resource utilization by letting you prioritize metrics via weights.

## Input

- **Scenario directory**: Path to a directory containing exactly the 16 simulation
  result JSONs (e.g. `case1_real_const_001.json` … `case1_real_const_016.json`).
- Typical path pattern:  
  `results/sim/others/<workflow_type>/<target_job_length>/<failure_rate>/<data_rate>/`

## Metrics

### Heatmap and CSV (11 metrics)

All eleven metrics appear in the heatmap and in the CSV, in this order (left to right):

| Order | Key | Label | Higher is better |
|-------|-----|--------|------------------|
| 1 | event_throughput | Throughput | Yes |
| 2 | total_cpu_cores_used | Alloc CPU Cores | No |
| 3 | cpu_utilization | CPU Util | Yes |
| 4 | cpu_cores_per_event | CPU Cores/Evt | No |
| 5 | total_memory_used_mb | Alloc Memory | No |
| 6 | memory_occupancy | Memory Occ | Yes |
| 7 | memory_mb_per_event | Memory MB/Evt | No |
| 8 | total_turnaround_time | Turnaround | No |
| 9 | wall_time_per_event | Wall Time/Evt | No |
| 10 | network_transfer_mb_per_event | Net MB/Evt | No |
| 11 | total_write_remote_mb | Write Remote | No |

### Normalization

Each metric is min–max normalized across the 16 constructions to [0, 1]:

- **Higher-is-better** (e.g. throughput, CPU util, memory occ):  
  `normalized = (x - min) / (max - min)`
- **Lower-is-better** (e.g. turnaround, network transfer):  
  `normalized = (max - x) / (max - min)`  
  so that “better” still corresponds to a **higher** normalized value.

After normalization, **1 = best** and **0 = worst** for every metric, which allows a
single weighted score and consistent heatmap coloring.

## Weighted score

A single score per construction is computed from a **subset** of metrics and
**weights** that sum to 1.0:

```
score = Σ (weight_i × normalized_metric_i)
```

### Default score metrics and weights

| Metric key | Default weight | Rationale |
|------------|----------------|-----------|
| event_throughput | 0.40 | Primary research focus: maximize throughput |
| cpu_cores_per_event | 0.20 | Per-event CPU intensity (lower is better; normalized) |
| memory_mb_per_event | 0.20 | Per-event memory intensity (lower is better; normalized) |
| network_transfer_mb_per_event | 0.20 | Per-event network transfer (lower is better; normalized) |
| **Sum** | **1.00** | |

The score is in **[0, 1]** (same scale as each normalized metric).

### Customizing the score

To change metrics or weights, edit in `scripts/construction_metrics_analysis.py`:

- **`SCORE_METRICS_WEIGHTS`**: dictionary mapping each metric key to its weight
  (e.g. `{'event_throughput': 0.4, 'cpu_cores_per_event': 0.2, ...}`). Keys must exist
  in the heatmap/CSV metric list. **Weights must sum to 1.0**.

The default score uses **cpu_cores_per_event** and **memory_mb_per_event** (per-event
intensity; correlation analysis supports these as strong drivers of throughput). You
can switch back to **cpu_utilization** and **memory_occupancy** (efficiency) in
`SCORE_METRICS_WEIGHTS` if preferred.

### Which CPU, memory, and I/O metrics are most meaningful for throughput?

The script **`scripts/analyze_throughput_drivers.py`** compares how well ten
metrics (including the one used in the score formula) correlate with **event_throughput**.
Output is ordered by resource (CPU, then Memory, then I/O):

- **CPU:** cpu_utilization (efficiency), total_cpu_cores_used, cpu_cores_per_event (intensity)
- **Memory:** memory_occupancy (efficiency), total_memory_used_mb, memory_mb_per_event (intensity)
- **I/O:** total_network_transfer_mb, network_transfer_mb_per_event (used in the score), total_write_remote_mb_per_event, total_read_remote_mb_per_event (lower is often better for efficiency)

It loads one scenario directory (same path as construction metrics) and prints
**Pearson correlation** of each of the ten with event_throughput. Higher |r|
means a stronger linear relationship; the sign indicates direction.

**Interpretation:**

- **Utilization metrics** (cpu_utilization, memory_occupancy): positive r with throughput
  means better utilization tends to go with higher throughput.
- **Total- or per-event usage** (total_cpu_cores_used, cpu_cores_per_event,
  total_memory_used_mb, memory_mb_per_event, total_network_transfer_mb,
  network_transfer_mb_per_event, total_write_remote_mb_per_event,
  total_read_remote_mb_per_event): negative r often means lower resource or I/O
  use (or per event) is associated with higher throughput (more efficient workflows).

Use the output to decide which **CPU**, **memory**, and **I/O** metrics to
include in the construction score (e.g. cpu_utilization vs cpu_cores_per_event,
memory_occupancy vs memory_mb_per_event); network_transfer_mb_per_event is already
in the default score formula.

**Usage:**

```bash
python scripts/analyze_throughput_drivers.py results/sim/others/case1_real/12h/fr5/100MBps
```

## Output directory schema

Outputs follow the same schema as other analysis scripts:

```
results/analysis/construction_metrics/<workflow_type>/<target_job_length>/<failure_rate>/<data_rate>/
```

If you run the script with a scenario path under `results/sim/others/...`, the
default output directory is derived automatically (e.g. from `.../case1_real/12h/fr5/100MBps`
to `results/analysis/construction_metrics/case1_real/12h/fr5/100MBps`). You can
override with `--output-dir`.

## Output files

| File | Description |
|------|-------------|
| **construction_metrics_heatmap.png** | Heatmap: rows = Const 1–16, columns = 11 metrics; color = normalized score (green = 1, red = 0). |
| **construction_score_bars.png** | Bar chart: construction (x) vs weighted score (y); color by score; mean line. |
| **construction_score_ranked.png** | Horizontal bar chart: constructions sorted by score (best on bottom). |
| **construction_metrics.csv** | One row per construction: `construction`, then for each metric `*_raw` and `*_normalized`, plus `weighted_score`. |

## Usage

### Command line

```bash
python scripts/construction_metrics_analysis.py <simulation_dir> [options]
```

Examples:

```bash
# Default output dir derived from scenario path
python scripts/construction_metrics_analysis.py results/sim/others/case1_real/12h/fr5/100MBps

# Explicit output dir and title label
python scripts/construction_metrics_analysis.py results/sim/others/case1_real/12h/fr5/100MBps \
  --output-dir results/analysis/construction_metrics/case1_real/12h/fr5/100MBps \
  --scenario-label "case1_real 12h fr5 100MBps"
```

Options:

- `--output-dir PATH`: output directory (default: derived from simulation path when it matches the standard sim tree).
- `--scenario-label TEXT`: label used in plot titles.

### Makefile

The Makefile target **`analyze-construction-metrics`** runs the analysis for a fixed
scenario: 12h job length, 100 MB/s data rate, and all three workflow types. It runs
for **three failure rates** (fr0, fr5, fr25). Output is written under
`results/analysis/construction_metrics/<use_case>/12h/<fr>/100MBps/`.

```bash
make analyze-construction-metrics
```

Scenario parameters are controlled by Make variables: `CONSTRUCTION_METRICS_TIME`,
`CONSTRUCTION_METRICS_FR_LIST` (fr0 fr5 fr25), `CONSTRUCTION_METRICS_RATE`, `USE_CASES`.

## See also

- [Scripts usage](scripts_usage.md) – overview of all scripts, including this one
- [Visualization usage](visualization_usage.md) – workflow comparison visualizations
