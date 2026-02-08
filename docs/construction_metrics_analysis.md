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

### Heatmap and CSV (nine metrics)

All nine metrics appear in the heatmap and in the CSV, in this order (left to right):

| Order | Key | Label | Higher is better |
|-------|-----|--------|------------------|
| 1 | event_throughput | Throughput | Yes |
| 2 | total_cpu_cores_used | Alloc CPU Cores | No |
| 3 | cpu_utilization | CPU Util | Yes |
| 4 | total_memory_used_mb | Alloc Memory | No |
| 5 | memory_occupancy | Memory Occ | Yes |
| 6 | total_turnaround_time | Turnaround | No |
| 7 | wall_time_per_event | Wall Time/Evt | No |
| 8 | network_transfer_mb_per_event | Net MB/Evt | No |
| 9 | total_write_remote_mb | Write Remote | No |

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
| cpu_utilization | 0.25 | Resource utilization |
| memory_occupancy | 0.25 | Resource utilization |
| network_transfer_mb_per_event | 0.10 | Lower is better (already normalized) |
| **Sum** | **1.00** | |

The score is in **[0, 1]** (same scale as each normalized metric).

### Customizing the score

To change metrics or weights, edit in `scripts/construction_metrics_analysis.py`:

- **`SCORE_METRICS_WEIGHTS`**: dictionary mapping each metric key to its weight
  (e.g. `{'event_throughput': 0.4, 'cpu_utilization': 0.25, ...}`). Keys must exist
  in the heatmap/CSV metric list. **Weights must sum to 1.0**.

Alternative metrics you might consider:

- **cpu_cores_per_event** (vs cpu_utilization): resource intensity per event;
  add to the script’s metric list if you want to use it in the score.
- **memory_mb_per_event** (vs memory_occupancy): same idea for memory.

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
| **construction_metrics_heatmap.png** | Heatmap: rows = Const 1–16, columns = 9 metrics; color = normalized score (green = 1, red = 0). |
| **construction_score_bars.png** | Bar chart: construction (x) vs weighted score (y); color by score; mean line. |
| **construction_score_ranked.png** | Horizontal bar chart: constructions sorted by score (best on top). |
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
scenario: 12h job length, 5% failure rate, 100 MB/s data rate, and all three workflow
types. Output is written under `results/analysis/construction_metrics/<use_case>/12h/fr5/100MBps/`.

```bash
make analyze-construction-metrics
```

Scenario parameters are controlled by Make variables: `CONSTRUCTION_METRICS_TIME`,
`CONSTRUCTION_METRICS_FR`, `CONSTRUCTION_METRICS_RATE`, `USE_CASES`.

## See also

- [Scripts usage](scripts_usage.md) – overview of all scripts, including this one
- [Visualization usage](visualization_usage.md) – workflow comparison visualizations
