# Failure Rate Impact Analysis

This document describes the failure rate impact analysis script, which performs cross-dimensional comparisons to evaluate how different workflow constructions perform across various failure rates.

## Overview

**Analysis Type**: Failure Rate Impact (Comparison #1)

- **Fixed Dimensions**: workflow_type + target_job_length
- **Variable Dimension**: failure_rate (fr0, fr1, fr5, fr10, fr25)
- **Comparison**: All 16 constructions across failure rates
- **Primary Metric**: Event throughput

## Purpose

This analysis helps demonstrate:
1. How different workflow constructions (1-16) handle increasing failure rates
2. Which hybrid constructions (2-15) maintain better performance under failures
3. Comparison of hybrid compositions vs. extremes (Const 1: all chained, Const 16: all independent)
4. Identification of the best hybrid construction for each failure rate (based on event throughput, and network activity as a tiebraker)

## Usage

### Command Line

```bash
python scripts/failure_rate_analysis.py \
    <base_path> \
    <workflow_type> \
    <target_job_length> \
    [--output-dir OUTPUT_DIR]
```

### Arguments

- `base_path`: Base path to results directory (e.g., `results/sim/others`)
- `workflow_type`: Workflow type (e.g., `seq_real`, `seq_homo`, `seq_hetero`)
- `target_job_length`: Target job length (e.g., `12h`, `15m`, `24h`)
- `--output-dir`: Optional output directory (default: `results/analysis/failure_rate/{workflow_type}/{target_job_length}`)

### Examples

#### Single Analysis

```bash
# Analyze seq_real at 12h
python scripts/failure_rate_analysis.py \
    results/sim/others \
    seq_real \
    12h
```

#### Using Makefile

```bash
# Run analysis for all workflow types, all target job lengths
make analyze-failure-rate
```

## Output Location

Results are saved to:
```
results/analysis/failure_rate/{workflow_type}/{target_job_length}/
```

This structure separates cross-dimensional analysis outputs from standard simulation results (`results/sim/`) and standard visualizations (`results/vis/`).

## Output Files

The script generates the following outputs in the specified output directory:

### Visualizations

1. **`throughput_vs_failure_rate.png`**
   - Line chart showing event throughput vs. failure rate for all 16 constructions
   - Const 1 (all chained) highlighted in red
   - Const 16 (all independent) highlighted in green
   - Hybrid constructions (2-15) shown as lighter lines

2. **`throughput_degradation.png`**
   - Line chart showing throughput degradation percentage (relative to fr0) vs. failure rate
   - Helps identify which constructions are most resilient to failures
   - Negative values indicate improvement (shouldn't happen), zero is baseline

3. **`network_activity_vs_failure_rate.png`**
   - Two-panel visualization showing network activity patterns
   - **Left panel**: Network transfer per event vs. failure rate for all 16 constructions
   - **Right panel**: Remote read vs. remote write breakdown for Const 1, Const 16, and best hybrid
   - Helps identify which constructions maintain efficient network usage under failures
   - Shows how failure rates affect remote I/O patterns

4. **`best_hybrid_comparison.png`**
   - Bar chart comparing Const 1, Const 16, and the best hybrid construction for each failure rate
   - Best hybrid is identified based on event throughput
   - Shows which hybrid construction performs best at each failure rate

### Data Tables

5. **`failure_rate_analysis_summary.csv`**
   - Comprehensive table with all metrics for all constructions across all failure rates
   - Columns include:
     - Composition number
     - Failure rate (intended, from simulation config)
     - Total jobs, total job retries, total logical jobs, failure rate actual %
     - Event throughput, wall time per event, CPU time per event
     - Network transfer per event, CPU utilization, memory occupancy, total groups

## Interpretation Guide

### Throughput vs. Failure Rate Plot

- **Steeper negative slopes** indicate constructions that degrade more with failures
- **Flatter lines** indicate more resilient constructions
- **Higher lines** indicate better absolute performance
- Look for hybrid constructions that maintain higher throughput across all failure rates

### Throughput Degradation Plot

- **Lower values** (closer to zero) indicate better resilience
- **Higher positive values** indicate worse degradation
- Compare Const 1 and Const 16 to see which extreme degrades more
- Identify hybrid constructions with minimal degradation

### Network Activity Plot

- **Left panel**: Shows total network transfer patterns across all constructions
  - Lower network transfer generally indicates better efficiency
  - Look for constructions that maintain low network usage even under failures
- **Right panel**: Shows remote read/write breakdown for key constructions
  - Helps understand I/O patterns: more reads indicate cross-group dependencies
  - Const 1 typically has minimal remote read (all in one group)
  - Const 16 typically has more remote read (groups are independent)
  - Best hybrid should show a balanced pattern

### Best Hybrid Comparison Plot

- Shows which hybrid construction (2-15) performs best at each failure rate
- Direct comparison with Const 1 and Const 16
- Helps identify if there's a single "best" hybrid or if it varies by failure rate
- The legend shows which construction number(s) are the best hybrid

## Key Insights to Look For

1. **Resilience**: Do hybrid constructions show better resilience (less degradation) than extremes?
2. **Consistency**: Is there a single best hybrid construction, or does it vary by failure rate?
3. **Performance Gap**: How much better are hybrids compared to Const 1 and Const 16?
4. **Failure Rate Sensitivity**: At what failure rate do differences become most pronounced?
5. **Network Efficiency**: Do hybrid constructions maintain better network efficiency under failures?
6. **I/O Patterns**: How do remote read/write patterns differ between extremes and hybrids?

## Requirements

- Python 3.x
- Required packages: `matplotlib`, `numpy`, `pandas`
- Simulation result JSON files organized in the hierarchical structure:
  ```
  results/sim/others/
    {workflow_type}/
      {target_job_length}/
        fr0/
          *.json
        fr1/
          *.json
        fr5/
          *.json
        fr10/
          *.json
        fr25/
          *.json
  ```

## Notes

- The script automatically processes all available failure rate directories (fr0, fr1, fr5, fr10, fr25)
- Missing directories are skipped with a warning
- The best hybrid is identified using event throughput as the primary metric
- Simulations always include overhead; the script reads simulation result *.json files
