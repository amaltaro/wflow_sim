# Workflow Type Sensitivity Analysis

This document describes the workflow type sensitivity analysis script, which evaluates how different workflow types respond to hybrid workflow constructions compared to extreme cases.

## Overview

**Analysis Type**: Workflow Type Sensitivity (Comparison #2)

- **Fixed Dimensions**: target_job_length + failure_rate
- **Variable Dimension**: workflow_type (seq_real, seq_homo, seq_hetero)
- **Comparison**: Const 1, Const 16, and best hybrid across workflow types
- **Primary Metric**: Event throughput
- **Second Metric**: Network transfer per event (as tiebreaker)

## Workflow Type Characteristics

The three workflow types differ in their resource characteristics:

- **seq_real**: Mixed/realistic resource requirements
  - Varied memory: 3000MB, 7000MB, 8000MB, 4000MB, 4000MB
  - Varied cores: 8, 4, 4, 2, 4

- **seq_homo**: Homogeneous resource requirements
  - Uniform memory: 8000MB for all tasksets
  - Uniform cores: 8 for all tasksets

- **seq_hetero**: Highly heterogeneous resource requirements
  - Extreme memory variation: 2000MB, 16000MB, 64000MB, 10000MB, 8000MB
  - Extreme core variation: 1, 8, 64, 4, 4

While time and size per event remain the same across all workflow scenarios. Likewise, requirements to stage output data out (KeepOutput option) shares the same setup.

## Usage

### Command Line

```bash
python scripts/workflow_type_sensitivity.py \
    <base_path> \
    <target_job_length> \
    <failure_rate> \
    [--workflow-types WORKFLOW_TYPES ...] \
    [--output-dir OUTPUT_DIR]
```

### Arguments

- `base_path`: Base path to results directory (e.g., `results/sim/others`)
- `target_job_length`: Target job length (e.g., `12h`, `15m`, `24h`)
- `failure_rate`: Failure rate directory (e.g., `fr0`, `fr10`)
- `--workflow-types`: Optional list of workflow types to analyze (default: `seq_real seq_homo seq_hetero`)
- `--output-dir`: Optional. If omitted, output goes under
  `results/analysis/workflow_type_sensitivity/`: if `seq_real` is among
  `--workflow-types`, use `sequential/{target_job_length}/{failure_rate}`; else
  if `fork_real` is present, use `fork/{target_job_length}/{failure_rate}`; else
  `{target_job_length}/{failure_rate}` with no `sequential`/`fork` segment.

### Examples

#### Single Analysis (Recommended: 12h, fr0)

```bash
# Analyze all workflow types at 12h with 0% failure rate (baseline)
python scripts/workflow_type_sensitivity.py \
    results/sim/others \
    12h \
    fr0
```

#### Custom Workflow Types

```bash
# Analyze only specific workflow types
python scripts/workflow_type_sensitivity.py \
    results/sim/others \
    12h \
    fr0 \
    --workflow-types seq_real seq_homo
```

#### Different Failure Rates

```bash
# Analyze at 10% failure rate
python scripts/workflow_type_sensitivity.py \
    results/sim/others \
    12h \
    fr10
```

#### Using Makefile

```bash
# Run analysis for 12h with failure rates 0%, 5%, 25% (fr0, fr5, fr25)
make analyze-workflow-type-sensitivity
```

## Output Location

Default root is `results/analysis/workflow_type_sensitivity/`, then a family
subfolder when the workflow set is unambiguous: `sequential/…` (sequential
family includes `seq_real`) or `fork/…` (fork family with `fork_real` and not
`seq_real`); otherwise `…/{target_job_length}/{failure_rate}/` with no
`sequential` or `fork` in the path.

This structure separates cross-dimensional analysis outputs from standard simulation results (`results/sim/`) and standard visualizations (`results/vis/`).

## Output Files

The script generates the following outputs in the specified output directory:

### Visualizations

The script writes two side-by-side comparison figures (IEEE-friendly layout) plus a summary CSV.

1. **`throughput_network_comparison.png`**
   - **(a) Throughput:** event throughput for most grouped, most ungrouped, and best hybrid
   - **(b) Network efficiency:** network transfer per event (MB) for the same three cases
   - Best hybrid construction id is labeled on hybrid bars

2. **`throughput_network_improvement.png`**
   - **(a) Throughput improvement:** `((best_hybrid_throughput - extreme_throughput) / extreme_throughput) × 100`
   - **(b) Network improvement:** `((extreme_network - best_hybrid_network) / extreme_network) × 100`
   - Paired bars compare best hybrid vs most grouped (red) and most ungrouped (green)
   - Positive throughput values indicate faster hybrids; positive network values indicate less I/O

### Data Tables

3. **`workflow_type_sensitivity_summary.csv`**
   - Comprehensive table with metrics for Const 1, Const 16, and best hybrid for each workflow type
   - Columns include:
     - Workflow type
     - Construction number
     - Event throughput
     - Wall time per event
     - CPU time per event
     - Network transfer per event
     - CPU utilization
     - Memory occupancy
     - Total groups

## Interpretation Guide

### Throughput Comparison Plot

- **Higher bars** indicate better performance
- Compare best hybrid (blue) against Const 1 (red) and Const 16 (green)
- Look for workflow types where best hybrid significantly outperforms both extremes
- The construction number above each best hybrid bar shows which hybrid is optimal for that workflow type

### Throughput Improvement Percentage Plot

- Shows **event throughput improvement** as a percentage
- **Positive values** indicate best hybrid has higher throughput than the extreme
- **Negative values** indicate best hybrid has lower throughput (should be rare)
- Compare throughput improvement over Const 1 vs. Const 16
- Higher positive values indicate larger throughput benefit from hybrid compositions
- Helps identify which workflow types show the most throughput improvement
- Example: +5% means best hybrid has 5% higher throughput than the extreme

### Network Efficiency Comparison Plot

- Shows **absolute network transfer values** (MB per event)
- **Lower values** indicate better network efficiency (less data transfer)
- Compare network usage patterns across workflow types
- Const 1 typically has minimal network transfer (all chained)
- Const 16 typically has more network transfer (independent groups)
- Best hybrid should show balanced network usage

### Network Improvement Percentage Plot

- Shows **network transfer reduction** as a percentage
- **Positive values** indicate best hybrid uses less network than the extreme
- **Negative values** indicate best hybrid uses more network (should be rare)
- Compare network reduction over Const 1 vs. Const 16
- Higher positive values indicate larger network efficiency benefit from hybrid compositions
- Helps identify which workflow types show the most network efficiency improvement
- Example: +20% means best hybrid uses 20% less network transfer than the extreme

## Key Insights to Look For

1. **Generalizability**: Do hybrid constructions show benefits across all workflow types?
2. **Workflow Type Sensitivity**: Which workflow types benefit most from hybrids?
   - Heterogeneous workflows (seq_hetero) may benefit more due to resource diversity
   - Homogeneous workflows (seq_homo) may show different patterns
3. **Consistency**: Is the same hybrid construction best across all workflow types, or does it vary?
4. **Improvement Magnitude**: How much better are hybrids compared to extremes for each workflow type?
5. **Network Patterns**: How do network efficiency patterns differ across workflow types?

## Recommended Configuration

For baseline analysis, use:
- **Target Job Length**: `12h` (middle ground, commonly used)
- **Failure Rate**: `fr0` (0% - clean baseline without failure complications)

This provides a clear view of how workflow characteristics affect hybrid construction benefits without the complexity of failures.

## Requirements

- Python 3.x
- Required packages: `matplotlib`, `numpy`, `pandas`
- Simulation result JSON files organized in the hierarchical structure:
  ```
  results/sim/others/
    {workflow_type}/
      {target_job_length}/
        {failure_rate}/
          *.json
- The analysis focuses on comparing extremes (Const 1, Const 16) with the best hybrid, not all 16 constructions
