# Target Job Length Optimization Analysis

This document describes the target job length optimization analysis script, which performs cross-dimensional comparisons to evaluate how different workflow constructions perform across various target job lengths.

## Overview

**Analysis Type**: Target Job Length Optimization (Comparison #3)

- **Fixed Dimensions**: workflow_type + failure_rate
- **Variable Dimension**: target_job_length (15m, 30m, 1h, 2h, 4h, 8h, 12h, 24h)
- **Comparison**: All 16 constructions across target job lengths
- **Primary Metric**: Event throughput
- **Second Metric**: Network transfer per event (as tiebreaker)

## Purpose

This analysis helps demonstrate:
1. How different workflow constructions (1-16) handle varying time constraints
2. Which hybrid constructions (2-15) maintain better performance across different target job lengths
3. Comparison of hybrid compositions vs. extremes (Const 1: all chained, Const 16: all independent)
4. Identification of the best hybrid construction for each target job length (based on event throughput, and network activity as a tiebreaker)
5. Understanding of time-dependent patterns and trade-offs between job granularity and overhead

## Key Insights

This analysis reveals important patterns:

1. **Overhead Sensitivity**: 
   - At shorter target lengths (15m, 30m, 1h), constructions with many groups may be penalized by overhead
   - At longer target lengths (8h, 12h, 24h), constructions with few groups may miss parallelism opportunities
   - Hybrid constructions may show optimal balance at intermediate lengths (2h, 4h)

2. **Optimal Construction Shifts**:
   - The best hybrid may change with target job length
   - Const 1 might be better at very short constraints (minimize overhead)
   - Const 16 might be better at very long constraints (maximize parallelism)
   - Hybrids may dominate in the middle range

3. **Time-Dependent Efficiency**:
   - Network efficiency patterns may change with batch size
   - CPU utilization may vary with job granularity
   - Memory occupancy patterns may shift

## Usage

### Command Line

```bash
python scripts/target_job_length_analysis.py \
    <base_path> \
    <workflow_type> \
    <failure_rate> \
    [--output-dir OUTPUT_DIR]
```

### Arguments

- `base_path`: Base path to results directory (e.g., `results/sim/others`)
- `workflow_type`: Workflow type (e.g., `case1_real`)
- `failure_rate`: Failure rate directory (e.g., `fr0`)
- `--output-dir`: Optional output directory (default: `results/analysis/target_job_length/{workflow_type}/{failure_rate}`)

### Examples

#### Single Analysis (Recommended: case1_real, fr0)

```bash
# Analyze case1_real at fr0
python scripts/target_job_length_analysis.py \
    results/sim/others \
    case1_real \
    fr0
```

#### Multiple Workflow Types

```bash
# Analyze different workflow types
python scripts/target_job_length_analysis.py \
    results/sim/others \
    case2_homo \
    fr0

python scripts/target_job_length_analysis.py \
    results/sim/others \
    case3_hetero \
    fr0
```

#### Different Failure Rates

```bash
# Analyze with 25% failure rate (fr25) to see impact of failures on target job length optimization
python scripts/target_job_length_analysis.py \
    results/sim/others \
    case1_real \
    fr25 \
    --overhead-type overhead

# Compare fr0 vs fr25 to understand how failures affect time-dependent patterns
```

#### Using Makefile

```bash
# Run analysis for all workflow types (case1_real, case2_homo, case3_hetero) 
# at both fr0 and fr25 failure rates
make analyze-target-job-length
```

## Output Location

Results are saved to:
```
results/analysis/target_job_length/{overhead_type}/{workflow_type}/{failure_rate}/
```

This structure separates cross-dimensional analysis outputs from standard simulation results (`results/sim/`) and standard visualizations (`results/vis/`).

## Output Files

The script generates the following outputs in the specified output directory:

### Visualizations

1. **`throughput_vs_target_length_{overhead|nooverhead}.png`**
   - Line chart showing event throughput vs. target job length for all 16 constructions
   - Const 1 (all chained) highlighted in red
   - Const 16 (all independent) highlighted in green
   - Hybrid constructions (2-15) shown as lighter lines
   - Best hybrid for each target length marked with triangle markers
   - Helps identify which constructions perform best at different time constraints

2. **`throughput_improvement_{overhead|nooverhead}.png`**
   - Line chart showing throughput improvement percentage (relative to 1h) vs. target job length
   - Helps identify which constructions benefit most from longer job lengths
   - Negative values indicate degradation relative to 1h baseline
   - Shows how constructions scale with time constraints

3. **`network_activity_vs_target_length_{overhead|nooverhead}.png`**
   - Two-panel visualization showing network activity patterns
   - **Left panel**: Network transfer per event vs. target job length for all 16 constructions
   - **Right panel**: Remote read vs. remote write breakdown for Const 1, Const 16, and best hybrid
   - Helps identify which constructions maintain efficient network usage across time constraints
   - Shows how target job length affects remote I/O patterns

4. **`best_hybrid_comparison_{overhead|nooverhead}.png`**
   - Bar chart comparing Const 1, Const 16, and the best hybrid construction for each target job length
   - Best hybrid is identified based on event throughput
   - Shows which hybrid construction performs best at each target job length
   - Helps identify if there's a single "best" hybrid or if it varies by target length

5. **`failure_cost_analysis_{overhead|nooverhead}.png`** (fr25 only)
   - Two-panel visualization showing failure cost metrics
   - **Left panel**: Average CPU cost per failure (CPU-hours) vs. target job length
     - Y-axis shows CPU-hours wasted per failure
     - CPU-hours represent total CPU time wasted (can be > wall time due to parallel execution)
     - Note: Wall-clock hours are not shown as they match the target job length (redundant)
   - **Right panel**: Risk profile - maximum single failure cost (CPU-hours) vs. target job length
     - Shows worst-case scenario: the maximum CPU cost of a single failure at each target length
     - Critical for understanding risk exposure at different target lengths
   - Shows Const 1, Const 16, and best hybrid (each best hybrid has a unique color)
   - Reveals that longer target job lengths have much higher cost per failure
   - Critical insight: A single 24h failure wastes 24x more CPU resources than a single 1h failure
   - Only generated when failure data is present (skipped for fr0)

6. **`failure_count_analysis_{overhead|nooverhead}.png`** (fr25 only)
   - Two-panel visualization showing failure count distribution
   - **Left panel**: Number of failed jobs vs. target job length
   - **Right panel**: Actual failure rate (%) vs. target job length (with expected rate line)
   - Shows how failure counts normalize across target lengths
   - Reveals that shorter jobs have more failures but each failure is cheaper
   - Only generated when failure data is present (skipped for fr0)

### Data Tables

7. **`target_job_length_analysis_summary_{overhead|nooverhead}.csv`**
   - Comprehensive table with all metrics for all constructions across all target job lengths
   - Columns include:
     - Composition number
     - Target job length
     - Event throughput
     - Wall time per event
     - CPU time per event
     - Network transfer per event
     - CPU utilization
     - Memory occupancy
     - Total groups
     - **Failure metrics** (for fr25):
       - Total logical jobs (excluding retries)
       - Total failed jobs
       - Actual failure rate (%)
       - Total wasted CPU/wall time and network transfer
       - Average cost per failure (CPU/wall/network)
       - Maximum single failure cost (risk profile)

## Interpretation Guide

### Throughput vs. Target Job Length Plot

- **Steeper positive slopes** indicate constructions that benefit more from longer job lengths
- **Flatter lines** indicate constructions that are less sensitive to time constraints
- **Higher lines** indicate better absolute performance
- Look for hybrid constructions that maintain higher throughput across all target lengths
- Triangle markers show which hybrid is best at each target length

### Throughput Improvement Plot

- **Positive values** indicate improvement over shortest target length baseline (15m)
- **Negative values** indicate degradation relative to shortest target length baseline
- **Steeper slopes** indicate constructions that scale better with longer job lengths
- Compare Const 1 and Const 16 to see which extreme benefits more from longer jobs
- Identify hybrid constructions with optimal scaling behavior

### Network Activity Plot

- **Left panel**: Shows total network transfer patterns across all constructions
  - Lower network transfer generally indicates better efficiency
  - Look for constructions that maintain low network usage across all target lengths
- **Right panel**: Shows remote read/write breakdown for key constructions
  - Helps understand I/O patterns: more reads indicate cross-group dependencies
  - Const 1 typically has minimal remote read (all in one group)
  - Const 16 typically has more remote read (groups are independent)
  - Best hybrid should show a balanced pattern

### Best Hybrid Comparison Plot

- Shows which hybrid construction (2-15) performs best at each target job length
- Direct comparison with Const 1 and Const 16
- Helps identify if there's a single "best" hybrid or if it varies by target length
- The legend shows which construction number(s) are the best hybrid

### Failure Cost Analysis Plot (fr25 only)

- **Left panel (Average CPU Cost per Failure)**:
  - Shows average CPU-hours wasted per failure across target job lengths
  - **Key insight**: Longer target job lengths have exponentially higher cost per failure
  - Example: 24h failures waste ~24x more CPU resources than 1h failures
  - This reveals the hidden cost that throughput metrics mask
  - CPU-hours can be much higher than wall-clock hours due to parallel execution
  - Note: Wall-clock hours are not shown as they directly match the target job length (redundant)

- **Right panel (Risk Profile - Max Single Failure Cost)**:
  - Shows the maximum CPU cost of a single failure at each target job length
  - Represents the worst-case scenario risk exposure
  - Critical for resource planning and cost estimation
  - Helps understand the risk profile: many small failures vs. few large failures
  - Shows the potential maximum loss from a single failure event

### Failure Count Analysis Plot (fr25 only)

- **Left panel (Failure Count)**:
  - Shows absolute number of failed jobs vs. target job length
  - **Key insight**: Shorter jobs have many more failures (more jobs = more failure opportunities)
  - Longer jobs have fewer failures but each failure is much more expensive
  - The total cost may be similar, but the risk profile is very different

- **Right panel (Actual Failure Rate)**:
  - Shows the actual failure rate percentage vs. target job length
  - Should be close to the expected failure rate (e.g., 25% for fr25)
  - Helps verify that failures are being properly tracked
  - Small variations are expected due to random sampling

## Key Insights to Look For

1. **Time Constraint Sensitivity**: Do hybrid constructions show better performance across all target lengths?
2. **Consistency**: Is there a single best hybrid construction, or does it vary by target job length?
3. **Performance Gap**: How much better are hybrids compared to Const 1 and Const 16?
4. **Scaling Behavior**: At what target length do differences become most pronounced?
5. **Network Efficiency**: Do hybrid constructions maintain better network efficiency across time constraints?
6. **I/O Patterns**: How do remote read/write patterns differ between extremes and hybrids across target lengths?
7. **Optimal Range**: Is there a "sweet spot" target job length where hybrid constructions show maximum benefit?
8. **Failure Impact**: How do failures (fr25) affect the relationship between target job length and optimal construction?
   - Does the best hybrid change under failure conditions?
   - Are some constructions more resilient to failures at certain target lengths?
   - Does target job length have a more significant impact on throughput when failures are present?
9. **Failure Cost Analysis** (fr25): Critical insights revealed by failure cost metrics:
   - **Cost per failure**: Longer target job lengths have much higher cost per failure (24h failures waste 24x more resources than 1h failures)
   - **Risk profile**: Maximum single failure cost shows the risk exposure at different target lengths
   - **Failure normalization**: Shorter jobs have more failures but each failure is cheaper, while longer jobs have fewer failures but each is much more expensive
   - **Throughput masking**: The throughput metric normalizes by total time (including retries), hiding the real cost difference between short and long job failures

## Recommended Configuration

For comprehensive analysis, use:
- **Workflow Types**: All workflow types (`case1_real`, `case2_homo`, `case3_hetero`) - the Makefile target processes all automatically
- **Failure Rates**: Both `fr0` (0% - clean baseline) and `fr25` (25% - high failure rate) - the Makefile target processes both automatically

**Baseline Analysis (fr0)**:
- Provides a clear view of how time constraints affect hybrid construction benefits across different workflow characteristics without the complexity of failures
- Useful for understanding fundamental time-dependent patterns

**Failure Impact Analysis (fr25)**:
- Shows how failures interact with target job length constraints
- May reveal different optimal constructions under failure conditions
- Helps identify which constructions are more resilient to failures at different time scales

**Note**: The Makefile target `analyze-target-job-length` automatically runs the analysis for all configured workflow types at both fr0 and fr25. To analyze a specific workflow type or failure rate, use the command line interface directly.

## Requirements

- Python 3.x
- Required packages: `matplotlib`, `numpy`, `pandas`, `seaborn`
- Simulation result JSON files organized in the hierarchical structure:
  ```
  results/sim/others/
    {workflow_type}/
      {target_job_length}/
        {failure_rate}/
          *.json
  ```

## Notes

- The script automatically processes target job length directories (15m, 30m, 1h, 2h, 4h, 8h, 12h, 24h)
- Missing directories are skipped with a warning
- The best hybrid is identified using event throughput as the primary metric
- The analysis focuses on comparing extremes (Const 1, Const 16) with the best hybrid, not all 16 constructions in every visualization
- Target job lengths are sorted by duration (15m, 30m, 1h, 2h, 4h, 8h, 12h, 24h) for consistent visualization
