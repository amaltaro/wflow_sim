# Workflow Simulation Usage Guide

This guide explains how to use DAGFlowSim's workflow simulation functionality to execute and analyze workflow compositions with group-based job scheduling.

## Overview

The workflow simulation system provides:

- **DAG Execution**: Follows workflow dependencies with sequential taskset execution within groups
- **Group-Based Job Scheduling**: Creates jobs at the group level based on event scaling
- **Wallclock Time Constraints**: Respects target job wallclock time limits (12h default)
- **Parallel Group Execution**: Independent groups can execute in parallel
- **Comprehensive Metrics**: Detailed performance analysis and resource utilization
- **Batch Job Logging**: Complete logging of job creation and execution

## Module Usage

**Choose your approach:**

- **`workflow_runner.py`** - Complete analysis (simulation + metrics) with one call
- **`workflow_simulator.py`** - Just simulation results for custom analysis
- **`job_metrics.py`** - Job-level metrics calculation (CPU, I/O, network)
- **`workflow_metrics.py`** - Workflow-level metrics aggregation

**Data Flow:** `workflow_runner.py` → `workflow_simulator.py` → `job_metrics.py` → `workflow_metrics.py` → Results

## Key Concepts

### Workflow Structure
- **Workflow**: Complete Directed Acyclic Graph (DAG) containing multiple groups
- **Group**: Set of tasksets that execute sequentially, materialized as grid jobs
- **Taskset**: Individual computational unit with defined inputs/outputs
- **Job**: Grid job created at group level, scaled based on event requirements

### Execution Model
- **Sequential Within Groups**: Tasksets in the same group execute one after another
- **Parallel Between Groups**: Independent groups can run simultaneously, if dependency allows
- **Job Scaling**: Number of jobs = ceil(RequestNumEvents / GroupInputEvents)
- **Wallclock Constraints**: Each job respects target wallclock time limits (batch size calculation based on target, actual wallclock may exceed target due to overhead)
- **Job Overhead**: Realistic overhead accounting including taskset setup (60s per taskset) and data transfer operations (1s per 100MB/s)

## Quick Start

### Command Line Usage

The easiest way to run workflow simulations is using the command line interface:

```bash
# Basic usage with default settings
python src/workflow_runner.py

# Custom wallclock time and job slots
python src/workflow_runner.py --target-wallclock-time 3600 --max-job-slots 10

# Specify custom workflow file
python src/workflow_runner.py --input-workflow-path templates/sequential/3tasks/3tasks_composition_001.json

# Simulation only (no metrics)
python src/workflow_simulator.py --target-wallclock-time 1800

# Show all available options
python src/workflow_runner.py --help
```

**Output Structure**: Results are automatically saved to the `results/sim/` directory with the same structure as the input file (excluding the `templates/` prefix).

### Python API Usage

```python
from src.workflow_runner import WorkflowRunner, ResourceConfig

# Configure resources
resource_config = ResourceConfig(
    target_wallclock_time=43200.0,  # 12 hours
    max_job_slots=-1  # Infinite slots
)

# Create runner and execute (failure_rate and data_transfer_rate passed explicitly)
runner = WorkflowRunner(
    resource_config,
    job_failure_rate=0,
    data_transfer_rate_mb_per_s=100.0
)
results = runner.run_workflow('templates/others/seq_real/seq_real_const_001.json')

# Print results
runner.print_complete_summary(results)
```

### Running the Example

```bash
# Command line example
python src/workflow_runner.py --target-wallclock-time 1800

# Python API example
python examples/workflow_simulation_example.py
```

## Command Line Interface

Both `workflow_runner.py` and `workflow_simulator.py` support comprehensive command line arguments:

### Available Arguments

- `--target-wallclock-time`: Target wallclock time in seconds (default: 43200 = 12 hours)
- `--max-job-slots`: Maximum number of job slots (-1 for infinite, default: -1)
- `--input-workflow-path`: Path to input workflow JSON file (default: templates/others/seq_real/seq_real_const_001.json)
- `--failure-rate`: Job failure rate as percentage (0-99, default: 0)
- `--data-transfer-rate`: Network data transfer rate in MB/s (default: 100.0)
- `--seed`: RNG seed for stochastic job failures (default: 42)

### Usage Examples

```bash
# Show help for all options
python src/workflow_runner.py --help
python src/workflow_simulator.py --help

# Reproducible run with an explicit seed
python -m src.workflow_runner \
  --input-workflow-path templates/others/seq_real/seq_real_const_001.json \
  --failure-rate 5 \
  --seed 7
```

### Output Structure

Results are automatically saved to the `results/sim/` directory with the same structure as the input file:

- **Input**: `templates/3tasks/seq/workflow.json` → **Output**: `results/sim/3tasks/seq/workflow.json`
- **Input**: `templates/workflow.json` → **Output**: `results/sim/workflow.json`
- **Input**: `custom/path/workflow.json` → **Output**: `results/sim/custom/path/workflow.json`

The system automatically creates necessary directories and preserves the file structure while removing the `templates/` prefix for cleaner organization.

## Detailed Usage

### Metrics Calculation

```python
from src.workflow_metrics import WorkflowMetricsCalculator
from src.job_metrics import JobMetricsCalculator

# Workflow-level metrics
workflow_calculator = WorkflowMetricsCalculator()
metrics = workflow_calculator.calculate_metrics(simulation_result)

# Job-level metrics
job_calculator = JobMetricsCalculator()
job_stats = job_calculator.calculate_job_statistics(simulation_result.jobs)

# Group statistics
group_stats = workflow_calculator.calculate_group_statistics(simulation_result)
```

### Resource Configuration

```python
from src.workflow_simulator import ResourceConfig

# Default configuration (12h wallclock, infinite slots)
config = ResourceConfig()

# Custom configuration
config = ResourceConfig(
    target_wallclock_time=21600.0,  # 6 hours
    max_job_slots=100,              # Limit to 100 concurrent jobs
    cpu_per_slot=2,                 # 2 CPUs per job slot
    memory_per_slot=2000           # 2GB memory per job slot
)
```

### Workflow Simulation Only

```python
from src.workflow_simulator import WorkflowSimulator, ResourceConfig

# Create simulator (failure_rate and data_transfer_rate passed explicitly)
simulator = WorkflowSimulator(
    ResourceConfig(),
    job_failure_rate=0,
    data_transfer_rate_mb_per_s=100.0
)

# Run simulation
result = simulator.simulate_workflow('templates/others/seq_real/seq_real_const_001.json')

# Print simulation summary
simulator.print_simulation_summary(result)

# Save results
simulator.write_simulation_result(result, 'simulation_results.json')
```

### Complete Workflow Analysis

```python
from src.workflow_runner import WorkflowRunner

# Create runner (failure_rate and data_transfer_rate passed explicitly)
runner = WorkflowRunner(
    resource_config,
    job_failure_rate=0,
    data_transfer_rate_mb_per_s=100.0
)

# Run complete analysis (simulation + metrics)
results = runner.run_workflow('templates/others/seq_real/seq_real_const_001.json')

# Access individual components
simulation = results['simulation_result']
metrics = results['metrics']

# Print complete summary
runner.print_complete_summary(results)

# Save complete results
runner.write_complete_results(results, 'complete_results.json')
```

## Workflow JSON Format

The simulation expects workflow JSON files with the following structure:

```json
{
  "Comments": "Workflow description",
  "NumTasks": 3,
  "RequestNumEvents": 1000000,
  "Taskset1": {
    "GroupName": "group_5",
    "GroupInputEvents": 1080,
    "TimePerEvent": 10,
    "Memory": 2000,
    "Multicore": 1,
    "SizePerEvent": 200,
    "InputTaskset": null,
    "ScramArch": ["el9_amd64_gcc11"],
    "RequiresGPU": "forbidden",
    "KeepOutput": false
  },
  "Taskset2": {
    "GroupName": "group_5",
    "GroupInputEvents": 1080,
    "TimePerEvent": 20,
    "Memory": 4000,
    "Multicore": 2,
    "SizePerEvent": 300,
    "InputTaskset": "Taskset1",
    "ScramArch": ["el9_amd64_gcc11"],
    "RequiresGPU": "forbidden",
    "KeepOutput": true
  },
  "CompositionNumber": 1
}
```

### Required Fields

- **RequestNumEvents**: Total number of events to process
- **TasksetX**: Individual taskset definitions
- **GroupName**: Groups tasksets together for job creation
- **GroupInputEvents**: Events per job for this group
- **TimePerEvent**: Processing time per event (seconds) for a given taskset
- **Memory**: Memory requirement (MB) for a given taskset
- **Multicore**: Number of CPU cores for a given taskset
- **InputTaskset**: Dependency on another taskset (null for first taskset)

## Output and Results

### Simulation Results

The simulation provides detailed information about:

- **Job Creation**: Number of jobs per group based on event scaling
- **Execution Timeline**: Sequential execution of jobs within groups
- **Resource Usage**: CPU, memory, and storage requirements
- **Wallclock Time**: Actual job execution times meeting constraints
- **Batch Sizes**: Events processed per job

### Metrics Analysis

The integrated metrics system provides:

- **Job-Level Metrics**: CPU time, I/O operations, complete network transfers per job
- **Workflow-Level Metrics**: Throughput, efficiency, success rate
- **Resource Utilization**: CPU, memory, storage usage patterns
- **Timing Analysis**: Execution times, wall times, queue times
- **Group Statistics**: Per-group performance breakdown

### Example Output

```
================================================================================
COMPLETE WORKFLOW EXECUTION SUMMARY
================================================================================

📊 SIMULATION RESULTS:
  Workflow ID: unknown
  Composition: 1
  Total Events: 1,000,000
  Total Groups: 1
  Total Jobs: 926
  Total Wall Time: 43200.00s (12.00h)

📈 PERFORMANCE METRICS:
  Resource Efficiency: 0.06
  Throughput: 23.15 events/second
  Success Rate: 1.00
  Total Execution Time: 43200.00s

🏗️  GROUP BREAKDOWN:
  Group group_5:
    Jobs: 926
    Events per Job: 1,080
    Wall Time per Job: 43200.00s
    Total Execution Time: 43200.00s
    Tasksets: 3
      Taskset1: 10s/event, 2000MB, 1 cores
      Taskset2: 20s/event, 4000MB, 2 cores
      Taskset3: 10s/event, 3000MB, 2 cores

⚡ JOB STATISTICS:
  Average Job Wall Time: 43200.00s
  Min Job Wall Time: 43200.00s
  Max Job Wall Time: 43200.00s
  Average Batch Size: 1080 events
  Min Batch Size: 1080 events
  Max Batch Size: 1080 events
  Total CPU Used Time: 60000000.00s
  Total CPU Allocated Time: 70000000.00s
  Total Write Local: 537109.38 MB
  Total Write Remote: 341796.88 MB
  Total Read Remote: 0.00 MB
  Total Network Transfer: 341796.88 MB
```

## Advanced Features

### Custom Wallclock Time Constraints

```python
# 6-hour job limit
config = ResourceConfig(target_wallclock_time=21600.0)

# 24-hour job limit
config = ResourceConfig(target_wallclock_time=86400.0)
```

### Limited Job Slots

```python
# Limit to 50 concurrent jobs
config = ResourceConfig(max_job_slots=50)
```

### Batch Size Optimization

The simulator automatically calculates optimal batch sizes to meet wallclock constraints:

- Calculates time per event for each group
- Determines maximum events that fit in target wallclock time
- Creates jobs with appropriate batch sizes
- Logs each job's batch size and wallclock time

**Important Note**: The batch size calculation is based on the target wallclock time and does not include overhead. This means:
- Batch sizes remain unchanged regardless of overhead
- Actual job wallclock time may exceed the target due to overhead
- Overhead is tracked separately in `job_overhead` metric
- CPU metrics (`total_cpu_used_time`, `total_cpu_allocated_time`) include overhead for realistic resource accounting

### Job Overhead Considerations

The simulation includes realistic job overhead to provide more accurate resource accounting:

- **Taskset Overhead**: 60 seconds per taskset (default: `TASKSET_OVERHEAD_SECONDS = 60.0`)
- **Data Transfer Overhead**:
  - Remote read: 1 second per 100MB/s of data (default: 100.0 MB/s, configurable via `--data-transfer-rate`)
  - Remote write: 1 second per 100MB/s of data
- **Total Overhead**: Sum of all overhead components, added to CPU metrics

The overhead is automatically calculated and included in:
- `job_overhead_secs` / `job_overhead_cpu_time`: Per-job overhead (wallclock and CPU time)
- `total_cpu_used_time`: CPU time includes overhead
- `total_cpu_allocated_time`: Allocated CPU time includes overhead
- `total_execution_time`: Execution time excludes overhead (pure computational time)

When results are written via `WorkflowRunner.write_complete_results()`, workflow-level totals (`total_job_overhead_secs`, `total_job_overhead_cpu_time`) appear in `metrics`, and a top-level **`simulation_stats`** object provides distribution statistics (mean, std, median, min, max, n) over all jobs for job overhead, events per job (`batch_size`), and I/O (total_write_local_mb, total_write_remote_mb, total_read_local_mb, total_read_remote_mb), for use in visualizations.

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the `src` directory is in your Python path
2. **File Not Found**: Check that workflow JSON files exist and are accessible
3. **Invalid JSON**: Validate workflow JSON format before simulation
4. **Resource Constraints**: Adjust wallclock time or job slots if needed

### Debugging

Enable detailed logging to see job creation and execution details:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Performance Considerations

- Large workflows with many jobs may take time to simulate
- Consider reducing logging verbosity for large simulations
- Use appropriate wallclock time constraints for realistic results

## API Reference

### WorkflowSimulator

Main simulation engine class.

**Methods:**
- `simulate_workflow(workflow_filepath)`: Run simulation from JSON file
- `print_simulation_summary(result)`: Print results
- `write_simulation_result(result, filepath)`: Save results

### WorkflowRunner

High-level interface combining simulation and metrics.

**Methods:**
- `run_workflow(workflow_filepath)`: Complete analysis from JSON file
- `print_complete_summary(results)`: Print comprehensive results
- `write_complete_results(results, filepath)`: Save complete results

### ResourceConfig

Configuration for simulation resources.

**Parameters:**
- `target_wallclock_time`: Target job wallclock time (seconds)
- `max_job_slots`: Maximum concurrent jobs (-1 for infinite)
- `cpu_per_slot`: CPUs per job slot
- `memory_per_slot`: Memory per job slot (MB)

