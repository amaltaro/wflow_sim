# Workflow Simulation Usage Guide

This guide explains how to use the workflow simulation functionality to execute and analyze workflow compositions with group-based job scheduling.

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
- **`workflow_metrics.py`** - Metrics from existing simulation results

**Data Flow:** `workflow_runner.py` → `workflow_simulator.py` → `workflow_metrics.py` → Results

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
- **Wallclock Constraints**: Each job respects target wallclock time limits

## Quick Start

### Command Line Usage

The easiest way to run workflow simulations is using the command line interface:

```bash
# Basic usage with default settings
python src/workflow_runner.py

# Custom wallclock time and job slots
python src/workflow_runner.py --target-wallclock-time 3600 --max-job-slots 10

# Specify custom workflow file
python src/workflow_runner.py --input-workflow-path templates/3tasks/seq/3tasks_001.json

# Simulation only (no metrics)
python src/workflow_simulator.py --target-wallclock-time 1800

# Show all available options
python src/workflow_runner.py --help
```

**Output Structure**: Results are automatically saved to the `results/` directory with the same structure as the input file (excluding the `templates/` prefix).

### Python API Usage

```python
from src.workflow_runner import WorkflowRunner, ResourceConfig

# Configure resources
resource_config = ResourceConfig(
    target_wallclock_time=43200.0,  # 12 hours
    max_job_slots=-1  # Infinite slots
)

# Create runner and execute
runner = WorkflowRunner(resource_config)
results = runner.run_workflow('templates/3tasks_composition_001.json')

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
- `--input-workflow-path`: Path to input workflow JSON file (default: templates/3tasks_composition_001.json)

### Usage Examples

```bash
# Show help for all options
python src/workflow_runner.py --help
python src/workflow_simulator.py --help
```

### Output Structure

Results are automatically saved to the `results/` directory with the same structure as the input file:

- **Input**: `templates/3tasks/seq/workflow.json` → **Output**: `results/3tasks/seq/workflow.json`
- **Input**: `templates/workflow.json` → **Output**: `results/workflow.json`
- **Input**: `custom/path/workflow.json` → **Output**: `results/custom/path/workflow.json`

The system automatically creates necessary directories and preserves the file structure while removing the `templates/` prefix for cleaner organization.

## Detailed Usage

### Metrics Calculation

```python
from src.workflow_metrics import WorkflowMetricsCalculator

calculator = WorkflowMetricsCalculator()

# Main metrics
metrics = calculator.calculate_metrics(simulation_result)

# Job statistics
job_stats = calculator.calculate_job_statistics(simulation_result)

# Group statistics
group_stats = calculator.calculate_group_statistics(simulation_result)
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

# Create simulator
simulator = WorkflowSimulator(ResourceConfig())

# Run simulation
result = simulator.simulate_workflow('templates/3tasks_composition_001.json')

# Print simulation summary
simulator.print_simulation_summary(result)

# Save results
simulator.write_simulation_result(result, 'simulation_results.json')
```

### Complete Workflow Analysis

```python
from src.workflow_runner import WorkflowRunner

# Create runner
runner = WorkflowRunner(resource_config)

# Run complete analysis (simulation + metrics)
results = runner.run_workflow('templates/3tasks_composition_001.json')

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

The integrated metrics calculator provides:

- **Performance Metrics**: Throughput, efficiency, success rate
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

