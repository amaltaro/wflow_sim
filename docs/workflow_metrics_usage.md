# Workflow Metrics Calculator

The `WorkflowMetricsCalculator` class provides comprehensive workflow-level metrics calculation by aggregating job-level metrics from simulation results, supporting performance analysis and comparison of different workflow compositions.

## Features

- **Workflow-Level Aggregation**: Aggregates job-level metrics into workflow insights
- **Job Metrics Integration**: Uses `JobMetricsCalculator` for detailed job analysis
- **Comprehensive Metrics**: Calculate execution times, resource usage, throughput, and efficiency measures
- **Group-Based Analysis**: Support for group-level metrics and job scaling calculations
- **Flexible Output**: Print metrics to console or save to JSON files
- **Extensible Design**: Easy to add new metrics and analysis methods

## Key Metrics Calculated

### Core Workflow Metrics
- **Total Tasksets**: Number of individual computational units
- **Total Groups**: Number of groups (job submission units)
- **Total Jobs**: Number of grid jobs (based on event scaling)
- **Execution Time**: Total computational time
- **Wall Time**: Real elapsed time
- **Wall Time per Event**: Average wall time per event processed
- **CPU Utilization**: Ratio of CPU used vs allocated (0.0 to 1.0)
- **Memory Occupancy**: Ratio of memory used vs allocated (0.0 to 1.0)
- **Throughput**: Events processed per second
- **Success Rate**: Percentage of successful executions

### Aggregated Job Metrics
- **Total CPU Time**: Sum of all job CPU times
- **Total Write Local**: Sum of local disk writes across all jobs
- **Total Write Remote**: Sum of remote storage writes across all jobs
- **Total Read Remote**: Sum of remote storage reads across all jobs
- **Total Read Local**: Sum of local disk reads across all jobs (within same group)
- **Total Network Transfer**: Sum of complete network transfers across all jobs (remote writes + remote reads)

### Per-Event Metrics
- **Total Write Local per Event**: Average local disk writes per event processed
- **Total Write Remote per Event**: Average remote storage writes per event processed
- **Total Read Remote per Event**: Average remote storage reads per event processed
- **Total Read Local per Event**: Average local disk reads per event processed

### Group-Level Metrics
- **Group Execution Time**: Time for each group to complete
- **Resource Usage**: CPU, memory, storage, and network usage per group
- **Job Count**: Number of jobs per group based on event scaling
- **Taskset Details**: Individual taskset performance within groups

## Usage

### Basic Usage

```python
from src.workflow_metrics import WorkflowMetricsCalculator
from src.job_metrics import JobMetricsCalculator

# Run simulation first
from src.workflow_runner import WorkflowRunner
runner = WorkflowRunner()
results = runner.run_workflow('templates/3tasks_composition_001.json')

# Create calculator and calculate metrics from simulation results
calculator = WorkflowMetricsCalculator()
metrics = calculator.calculate_metrics(results['simulation_result'])

# Print metrics
calculator.print_metrics()

# Calculate job statistics (includes aggregated job metrics)
job_stats = calculator.calculate_job_statistics(results['simulation_result'])
print(f"Total CPU Time: {job_stats['total_cpu_time']:.2f}s")
print(f"Total Network Transfer: {job_stats['total_network_transfer_mb']:.2f} MB")

# Save to file
calculator.write_metrics_to_file('metrics_output.json')
```

### Advanced Usage

```python
# Get metrics summary for further processing
summary = calculator.get_metrics_summary()

# Access individual workflow metrics
print(f"Total execution time: {metrics.total_turnaround_time}")
print(f"Wall time per event: {metrics.wall_time_per_event:.6f}s/event")
print(f"Throughput: {metrics.throughput} events/second")

# Access aggregated job metrics
job_stats = calculator.calculate_job_statistics(results['simulation_result'])
print(f"Total CPU Time: {job_stats['total_cpu_time']:.2f}s")
print(f"Total Write Local: {job_stats['total_write_local_mb']:.2f} MB")
print(f"Total Write Remote: {job_stats['total_write_remote_mb']:.2f} MB")
print(f"Total Read Remote: {job_stats['total_read_remote_mb']:.2f} MB")
print(f"Total Read Local: {job_stats['total_read_local_mb']:.2f} MB")

# Access resource utilization metrics
if metrics.resource_utilization:
    print(f"CPU Utilization: {metrics.resource_utilization.cpu_utilization:.2%}")
    print(f"Memory Occupancy: {metrics.resource_utilization.memory_occupancy:.2%}")

# Access per-event metrics
print(f"Write Local per Event: {metrics.total_write_local_mb_per_event:.6f} MB/event")
print(f"Write Remote per Event: {metrics.total_write_remote_mb_per_event:.6f} MB/event")
print(f"Read Remote per Event: {metrics.total_read_remote_mb_per_event:.6f} MB/event")
print(f"Read Local per Event: {metrics.total_read_local_mb_per_event:.6f} MB/event")

# Access group-level details
for group in metrics.group_metrics:
    print(f"Group {group.group_id}: {group.job_count} jobs")
    print(f"  Execution time: {group.total_execution_time}s")
    print(f"  CPU usage: {group.total_resource_usage.cpu_usage}%")
```

## Architecture Integration

The `WorkflowMetricsCalculator` integrates with the job metrics system:

- **Job Metrics Integration**: Uses `JobMetricsCalculator` internally for job-level calculations
- **Aggregation Layer**: Aggregates job metrics into workflow-level insights
- **Separation of Concerns**: Focuses on workflow-level analysis, delegates job-level calculations
- **Single Source of Truth**: All workflow metrics calculated in one place

## Workflow Data Format

The calculator expects simulation results from `WorkflowSimulator`:

```json
{
  "Comments": "Workflow description",
  "NumTasks": 3,
  "RequestNumEvents": 1000000,
  "Taskset1": {
    "Memory": 2000,
    "Multicore": 1,
    "TimePerEvent": 10,
    "SizePerEvent": 200,
    "GroupName": "group_1",
    "GroupInputEvents": 1000
  },
  "Taskset2": {
    "Memory": 4000,
    "Multicore": 2,
    "TimePerEvent": 20,
    "SizePerEvent": 300,
    "InputTaskset": "Taskset1",
    "GroupName": "group_1",
    "GroupInputEvents": 1000
  },
  "CompositionNumber": 1
}
```

## Job Metrics Integration

The calculator integrates job-level metrics through `JobMetricsCalculator`:

### Job-Level Metrics Aggregated
- **CPU Time**: Sum of `time_per_event × input_events × multicore` across all jobs
- **Local I/O**: Sum of local disk writes across all jobs
- **Remote I/O**: Sum of remote storage writes across all jobs  
- **Network Transfer**: Sum of complete network transfers across all jobs (remote writes + remote reads)
- **Remote Read**: Sum of cross-group data reads across all jobs
- **Local Read**: Sum of within-group data reads across all jobs

### Calculation Flow
1. **Job Creation**: `WorkflowSimulator` creates jobs with basic metrics
2. **Job Metrics**: `JobMetricsCalculator` calculates detailed job metrics
3. **Workflow Aggregation**: `WorkflowMetricsCalculator` aggregates job metrics
4. **Final Output**: Combined workflow and job statistics

## Output Formats

### Console Output
```
============================================================
WORKFLOW EXECUTION METRICS
============================================================
Workflow ID: workflow_001
Composition Number: 1
Total Tasksets: 3
Total Groups: 1
Total Jobs: 1000
Total Execution Time: 50000.00 seconds
Total Wall Time: 50000.00 seconds
Wall Time per Event: 0.050000s/event
CPU Utilization: 87.50%
Memory Occupancy: 81.25%
Throughput: 20.00 events/second
Success Rate: 1.00

⚡ JOB STATISTICS:
  Total CPU Time: 120000.00s
  Total Write Local: 20000.00 MB
  Total Write Remote: 15000.00 MB
  Total Read Remote: 0.00 MB
  Total Read Local: 5000.00 MB
  Total Network Transfer: 15000.00 MB
```

### JSON Output
```json
{
  "workflow_id": "workflow_001",
  "composition_number": 1,
  "total_tasksets": 3,
  "total_groups": 1,
  "total_jobs": 1000,
  "total_execution_time": 50000.0,
  "total_wall_time": 50000.0,
  "throughput": 20.0,
  "success_rate": 1.0,
  "group_metrics": [...]
}
```

## Testing

Run the test suite to verify functionality:

```bash
pytest tests/test_workflow_metrics.py -v
```

## Examples

See `examples/metrics_example.py` for a complete working example using the 3-taskset template.

## Extending the Metrics

To add new metrics, extend the `WorkflowMetrics` dataclass and add calculation methods to the `WorkflowMetricsCalculator` class:

```python
@dataclass
class WorkflowMetrics:
    # ... existing fields ...
    custom_metric: float = 0.0

class WorkflowMetricsCalculator:
    def _calculate_custom_metric(self) -> float:
        """Calculate custom metric."""
        # Implementation here
        return 0.0
```
