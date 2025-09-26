# Workflow Metrics Calculator

The `WorkflowMetricsCalculator` class provides comprehensive metrics calculation for workflow simulation results, supporting performance analysis and comparison of different workflow compositions.

## Features

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
- **Resource Efficiency**: Overall resource utilization efficiency
- **Throughput**: Events processed per second
- **Success Rate**: Percentage of successful executions

### Group-Level Metrics
- **Group Execution Time**: Time for each group to complete
- **Resource Usage**: CPU, memory, storage, and network usage per group
- **Job Count**: Number of jobs per group based on event scaling
- **Taskset Details**: Individual taskset performance within groups

## Usage

### Basic Usage

```python
from workflow_metrics import WorkflowMetricsCalculator
import json

# Load workflow data
with open('workflow_template.json', 'r') as f:
    workflow_data = json.load(f)

# Create calculator
calculator = WorkflowMetricsCalculator(workflow_data)

# Calculate metrics
metrics = calculator.calculate_metrics()

# Print metrics
calculator.print_metrics(detailed=True)

# Save to file
calculator.write_metrics_to_file('metrics_output.json')
```

### Advanced Usage

```python
# Get metrics summary for further processing
summary = calculator.get_metrics_summary()

# Access individual metrics
print(f"Total execution time: {metrics.total_execution_time}")
print(f"Resource efficiency: {metrics.resource_efficiency}")
print(f"Throughput: {metrics.throughput} events/second")

# Access group-level details
for group in metrics.group_metrics:
    print(f"Group {group.group_id}: {group.job_count} jobs")
    print(f"  Execution time: {group.total_execution_time}s")
    print(f"  CPU usage: {group.total_resource_usage.cpu_usage}%")
```

## Workflow Data Format

The calculator expects workflow data in the following JSON format:

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

## Job Scaling Calculation

The calculator automatically determines the number of jobs per group based on:

```
jobs_per_group = max(1, RequestNumEvents / GroupInputEvents)
```

This ensures that:
- Each group gets at least 1 job
- Job count scales with the ratio of requested to actual input events
- Resource requirements are properly distributed across jobs

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
Resource Efficiency: 0.85
Throughput: 20.00 events/second
Success Rate: 1.00
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
  "resource_efficiency": 0.85,
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
