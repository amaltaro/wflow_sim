# Job Metrics Calculator

The `JobMetricsCalculator` class provides job-level metrics calculation for workflow simulation, handling individual job resource usage, I/O operations, and performance metrics.

## Overview

This module is responsible for calculating detailed metrics for individual jobs within a workflow simulation, including:

- **CPU Time Calculations**: Based on taskset processing time and multicore usage
- **I/O Operations**: Local and remote data write operations
- **Network Transfers**: Complete network data transfer (remote writes + remote reads)
- **Remote Reads**: Data read from shared storage across group boundaries

## Key Features

- **Job-Level Granularity**: Calculate metrics for individual jobs within groups
- **I/O Classification**: Distinguish between local and remote data operations
- **Cross-Group Dependencies**: Handle remote reads from different groups
- **Resource Aggregation**: Sum metrics across multiple jobs for analysis
- **Unit Conversion**: Automatic conversion from KB to MB for storage metrics

## Architecture

The `JobMetricsCalculator` is designed with clear separation of concerns:

- **Single Responsibility**: Only handles job-level metric calculations
- **Input Validation**: Validates taskset data and batch sizes
- **Flexible Integration**: Works with any workflow simulation results
- **No Dependencies**: Pure calculation logic without external dependencies

## Job Metrics Calculated

### CPU Time Metrics
- **Total CPU Used Time**: Actual CPU time used from event processing including overhead - `sum(time_per_event × input_events × multicore)` for each taskset + `job_overhead`
- **Total CPU Allocated Time**: CPU time allocated for the whole job including overhead - `(total_execution_time + job_overhead) × max_multicore`, where total execution time is the sum of all taskset execution times and max_multicore is the maximum multicore setting among all tasksets in the job
- **Total Execution Time**: Total sequential execution time for all tasksets in a job - `sum(time_per_event × batch_size)` for each taskset in the job (excluding overhead)

Note: All tasksets in a job execute sequentially and share the same allocated resources (max cores needed), so allocated time represents the actual resource reservation for the job.

### Job Overhead Metrics
- **Job Overhead**: Realistic overhead time accounting for job setup and data transfer operations
  - **Taskset Overhead**: 60 seconds per taskset in the job (default: `TASKSET_OVERHEAD_SECONDS = 60.0`)
  - **Remote Read Overhead**: Data transfer time for remote reads - `total_read_remote_mb / 100.0` seconds (based on 100 MB/s transfer rate)
  - **Remote Write Overhead**: Data transfer time for remote writes - `total_write_remote_mb / 100.0` seconds (based on 100 MB/s transfer rate)
  - **Total Overhead**: `taskset_overhead + remote_read_overhead + remote_write_overhead`

The overhead is automatically added to both `total_cpu_used_time` and `total_cpu_allocated_time` to provide realistic resource accounting. The batch size calculation remains unchanged (based on target wallclock time), meaning actual job wallclock time may exceed the target due to overhead.

### I/O Metrics
- **Total Write Local MB**: All data written to local disk (regardless of `keep_output` flag)
- **Total Write Remote MB**: Data written to shared storage (explicit `keep_output=true` OR input for other groups)
- **Total Read Remote MB**: Data read from shared storage (only for first taskset with cross-group input)
- **Total Network Transfer MB**: Complete network data transfer (remote writes + remote reads)

### Calculation Logic

#### Local vs Remote Write Classification
```python
# All data is written locally first
total_write_local_mb += write_mb

# Remote write if:
# 1. keep_output=True (explicitly marked for remote storage), OR
# 2. This taskset is an input taskset for another group
is_remote_write = (taskset.keep_output or 
                   taskset.taskset_id in input_tasksets_for_other_groups)
```

#### Remote Read Logic
```python
# Only consider first taskset of group for remote read
# Only if input taskset is from a different group
if input_taskset_size_per_event is not None:
    total_read_remote_mb = (input_taskset_size_per_event * batch_size) / 1024.0
```

#### Network Transfer Calculation
```python
# Network transfer includes both remote write and remote read
total_network_transfer_mb = total_write_remote_mb + total_read_remote_mb
```

## Usage

### Basic Usage

```python
from src.job_metrics import JobMetricsCalculator, JobMetrics

# Create calculator
calculator = JobMetricsCalculator()

# Calculate metrics for a single job
job_metrics = calculator.calculate_job_metrics(
    tasksets=group.tasksets,
    batch_size=1000,
    input_tasksets_for_other_groups={'Taskset1'},
    input_taskset_size_per_event=200  # KB
)

# Access individual metrics
print(f"CPU Used Time: {job_metrics.total_cpu_used_time:.2f}s")
print(f"CPU Allocated Time: {job_metrics.total_cpu_allocated_time:.2f}s")
print(f"Execution Time: {job_metrics.total_execution_time:.2f}s")
print(f"Job Overhead: {job_metrics.job_overhead:.2f}s")
print(f"Local Write: {job_metrics.total_write_local_mb:.2f} MB")
print(f"Remote Write: {job_metrics.total_write_remote_mb:.2f} MB")
print(f"Remote Read: {job_metrics.total_read_remote_mb:.2f} MB")
```

### Aggregating Across Multiple Jobs

Note: Aggregation across multiple jobs is now handled by `WorkflowMetricsCalculator._aggregate_job_metrics()`. For direct job-level metrics calculation, use the methods on individual `JobMetrics` objects or aggregate them manually.

# Access aggregated metrics
print(f"Total CPU Used Time: {job_stats['total_cpu_used_time']:.2f}s")
print(f"Total CPU Allocated Time: {job_stats['total_cpu_allocated_time']:.2f}s")
print(f"Total Local Write: {job_stats['total_write_local_mb']:.2f} MB")
print(f"Total Remote Write: {job_stats['total_write_remote_mb']:.2f} MB")
```

### Integration with Workflow Simulation

```python
from src.workflow_simulator import WorkflowSimulator
from src.job_metrics import JobMetricsCalculator

# Run simulation (failure_rate and data_transfer_rate from parser or pass explicitly)
simulator = WorkflowSimulator(failure_rate=0, data_transfer_rate_mb_per_s=100.0)
result = simulator.simulate_workflow('workflow.json')

# Calculate job statistics
from src.workflow_metrics import WorkflowMetricsCalculator

metrics_calculator = WorkflowMetricsCalculator()
job_stats = metrics_calculator.calculate_job_statistics(result)

# Display results
print(f"Total Jobs: {job_stats['total_jobs']}")
print(f"Total CPU Used Time: {job_stats['total_cpu_used_time']:.2f}s")
print(f"Total CPU Allocated Time: {job_stats['total_cpu_allocated_time']:.2f}s")
print(f"Total Network Transfer: {job_stats['total_network_transfer_mb']:.2f} MB")
```

## Data Structures

### JobMetrics Dataclass

```python
@dataclass
class JobMetrics:
    """Job-level metrics for a single job execution."""
    total_cpu_used_time: float  # Actual CPU time used from event processing (including overhead)
    total_cpu_allocated_time: float  # CPU time allocated for the whole job (including overhead)
    total_execution_time: float  # Total sequential execution time (tasksets execute one after another)
    total_write_local_mb: float
    total_write_remote_mb: float
    total_read_remote_mb: float
    total_read_local_mb: float
    total_network_transfer_mb: float  # remote_write + remote_read
    job_overhead: float  # Total job overhead time in seconds
```

### Job Statistics Dictionary

```python
{
    'total_cpu_used_time': float,
    'total_cpu_allocated_time': float,
    'total_execution_time': float,
    'total_write_local_mb': float,
    'total_write_remote_mb': float,
    'total_read_remote_mb': float,
    'total_read_local_mb': float,
    'total_network_transfer_mb': float,
    'job_overhead': float
}
```

## Input Requirements

### Taskset Information
Each taskset must provide:
- `time_per_event`: Processing time per event (seconds)
- `multicore`: Number of CPU cores
- `size_per_event`: Data size per event (KB)
- `keep_output`: Whether to keep output in shared storage
- `taskset_id`: Unique identifier
- `input_taskset`: Input dependency (if any)

### Job Information
- `batch_size`: Number of events to process
- `input_tasksets_for_other_groups`: Set of taskset IDs that are inputs for other groups
- `input_taskset_size_per_event`: Size per event of input taskset (if remote read)

## Unit Conversions

- **SizePerEvent**: Input in KB, automatically converted to MB for output
- **Time**: All time calculations in seconds
- **Storage**: All storage metrics in MB

## Example Output

```
Job Metrics for group_1_job_1:
  CPU Used Time: 1260.00s (includes 60s overhead)
  CPU Allocated Time: 1620.00s (includes 60s overhead × 2 cores)
  Execution Time: 1200.00s
  Job Overhead: 60.00s (1 taskset × 60s)
  Local Write: 200.00 MB
  Remote Write: 150.00 MB
  Remote Read: 0.00 MB
  Network Transfer: 150.00 MB

Aggregated Job Statistics:
  Total Jobs: 100
  Total CPU Used Time: 126000.00s
  Total CPU Allocated Time: 162000.00s
  Total Execution Time: 120000.00s
  Total Job Overhead: 6000.00s
  Total Local Write: 20000.00 MB
  Total Remote Write: 15000.00 MB
  Total Remote Read: 0.00 MB
  Total Network Transfer: 15000.00 MB
```

## Integration Points

### With WorkflowSimulator
- Called during job creation to calculate job metrics
- Receives taskset information and batch size
- Returns `JobMetrics` object for job storage

### With WorkflowMetricsCalculator
- Used for aggregating job statistics
- Provides job-level metrics for workflow analysis
- Enables detailed resource usage reporting

## Error Handling

- **Empty Jobs List**: Returns zero values for all metrics
- **Invalid Batch Size**: Handles zero or negative batch sizes gracefully
- **Missing Input Data**: Returns None for optional parameters
- **Unit Conversion**: Handles division by zero in unit conversions

## Testing

Run the test suite to verify functionality:

```bash
pytest tests/test_job_metrics.py -v
```

## Overhead Configuration

The overhead calculation uses configurable constants defined in `workflow_simulator.py`:

- **`TASKSET_OVERHEAD_SECONDS`**: Default 60.0 seconds per taskset
- **Data transfer rate**: Default 100.0 MB/s for data transfer overhead (configurable via `--data-transfer-rate` in simulator/runner)

These can be customized when initializing `JobMetricsCalculator`:

```python
from src.job_metrics import JobMetricsCalculator

# Custom overhead configuration
calculator = JobMetricsCalculator(
    taskset_overhead_seconds=90.0,  # 90 seconds per taskset
    data_transfer_rate_mb_per_s=150.0  # 150 MB/s transfer rate
)
```

## Future Enhancements

- **I/O Timing**: Add I/O operation timing metrics
- **Resource Efficiency**: Calculate I/O efficiency per job
- **Custom Metrics**: Support for user-defined job metrics
- **Configurable Overhead**: Runtime configuration of overhead parameters
