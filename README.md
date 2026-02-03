# DAGFlowSim

**DAGFlowSim** (DAG Workflow Simulator) is a comprehensive workflow simulation system for analyzing and comparing different workflow compositions in grid computing environments.

[![Tests](https://github.com/amaltaro/wflow_sim/workflows/Run%20Tests/badge.svg)](https://github.com/amaltaro/wflow_sim/actions/workflows/test.yml)
[![Release Notes](https://github.com/amaltaro/wflow_sim/workflows/Generate%20Release%20Notes/badge.svg)](https://github.com/amaltaro/wflow_sim/actions/workflows/release-notes.yml)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

## Overview

DAGFlowSim provides a powerful workflow simulation engine that:
- Takes workflow descriptions as input (JSON format)
- Simulates workflow execution with realistic timing and resource constraints
- Produces execution metrics and performance analytics as output

## Project Structure

```
├── src/           # Python source code
├── tests/         # Unit tests (pytest)
├── docs/          # Detailed documentation
├── templates/     # JSON workflow templates
├── results/       # Simulation, visualization, and analysis outputs
│   ├── sim/       # Simulation results (JSON); batch: sim/others/<case>/<time>/fr<fr>/<rate>/
│   ├── vis/       # Visualization diagrams (PNG); same nesting as sim
│   ├── analysis/  # Cross-dimensional analyses (failure_rate, workflow_type_sensitivity, etc.)
│   ├── real/      # Real workflow execution data (summaries, visualizations)
│   └── real_norm/ # Normalized real data (per requested events) for fair comparison
├── examples/      # Usage examples
└── README.md      # Project overview
```

## Key Features

### Workflow Execution Model
- **Group-Based Execution**: Tasksets organized into groups for job submission
- **Sequential Within Groups**: Tasksets in the same group execute sequentially
- **Parallel Between Groups**: Independent groups can run simultaneously
- **Job Scaling**: Number of jobs per group depends on requested vs actual events

### Comprehensive Metrics
- **Execution Metrics**: Runtime, throughput, resource utilization
- **Job-Level Metrics**: CPU time, local/remote I/O, network transfers per job
- **Group-Level Analysis**: Performance metrics for each group
- **Resource Efficiency**: CPU, memory, storage, and network usage
- **Scalability Analysis**: Job scaling and parallel execution efficiency

## Architecture

**Four modules with distinct responsibilities:**

- **`workflow_simulator.py`** - Core simulation engine (executes workflow DAGs)
- **`job_metrics.py`** - Job-level metrics calculator (CPU time, I/O operations, network transfers)
- **`workflow_metrics.py`** - Authoritative workflow metrics calculator (aggregates job metrics)
- **`workflow_runner.py`** - High-level orchestrator (simulation + metrics)

**Key Design:**
- **Clear Separation of Concerns**: Each module has a single, well-defined responsibility
- **Job-Level Metrics**: `job_metrics.py` handles individual job resource calculations
- **Workflow-Level Metrics**: `workflow_metrics.py` aggregates job metrics into workflow-level insights
- **Simulation Results Only**: All metrics work with simulation results (not raw workflow data)
- **No Redundancy**: Each calculation happens in exactly one place

## Installation

### Prerequisites

- Python 3.8 or higher
- No external dependencies required (uses only Python standard library)

### Quick Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/amaltaro/wflow_sim.git
   cd wflow_sim
   ```

2. **Run the example:**
   ```bash
   python examples/metrics_example.py
   ```

DAGFlowSim requires only Python standard library, making it easy to get started quickly.

### Optional: Install Testing Dependencies

If you want to run the tests:

```bash
pip install -r requirements.txt
pytest tests/ -v
```

## Quick Start

### 1. Command Line Usage

The easiest way to run workflow simulations is using the command line interface:

```bash
# Basic usage with default settings
python src/workflow_runner.py

# Custom wallclock time and job slots
python src/workflow_runner.py --target-wallclock-time 3600 --max-job-slots 10

# Specify custom workflow file
python src/workflow_runner.py --input-workflow-path templates/3tasks/seq/3tasks_001.json

# Show all available options
python src/workflow_runner.py --help
```

**Output**: Single-run results go to `results/sim/` mirroring the input path. Batch runs (Makefile) use a unified tree under `results/sim/others/` (see Batch Processing).

### 2. Python API Usage

```python
from src.workflow_runner import WorkflowRunner
from src.workflow_metrics import WorkflowMetricsCalculator

# Run simulation and get results (failure_rate and data_transfer_rate from parser or pass explicitly)
runner = WorkflowRunner(failure_rate=0, data_transfer_rate_mb_per_s=100.0)
results = runner.run_workflow('templates/3tasks_composition_001.json')

# Calculate metrics from simulation results
calculator = WorkflowMetricsCalculator()
metrics = calculator.calculate_metrics(results['simulation_result'])

# Display results
calculator.print_metrics()
```

### 3. Run Examples

```bash
# Command line example
python src/workflow_runner.py --target-wallclock-time 1800

# Python API example
python examples/metrics_example.py
```

### 4. Run Tests

```bash
pytest tests/ -v
```

## Command Line Interface

Both `workflow_runner.py` and `workflow_simulator.py` support command line arguments for easy usage:

### Available Arguments

- `--target-wallclock-time`: Target wallclock time in seconds (default: 43200 = 12 hours)
- `--max-job-slots`: Maximum number of job slots (-1 for infinite, default: -1)
- `--input-workflow-path`: Path to input workflow JSON file (default: templates/3tasks_composition_001.json)

### Usage Examples

```bash
# Show help
python src/workflow_runner.py --help
```

### Output Structure

Single-run: results follow the input path under `results/sim/` (e.g. `templates/others/case1_real/...` → `results/sim/others/case1_real/...`). Batch runs use the unified structure described under Batch Processing.

## Batch Processing with Makefile

The Makefile provides convenient targets for running batch simulations and generating visualizations for multiple workflow use cases.

### Quick Start

Run simulations and visualizations for all configured use cases with a single command:

```bash
make all
```

This will:
1. Simulate all workflows in the configured use cases (default: `case1_real`, `case2_homo`, `case3_hetero`)
2. Generate visualization diagrams for all simulation results

### Available Makefile Targets

```bash
make help                    # Show all targets

# Simulations and visualizations
make simulate-all            # All use cases × times × failure rates × data rates
make visualize-all           # Visualizations for existing sim results
make all                     # simulate-all + visualize-all
make run                     # Single workflow (case1_real const_001, 12h)

# Analysis (run after simulate-all; writes to results/analysis/)
make analyze-failure-rate
make analyze-workflow-type-sensitivity
make analyze-target-job-length
make analyze-data-transfer-rate

# Cleanup
make clean                   # All generated files
make clean-viz               # Only visualizations
make clean-results           # Only simulation results
```

### Customizing Use Cases

You can customize which use cases to process by setting the `USE_CASES` variable:

```bash
# Run only specific use cases
make all USE_CASES='case1_real case2_homo'

# Run a single use case
make all USE_CASES='case1_real'
```

### Configuration

The Makefile uses the following default configuration (editable in `Makefile`):

- **Target wallclock time**: 43200 seconds (12 hours)
- **Max job slots**: -1 (infinite)
- **Use cases**: `case1_real case2_homo case3_hetero`
- **Template directory**: `templates/others/`
- **Results directory**: `results/sim/others`
- **Visualization output**: `results/vis/others`

### Output Locations

Batch outputs use a unified tree (dimensions: use case, target job length, failure rate, data transfer rate):

- **Simulations**: `results/sim/others/<use_case>/<time_dir>/fr<failure_rate>/<data_rate>/` (e.g. `12h`, `fr0`, `100MBps`)
- **Visualizations**: `results/vis/others/<use_case>/<time_dir>/fr<failure_rate>/<data_rate>/`
- **Analysis**: `results/analysis/<analysis_name>/...` (e.g. `failure_rate/<use_case>/<time_dir>/`, `workflow_type_sensitivity/12h/fr0/`, `target_job_length/<use_case>/fr0/`, `data_transfer_rate/fr0/`)
- **Real execution**: `results/real/` — real workflow run summaries and visualizations
- **Real (normalized)**: `results/real_norm/` — same data normalized so each workflow is scaled to the requested number of events for fair comparison

The visualization script generates comparison plots including:
- I/O patterns analysis (per event and total volumes)
- Resource utilization (CPU, memory, network)
- Performance metrics (throughput, efficiency)

## Workflow Data Format

Workflows are defined in JSON format with the following structure:

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
  "CompositionNumber": 1
}
```

## Metrics Calculated

### Core Workflow Metrics
- **Total Tasksets**: Number of computational units
- **Total Groups**: Number of job submission units
- **Total Jobs**: Number of grid jobs (scaled by events)
- **Execution Time**: Total computational time
- **Resource Efficiency**: Overall resource utilization
- **Throughput**: Events processed per second
- **Success Rate**: Percentage of successful executions
- **Total Job Overhead**: Sum of job overhead over all jobs (wallclock: `total_job_overhead_secs`; CPU: `total_job_overhead_cpu_time`)

### Job-Level Metrics
- **CPU Time**: Total CPU time per job including overhead (time_per_event × events × multicore + job_overhead)
- **Execution Time**: Total sequential execution time for all tasksets in a job (time_per_event × events for each taskset)
- **Job Overhead**: Realistic overhead accounting including:
  - Taskset overhead: 60 seconds for the bootstrap of each taskset in a group.
  - Remote read overhead: 1 second per 100MB/s of data transfer
  - Remote write overhead: 1 second per 100MB/s of data transfer
- **Local I/O**: Data written to local disk per job
- **Remote I/O**: Data written to shared storage per job
- **Network Transfer**: Complete network data transfer per job (remote writes + remote reads)
- **Remote Read**: Data read from shared storage per job (cross-group dependencies)
- **Local Read**: Data read from local disk per job (within-group dependencies)

### Group-Level Metrics
- **Group Execution Time**: Time per group
- **Resource Usage**: CPU, memory, storage per group
- **Job Count**: Jobs per group based on event scaling
- **Taskset Performance**: Individual taskset metrics

Simulation JSON output also includes a top-level **`simulation_stats`** object with distribution statistics (mean, std, median, min, max, n) over all jobs for job overhead, for use in visualizations and error bars.

## Development

### Testing

```bash
# Install testing dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html
```

### Continuous Integration

The project includes a GitHub Actions workflow that automatically:
- Runs on every push to `main` and `develop` branches
- Runs on pull requests to `main` branch
- Tests on Python 3.12
- Installs dependencies and runs unit tests
- Verifies the example script works correctly

If any tests fail, the CI pipeline will report the error and fail the build.

### Code Quality

The project follows Python best practices:
- Use type hints where appropriate
- Follow PEP 8 style guidelines
- Write comprehensive docstrings
- Maintain high test coverage

### Release Process
```bash
# Test release workflow locally
./scripts/test-release.sh

# Create a release (triggers automated release notes)
git tag -a v1.0.0 -m "Release 1.0.0"
git push origin v1.0.0
```

## Documentation

- [Workflow Simulation Usage](docs/workflow_simulation_usage.md) - Complete simulation guide
- [Workflow Metrics Usage](docs/workflow_metrics_usage.md) - Workflow-level metrics documentation
- [Job Metrics Usage](docs/job_metrics_usage.md) - Job-level metrics documentation
- [Visualization Usage](docs/visualization_usage.md) - Visualization tool documentation
- [Scripts Usage](docs/scripts_usage.md) - Utility scripts for data processing and analysis
- [Release Process](docs/release-process.md) - Automated release notes system
- [Agent Instructions](AGENTS.md) - AI agent development guidelines
- [Contributing Guidelines](CONTRIBUTING.md) - How to contribute to the project

## Examples

See the `examples/` directory for complete working examples:
- `metrics_example.py` - Basic metrics calculation
- Template files in `templates/` directory

## Contributing

This project follows specific development patterns and constraints. See [AGENTS.md](AGENTS.md) for detailed guidelines for AI agents working on this project.

## License

See [LICENSE](LICENSE) for license information.