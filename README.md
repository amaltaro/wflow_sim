# Workflow Simulator

[![Tests](https://github.com/amaltaro/wflow_sim/workflows/Run%20Tests/badge.svg)](https://github.com/amaltaro/wflow_sim/actions/workflows/test.yml)
[![Release Notes](https://github.com/amaltaro/wflow_sim/workflows/Generate%20Release%20Notes/badge.svg)](https://github.com/amaltaro/wflow_sim/actions/workflows/release-notes.yml)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A comprehensive workflow simulation system for analyzing and comparing different workflow compositions in grid computing environments.

## Overview

This repository provides a **Workflow Simulator** that:
- Takes workflow descriptions as input (JSON format)
- Simulates workflow execution with realistic timing and resource constraints
- Produces execution metrics and performance analytics as output

## Project Structure

```
├── src/           # Python source code
├── tests/         # Unit tests (pytest)
├── docs/          # Detailed documentation
├── templates/     # JSON workflow templates
├── results/       # Simulation output (JSON)
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
- **Group-Level Analysis**: Performance metrics for each group
- **Resource Efficiency**: CPU, memory, storage, and network usage
- **Scalability Analysis**: Job scaling and parallel execution efficiency

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

### Optional: Install Testing Dependencies

If you want to run the tests:

```bash
pip install -r requirements.txt
pytest tests/ -v
```

## Quick Start

### 1. Calculate Workflow Metrics

```python
from src.workflow_metrics import WorkflowMetricsCalculator
import json

# Load workflow template
with open('templates/3tasks_composition_001.json', 'r') as f:
    workflow_data = json.load(f)

# Calculate metrics
calculator = WorkflowMetricsCalculator(workflow_data)
metrics = calculator.calculate_metrics()

# Display results
calculator.print_metrics(detailed=True)
```

### 2. Run Example

```bash
python examples/metrics_example.py
```

### 3. Run Tests

```bash
pytest tests/ -v
```

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

### Core Metrics
- **Total Tasksets**: Number of computational units
- **Total Groups**: Number of job submission units
- **Total Jobs**: Number of grid jobs (scaled by events)
- **Execution Time**: Total computational time
- **Resource Efficiency**: Overall resource utilization
- **Throughput**: Events processed per second
- **Success Rate**: Percentage of successful executions

### Group-Level Metrics
- **Group Execution Time**: Time per group
- **Resource Usage**: CPU, memory, storage per group
- **Job Count**: Jobs per group based on event scaling
- **Taskset Performance**: Individual taskset metrics

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

- [Workflow Metrics Usage](docs/workflow_metrics_usage.md) - Detailed metrics documentation
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