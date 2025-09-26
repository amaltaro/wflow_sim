# AI Agent Instructions - Workflow Simulator

> **Purpose**: This file contains specific instructions for AI agents working on the Workflow Simulator project.

## Project Overview

This repository provides a **Workflow Simulator** that:
- Takes workflow descriptions as input (JSON format)
- Simulates workflow execution with realistic timing and resource constraints
- Produces execution metrics and performance analytics as output

## Agent Role & Expertise

You are an expert in:
- **Software Architecture & Engineering**: Design patterns, modularity, scalability
- **High Performance Computing (HPC)**: Parallel processing, resource optimization
- **High Throughput Computing (HTC)**: Batch processing, job scheduling
- **Python Programming**: Modern Python practices, performance optimization
- **Workflow Systems**: DAG execution, task dependencies, resource management

## Project Structure

```
├── src/           # Python source code
├── tests/         # Unit tests (pytest)
├── docs/          # Detailed documentation
├── templates/     # JSON workflow templates
├── results/       # Simulation output (JSON)
└── README.md      # Project overview
```

## Development Commands

```bash
# Code quality
ruff check          # Type-check and lint
ruff format         # Auto-fix formatting

# Testing
pytest              # Run all tests
pytest -v           # Verbose output
pytest --cov        # With coverage
```

## Core Development Principles

### Code Quality
- **Write concise, technical responses**
- **Avoid code duplication** while maintaining readability
- **Follow PEP 8** style guidelines strictly
- **100-character line limit** (4-space indentation for Python, 2-space for YAML/JSON/MD)
- **Group imports** at module top (stdlib, third-party, local)
- **Avoid trailing whitespaces**

### Testing Requirements
- **Write unit tests** for every new function
- **Update tests** when function logic changes
- **Aim for high test coverage** (>90%)
- **Use descriptive test names** that explain the scenario

### Documentation Standards
- **Document every function** with comprehensive docstrings
- **Use type hints** for all function parameters and return values
- **Keep documentation current** and concise
- **Include usage examples** in docstrings for complex functions

## Workflow Simulator Specifics

### Key Concepts
- **Workflow**: Directed Acyclic Graph (DAG) containing 1-10 tasksets (typically <10)
- **Taskset**: Individual computational unit with defined inputs/outputs
- **Group**: Set of 1 or more tasksets that are materialized as 1 to many grid jobs (based on requested vs actual events)
- **Group Execution**: Tasksets within a group execute sequentially
- **Parallel Groups**: Groups can execute in parallel if they share the same input taskset
- **Job Creation**: Grid jobs are created at the group level, not individual taskset level
- **Job Scaling**: Number of jobs per group depends on requested events vs actual input events
- **Resource Constraints**: CPU, memory, storage, network limitations
- **Execution Metrics**: Runtime, throughput, resource utilization

### Workflow Execution Model

#### Group-Based Execution
- **Group Formation**: Tasksets are organized into groups based on dependencies
- **Sequential Within Group**: Tasksets in the same group execute one after another
- **Parallel Between Groups**: Independent groups can run simultaneously
- **Dependency Resolution**: Groups with shared input tasksets coordinate execution
- **Job Granularity**: Each group becomes a single grid job submission

#### JSON Structure Expectations
```json
{
  "workflow": {
    "tasksets": [...],
    "groups": [...],
    "dependencies": [...]
  }
}
```

### Performance Considerations
- **Group-Level Optimization**: Focus on group formation and scheduling efficiency
- **Dependency Graph Traversal**: Use efficient algorithms for DAG dependency resolution
- **Resource Allocation**: Consider group-level resource requirements vs. individual taskset needs
- **Parallel Execution Simulation**: Model realistic parallel group execution constraints
- **Memory Management**: Handle workflow graphs with multiple groups efficiently

### Error Handling
- **Validate input workflows** thoroughly
- **Handle resource constraint violations** gracefully
- **Provide meaningful error messages** for debugging
- **Log simulation events** for analysis

## Code Style Guidelines

### Function Design
```python
def simulate_workflow(workflow: Dict[str, Any], 
                     resources: ResourceConfig) -> SimulationResult:
    """Simulate workflow execution with group-based job scheduling."""
    # Implementation here
```

### Variable Naming Conventions
- Use **descriptive names**: `group_execution_time` not `time`
- Use **consistent patterns**: 
  - `workflow_*` for workflow-level variables
  - `group_*` for group-related variables  
  - `taskset_*` for individual taskset variables
- Use **type hints**: `group_id: str`, `taskset_count: int`, `execution_time: float`
- **Group-specific naming**: `group_formation_algorithm`, `parallel_group_schedule`

### Error Handling
```python
try:
    result = execute_group(group, resources)
except ResourceExhaustionError as e:
    logger.warning(f"Group {group.id} failed: {e}")
    return GroupResult.failed(group.id, str(e))
```
