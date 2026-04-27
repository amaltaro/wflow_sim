# Workflow Builder Usage Guide

The **workflow builder** (`src/workflow_builder.py`) generates all possible workflow constructions from a generic workflow description. It applies only **hard constraints** (operating system and CPU architecture compatibility) and produces composition JSON files suitable for simulation.

## Overview

The workflow builder:

- **Input**: A generic workflow JSON (tasks, dependencies, resource requirements)
- **Output**: All valid workflow constructions as JSON files (compositions summary + individual composition files)
- **Scope**: Grouping logic only — no metrics, soft constraints, or visualization

### Hard Constraints

Tasks can be grouped together only if:

1. **Dependency path**: There is a path between them in the workflow DAG (in either direction)
2. **Same OS version**: Extracted from `ScramArch` (e.g. `el8_amd64_gcc11` → OS `8`)
3. **Same CPU architecture**: Extracted from `ScramArch` (e.g. `amd64`)

Additionally, all dependency paths between tasks in a group must stay within the group.

## Command Line Usage

```bash
# Basic usage
python -m src.workflow_builder --input templates/generic/seq_real.json --output templates/others/seq_real

# Short options
python -m src.workflow_builder -i templates/generic/seq_homo.json -o templates/others/seq_homo

# Show help
python -m src.workflow_builder --help
```

### Arguments

| Option | Short | Description |
|--------|-------|-------------|
| `--input` | `-i` | Path to input JSON file (generic workflow description) |
| `--output` | `-o` | Path to output directory for generated JSON files |

## Output Files

For input `templates/generic/seq_real.json` and output `templates/others/seq_real`:

- **`seq_real_compositions_summary.json`** — Summary of all constructions (composition_number, num_groups, groups, group_details)
- **`seq_real_const_001.json`**, **`seq_real_const_002.json`**, … — Individual workflow compositions with `GroupName` assigned per taskset

Each composition file is a full workflow JSON that can be used directly by the workflow simulator.

## Makefile Target

Build generic workflows in one command:

```bash
make build-workflows
```

`USE_CASES` is set from **`WORKFLOW_PRESET`** in the `Makefile` (default **`sequential`**): `seq_real`, `seq_homo`, and `seq_hetero`. For the fork templates, run `make build-workflows WORKFLOW_PRESET=fork` (uses `fork_real`, `fork_homo`, `fork_hetero`). You can still pass `USE_CASES='...'` to override the list.

## Input Format

The input JSON must include:

- `NumTasks` — Number of tasks
- `RequestNumEvents` — (Optional) Total events to process
- `Taskset1`, `Taskset2`, … — Per-taskset data with:
  - `ScramArch` — e.g. `["el8_amd64_gcc11"]` (used for OS/arch extraction)
  - `TimePerEvent`, `Memory`, `Multicore`, `SizePerEvent` — Required
  - `InputTaskset` — (Optional) Parent taskset for dependency

## Determinism

The workflow builder is **deterministic**: running it multiple times with the same input produces identical output. This holds even with different `PYTHONHASHSEED` values.

## Related Work

For full details on workflow construction, including:

- Group metrics calculation (CPU, memory, throughput, I/O)
- Soft constraints and scoring
- Comprehensive tests (sequential, fork workflows)
- Visualization and analysis tools
- Statistical analysis of construction metrics

see the **[workflow_construction](https://github.com/amaltaro/workflow_construction)** repository.