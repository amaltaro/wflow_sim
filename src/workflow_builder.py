# Standard library imports
import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Third-party library imports
import networkx as nx


def extract_os_and_arch(scram_arch: List[str]) -> Tuple[str, str]:
    """Extract OS version and CPU architecture from ScramArch string.

    Args:
        scram_arch: List of ScramArch strings (e.g., ["el8_amd64_gcc11"])

    Returns:
        Tuple of (os_version, cpu_arch)
        e.g., ("8", "amd64")
    """
    os_part, arch, _ = scram_arch[0].split('_')
    os_version = ''.join(c for c in os_part if c.isdigit())
    return os_version, arch


@dataclass
class TaskResources:
    """Resources of a task (minimal set for grouping)."""

    os_version: str
    cpu_arch: str
    accelerator: Optional[str]


@dataclass
class Task:
    """Task with resources and dependencies."""

    id: str
    resources: TaskResources
    input_task: Optional[str] = None
    output_tasks: Optional[Set[str]] = None
    order: int = 0


class TaskGrouper:
    """Groups tasks based on hard requirements (OS, CPU arch, dependency paths)."""

    def __init__(self, tasks: Dict[str, Task]) -> None:
        self.tasks = tasks
        self.dag = self._build_dag()

    def _build_dag(self) -> nx.DiGraph:
        """Build a directed acyclic graph from tasks."""
        dag = nx.DiGraph()
        for task_id, task in self.tasks.items():
            dag.add_node(task_id)
            if task.input_task:
                dag.add_edge(task.input_task, task_id)
        return dag

    def _can_be_grouped(self, task1: Task, task2: Task) -> bool:
        """Check if tasks can be grouped (dependency path, same OS, same CPU arch).

        Two tasks can be grouped iff one depends on the other (i.e., there exists a
        directed path from one to the other in the DAG). The OR handles argument
        order: when (T3, T1) is passed for chain T1->T2->T3, has_path(T1,T3) is
        True (T3 depends on T1); has_path(T3,T1) would be False.
        """
        has_path_1_to_2 = nx.has_path(self.dag, task1.id, task2.id)
        has_path_2_to_1 = nx.has_path(self.dag, task2.id, task1.id)
        if not (has_path_1_to_2 or has_path_2_to_1):
            return False
        if task1.resources.os_version != task2.resources.os_version:
            return False
        if task1.resources.cpu_arch != task2.resources.cpu_arch:
            return False
        return True

    def _all_dependency_paths_within_group(self, group: Set[str]) -> bool:
        """Ensure all dependency paths between tasks in the group stay within it."""
        for src in group:
            for dst in group:
                if src == dst:
                    continue
                if nx.has_path(self.dag, src, dst):
                    for path in nx.all_simple_paths(self.dag, src, dst):
                        if not all(node in group for node in path):
                            return False
        return True

    def _is_valid_group(self, group: Set[str]) -> bool:
        """Check if a group of tasks is valid for grouping."""
        task_list = list(group)
        for i, t1 in enumerate(task_list):
            for t2 in task_list[i + 1 :]:
                if not self._can_be_grouped(self.tasks[t1], self.tasks[t2]):
                    return False
        return self._all_dependency_paths_within_group(group)

    def generate_all_possible_groups(self) -> List[List[str]]:
        """Generate all valid groups of tasks (deterministic order)."""
        all_tasks = set(self.tasks.keys())
        sorted_task_ids = sorted(all_tasks, key=lambda t: self.tasks[t].order)
        valid_groups: List[List[str]] = []

        for size in range(1, len(sorted_task_ids) + 1):
            for task_combo in combinations(sorted_task_ids, size):
                group = set(task_combo)
                if self._is_valid_group(group):
                    valid_groups.append(sorted(list(group)))

        return valid_groups


def find_all_workflow_constructions(
    grouper: TaskGrouper,
    valid_groups: List[List[str]],
) -> List[List[Tuple[str, List[str]]]]:
    """Find all valid workflow constructions from the given groups.

    Each construction is a list of (group_id, task_ids) tuples that together
    cover all tasks and respect dependencies.

    Returns:
        List of constructions; each construction is a list of (group_id, tasks).
    """
    all_tasks = set(grouper.tasks.keys())
    sorted_tasks = list(nx.topological_sort(grouper.dag))

    # Assign stable group_ids to valid groups (by sorted task list)
    group_id_map: Dict[Tuple[str, ...], str] = {}
    for idx, group in enumerate(valid_groups):
        key = tuple(sorted(group))
        if key not in group_id_map:
            group_id_map[key] = f"group_{idx}"

    # Build list of (group_id, task_ids) for each valid group
    group_entries: List[Tuple[str, List[str]]] = []
    for group in valid_groups:
        key = tuple(sorted(group))
        group_entries.append((group_id_map[key], group))

    valid_constructions: List[List[Tuple[str, List[str]]]] = []
    seen: Set[frozenset] = set()

    def get_available_tasks(
        construction: List[Tuple[str, List[str]]],
    ) -> Set[str]:
        tasks_in = set()
        for _, tasks in construction:
            tasks_in.update(tasks)
        available = set()
        for task in all_tasks - tasks_in:
            preds = set(nx.ancestors(grouper.dag, task))
            if preds.issubset(tasks_in):
                available.add(task)
        return available

    def get_valid_groups(
        construction: List[Tuple[str, List[str]]],
        available: Set[str],
    ) -> List[Tuple[str, List[str]]]:
        tasks_in = set()
        for _, tasks in construction:
            tasks_in.update(tasks)
        result = []
        for gid, tasks in group_entries:
            task_set = set(tasks)
            if task_set & available and not (task_set & tasks_in):
                result.append((gid, tasks))
        return result

    def find_constructions(
        current: List[Tuple[str, List[str]]],
    ) -> None:
        tasks_in = set()
        for _, tasks in current:
            tasks_in.update(tasks)

        if tasks_in == all_tasks:
            key = frozenset(gid for gid, _ in current)
            if key not in seen:
                seen.add(key)
                valid_constructions.append(current.copy())
            return

        available = get_available_tasks(current)
        if not available:
            return

        for entry in get_valid_groups(current, available):
            current.append(entry)
            find_constructions(current)
            current.pop()

    find_constructions([])

    # Sort by number of groups (ascending)
    valid_constructions.sort(key=lambda x: len(x))

    return valid_constructions


def validate_task_parameters(task_data: dict, task_name: str) -> None:
    """Validate required parameters in task data."""
    required = ["ScramArch", "TimePerEvent", "Memory", "Multicore", "SizePerEvent"]
    missing = [p for p in required if p not in task_data]
    if missing:
        raise ValueError(
            f"Missing required parameters for {task_name}: {', '.join(missing)}"
        )


def load_workflow(input_path: Path) -> dict:
    """Load workflow JSON from file."""
    with open(input_path, encoding="utf-8") as f:
        return json.load(f)


def create_tasks_from_workflow(workflow_data: dict) -> Dict[str, Task]:
    """Create Task objects from workflow JSON (minimal for grouping)."""
    tasks: Dict[str, Task] = {}
    for i in range(1, workflow_data["NumTasks"] + 1):
        task_name = f"Taskset{i}"
        task_data = workflow_data[task_name]
        validate_task_parameters(task_data, task_name)
        os_version, cpu_arch = extract_os_and_arch(task_data["ScramArch"])
        resources = TaskResources(
            os_version=os_version,
            cpu_arch=cpu_arch,
            accelerator="GPU" if task_data.get("RequiresGPU") == "required" else None,
        )
        tasks[task_name] = Task(
            id=task_name,
            resources=resources,
            input_task=task_data.get("InputTaskset"),
            output_tasks=set(),
            order=i,
        )

    for task_name, task in tasks.items():
        if task.input_task:
            if tasks[task.input_task].output_tasks is None:
                tasks[task.input_task].output_tasks = set()
            tasks[task.input_task].output_tasks.add(task_name)

    return tasks


def build_all_constructions(
    workflow_data: dict,
) -> Tuple[List[List[Tuple[str, List[str]]]], Dict[str, Task]]:
    """Build all possible workflow constructions from workflow data."""
    tasks = create_tasks_from_workflow(workflow_data)
    grouper = TaskGrouper(tasks)
    valid_groups = grouper.generate_all_possible_groups()
    constructions = find_all_workflow_constructions(grouper, valid_groups)
    return constructions, tasks


def write_output(
    output_path: Path,
    workflow_data: dict,
    constructions: List[List[Tuple[str, List[str]]]],
    base_name: str,
) -> None:
    """Write compositions summary and individual composition JSON files."""
    output_path.mkdir(parents=True, exist_ok=True)

    compositions_summary = {
        "template_name": base_name,
        "total_compositions": len(constructions),
        "compositions": [],
    }

    for idx, construction in enumerate(constructions, 1):
        comp_num = idx
        groups = [gid for gid, _ in construction]
        group_details = [
            {"group_id": gid, "tasks": tasks} for gid, tasks in construction
        ]
        compositions_summary["compositions"].append(
            {
                "composition_number": comp_num,
                "num_groups": len(construction),
                "groups": groups,
                "group_details": group_details,
            }
        )

    summary_file = output_path / f"{base_name}_compositions_summary.json"
    print(f"=> Writing compositions summary to {summary_file}")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(compositions_summary, f, indent=2)

    # Write individual composition files (workflow JSON with GroupName assigned)
    for idx, construction in enumerate(constructions, 1):
        out_workflow = dict(workflow_data)
        out_workflow["CompositionNumber"] = idx
        out_workflow["Comments"] = f"Workflow Composition {idx} - {len(construction)} groups"

        for gid, task_ids in construction:
            for task_id in task_ids:
                if task_id in out_workflow and isinstance(out_workflow[task_id], dict):
                    out_workflow[task_id] = dict(out_workflow[task_id])
                    out_workflow[task_id]["GroupName"] = gid
                    # Omit GroupInputEvents (no metrics)

        comp_file = output_path / f"{base_name}_const_{idx:03d}.json"
        print(f"=> Writing composition to {comp_file}")
        with open(comp_file, "w", encoding="utf-8") as f:
            json.dump(out_workflow, f, indent=2)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build all possible workflow constructions from a workflow JSON."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to input JSON file with workflow description",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Path to output directory for generated JSON files",
    )
    args = parser.parse_args()

    workflow_data = load_workflow(args.input)
    constructions, _ = build_all_constructions(workflow_data)

    base_name = args.input.stem
    # Strip common suffixes to get a clean base name
    for suffix in ("_composition_001", "_const_001", "_composition", "_const"):
        if base_name.endswith(suffix):
            base_name = base_name[: -len(suffix)]
            break

    write_output(args.output, workflow_data, constructions, base_name)
    print(f"Wrote {len(constructions)} constructions to {args.output}")


if __name__ == "__main__":
    main()
