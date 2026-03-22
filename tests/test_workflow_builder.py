"""Unit tests for workflow_builder module."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent / "src"))
from workflow_builder import (
    Task,
    TaskGrouper,
    TaskResources,
    build_all_constructions,
    create_tasks_from_workflow,
    extract_os_and_arch,
    find_all_workflow_constructions,
    load_workflow,
    validate_task_parameters,
    write_output,
)


class TestExtractOsAndArch:
    """Tests for extract_os_and_arch."""

    def test_el8_amd64(self) -> None:
        assert extract_os_and_arch(["el8_amd64_gcc11"]) == ("8", "amd64")

    def test_el9_amd64(self) -> None:
        assert extract_os_and_arch(["el9_amd64_gcc11"]) == ("9", "amd64")


class TestValidateTaskParameters:
    """Tests for validate_task_parameters."""

    def test_valid_params(self) -> None:
        task_data = {
            "ScramArch": ["el8_amd64_gcc11"],
            "TimePerEvent": 1.0,
            "Memory": 2000,
            "Multicore": 2,
            "SizePerEvent": 100,
        }
        validate_task_parameters(task_data, "Taskset1")

    def test_missing_param_raises(self) -> None:
        task_data = {"ScramArch": ["el8_amd64_gcc11"], "Memory": 2000}
        with pytest.raises(ValueError, match="Missing required parameters"):
            validate_task_parameters(task_data, "Taskset1")


class TestTaskGrouper:
    """Tests for TaskGrouper."""

    def test_sequential_3tasks_same_arch(self) -> None:
        tasks = {
            "Taskset1": Task(
                id="Taskset1",
                resources=TaskResources("8", "amd64", None),
                input_task=None,
                output_tasks={"Taskset2"},
                order=1,
            ),
            "Taskset2": Task(
                id="Taskset2",
                resources=TaskResources("8", "amd64", None),
                input_task="Taskset1",
                output_tasks={"Taskset3"},
                order=2,
            ),
            "Taskset3": Task(
                id="Taskset3",
                resources=TaskResources("8", "amd64", None),
                input_task="Taskset2",
                output_tasks=set(),
                order=3,
            ),
        }
        grouper = TaskGrouper(tasks)
        groups = grouper.generate_all_possible_groups()
        assert len(groups) > 0
        assert ["Taskset1", "Taskset2", "Taskset3"] in groups

    def test_tasks_different_arch_not_grouped(self) -> None:
        tasks = {
            "Taskset1": Task(
                id="Taskset1",
                resources=TaskResources("8", "amd64", None),
                input_task=None,
                output_tasks={"Taskset2"},
                order=1,
            ),
            "Taskset2": Task(
                id="Taskset2",
                resources=TaskResources("9", "amd64", None),
                input_task="Taskset1",
                output_tasks=set(),
                order=2,
            ),
        }
        grouper = TaskGrouper(tasks)
        groups = grouper.generate_all_possible_groups()
        assert ["Taskset1", "Taskset2"] not in groups

    def test_sequential_4tasks_path_containment(self) -> None:
        """T1->T2->T3->T4: siblings on a path can group; non-adjacent need intermediates."""
        tasks = {
            "Taskset1": Task(
                id="Taskset1",
                resources=TaskResources("8", "amd64", None),
                input_task=None,
                output_tasks={"Taskset2"},
                order=1,
            ),
            "Taskset2": Task(
                id="Taskset2",
                resources=TaskResources("8", "amd64", None),
                input_task="Taskset1",
                output_tasks={"Taskset3"},
                order=2,
            ),
            "Taskset3": Task(
                id="Taskset3",
                resources=TaskResources("8", "amd64", None),
                input_task="Taskset2",
                output_tasks={"Taskset4"},
                order=3,
            ),
            "Taskset4": Task(
                id="Taskset4",
                resources=TaskResources("8", "amd64", None),
                input_task="Taskset3",
                output_tasks=set(),
                order=4,
            ),
        }
        grouper = TaskGrouper(tasks)
        groups = grouper.generate_all_possible_groups()
        assert ["Taskset1", "Taskset2", "Taskset3", "Taskset4"] in groups
        assert ["Taskset1", "Taskset2"] in groups
        assert ["Taskset2", "Taskset3"] in groups
        assert ["Taskset3", "Taskset4"] in groups
        assert ["Taskset1", "Taskset3"] not in groups
        assert ["Taskset1", "Taskset4"] not in groups
        assert ["Taskset2", "Taskset4"] not in groups

    def test_fork_t1_branches_t2_t3_siblings_not_grouped(self) -> None:
        """T1->T2, T1->T3: siblings T2 and T3 have no dependency path; cannot group."""
        tasks = {
            "Taskset1": Task(
                id="Taskset1",
                resources=TaskResources("8", "amd64", None),
                input_task=None,
                output_tasks={"Taskset2", "Taskset3"},
                order=1,
            ),
            "Taskset2": Task(
                id="Taskset2",
                resources=TaskResources("8", "amd64", None),
                input_task="Taskset1",
                output_tasks=set(),
                order=2,
            ),
            "Taskset3": Task(
                id="Taskset3",
                resources=TaskResources("8", "amd64", None),
                input_task="Taskset1",
                output_tasks=set(),
                order=3,
            ),
        }
        grouper = TaskGrouper(tasks)
        groups = grouper.generate_all_possible_groups()
        assert ["Taskset2", "Taskset3"] not in groups
        assert ["Taskset1", "Taskset2"] in groups
        assert ["Taskset1", "Taskset3"] in groups

    def test_fork_t2_branches_t3_t4_siblings_not_grouped(self) -> None:
        """T1->T2->(T3,T4): T2 feeds both T3 and T4; siblings T3 and T4 cannot group."""
        tasks = {
            "Taskset1": Task(
                id="Taskset1",
                resources=TaskResources("8", "amd64", None),
                input_task=None,
                output_tasks={"Taskset2"},
                order=1,
            ),
            "Taskset2": Task(
                id="Taskset2",
                resources=TaskResources("8", "amd64", None),
                input_task="Taskset1",
                output_tasks={"Taskset3", "Taskset4"},
                order=2,
            ),
            "Taskset3": Task(
                id="Taskset3",
                resources=TaskResources("8", "amd64", None),
                input_task="Taskset2",
                output_tasks=set(),
                order=3,
            ),
            "Taskset4": Task(
                id="Taskset4",
                resources=TaskResources("8", "amd64", None),
                input_task="Taskset2",
                output_tasks=set(),
                order=4,
            ),
        }
        grouper = TaskGrouper(tasks)
        groups = grouper.generate_all_possible_groups()
        assert ["Taskset3", "Taskset4"] not in groups
        assert ["Taskset1", "Taskset2", "Taskset3"] in groups
        assert ["Taskset1", "Taskset2", "Taskset4"] in groups
        assert ["Taskset1", "Taskset2", "Taskset3", "Taskset4"] not in groups

    def test_diamond_t2_t3_siblings_not_grouped(self) -> None:
        """T1->(T2,T3), T2->T4, T3->T5: T2 and T3 are siblings; T4 and T5 are siblings."""
        tasks = {
            "Taskset1": Task(
                id="Taskset1",
                resources=TaskResources("8", "amd64", None),
                input_task=None,
                output_tasks={"Taskset2", "Taskset3"},
                order=1,
            ),
            "Taskset2": Task(
                id="Taskset2",
                resources=TaskResources("8", "amd64", None),
                input_task="Taskset1",
                output_tasks={"Taskset4"},
                order=2,
            ),
            "Taskset3": Task(
                id="Taskset3",
                resources=TaskResources("8", "amd64", None),
                input_task="Taskset1",
                output_tasks={"Taskset5"},
                order=3,
            ),
            "Taskset4": Task(
                id="Taskset4",
                resources=TaskResources("8", "amd64", None),
                input_task="Taskset2",
                output_tasks=set(),
                order=4,
            ),
            "Taskset5": Task(
                id="Taskset5",
                resources=TaskResources("8", "amd64", None),
                input_task="Taskset3",
                output_tasks=set(),
                order=5,
            ),
        }
        grouper = TaskGrouper(tasks)
        groups = grouper.generate_all_possible_groups()
        assert ["Taskset2", "Taskset3"] not in groups
        assert ["Taskset4", "Taskset5"] not in groups
        assert ["Taskset1", "Taskset2", "Taskset4"] in groups
        assert ["Taskset1", "Taskset3", "Taskset5"] in groups
        assert ["Taskset1", "Taskset2", "Taskset3"] not in groups


class TestFindAllWorkflowConstructions:
    """Tests for find_all_workflow_constructions."""

    def test_sequential_3tasks_finds_constructions(self) -> None:
        tasks = {
            "Taskset1": Task(
                id="Taskset1",
                resources=TaskResources("8", "amd64", None),
                input_task=None,
                output_tasks={"Taskset2"},
                order=1,
            ),
            "Taskset2": Task(
                id="Taskset2",
                resources=TaskResources("8", "amd64", None),
                input_task="Taskset1",
                output_tasks={"Taskset3"},
                order=2,
            ),
            "Taskset3": Task(
                id="Taskset3",
                resources=TaskResources("8", "amd64", None),
                input_task="Taskset2",
                output_tasks=set(),
                order=3,
            ),
        }
        grouper = TaskGrouper(tasks)
        valid_groups = grouper.generate_all_possible_groups()
        constructions = find_all_workflow_constructions(grouper, valid_groups)
        assert len(constructions) >= 1
        all_tasks = set()
        for gid, task_ids in constructions[0]:
            all_tasks.update(task_ids)
        assert all_tasks == {"Taskset1", "Taskset2", "Taskset3"}


class TestBuildAllConstructions:
    """Tests for build_all_constructions."""

    def test_3tasks_workflow(self) -> None:
        workflow = {
            "NumTasks": 3,
            "RequestNumEvents": 1000000,
            "Taskset1": {
                "ScramArch": ["el9_amd64_gcc11"],
                "TimePerEvent": 10,
                "Memory": 2000,
                "Multicore": 1,
                "SizePerEvent": 200,
            },
            "Taskset2": {
                "ScramArch": ["el9_amd64_gcc11"],
                "TimePerEvent": 20,
                "Memory": 4000,
                "Multicore": 2,
                "SizePerEvent": 300,
                "InputTaskset": "Taskset1",
            },
            "Taskset3": {
                "ScramArch": ["el9_amd64_gcc11"],
                "TimePerEvent": 10,
                "Memory": 3000,
                "Multicore": 2,
                "SizePerEvent": 50,
                "InputTaskset": "Taskset2",
            },
        }
        constructions, tasks = build_all_constructions(workflow)
        assert len(constructions) >= 1
        assert len(tasks) == 3


class TestWriteOutput:
    """Tests for write_output."""

    def test_writes_summary_and_composition_files(self) -> None:
        workflow = {
            "NumTasks": 2,
            "RequestNumEvents": 1000,
            "Taskset1": {
                "ScramArch": ["el9_amd64_gcc11"],
                "TimePerEvent": 1,
                "Memory": 2000,
                "Multicore": 1,
                "SizePerEvent": 100,
            },
            "Taskset2": {
                "ScramArch": ["el9_amd64_gcc11"],
                "TimePerEvent": 1,
                "Memory": 2000,
                "Multicore": 1,
                "SizePerEvent": 100,
                "InputTaskset": "Taskset1",
            },
        }
        constructions = [
            [("group_0", ["Taskset1", "Taskset2"])],
            [("group_0", ["Taskset1"]), ("group_1", ["Taskset2"])],
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir)
            write_output(out_path, workflow, constructions, "test")
            summary_file = out_path / "test_compositions_summary.json"
            assert summary_file.exists()
            with open(summary_file) as f:
                summary = json.load(f)
            assert summary["total_compositions"] == 2
            assert len(summary["compositions"]) == 2
            assert (out_path / "test_const_001.json").exists()
            assert (out_path / "test_const_002.json").exists()
