"""
Unit tests for workflow_simulator.py module.
Tests the WorkflowSimulator class and its methods.
"""

import json
import pytest
from pathlib import Path
import sys
import tempfile

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from workflow_simulator import WorkflowSimulator, ResourceConfig, SimulationResult, GroupInfo, TasksetInfo, JobInfo


class TestWorkflowSimulator:
    """Test cases for WorkflowSimulator."""

    def test_initialization(self):
        """Test simulator initialization."""
        simulator = WorkflowSimulator()
        assert simulator.resource_config is not None
        assert simulator.logger is not None
        # test default resource configs
        assert simulator.resource_config.target_wallclock_time == 43200.0
        assert simulator.resource_config.max_job_slots == -1
        assert simulator.resource_config.cpu_per_slot == 1
        assert simulator.resource_config.memory_per_slot == 1000

    def test_initialization_with_config(self):
        """Test simulator initialization with custom config."""
        config = ResourceConfig(
            target_wallclock_time=21600.0,  # 6 hours
            max_job_slots=50,
            cpu_per_slot=2,
            memory_per_slot=2000
        )
        simulator = WorkflowSimulator(config)
        assert simulator.resource_config.target_wallclock_time == 21600.0
        assert simulator.resource_config.max_job_slots == 50
        assert simulator.resource_config.cpu_per_slot == 2
        assert simulator.resource_config.memory_per_slot == 2000

    def test_simulate_workflow_success(self):
        """Test successful workflow simulation."""
        # Create a temporary workflow file
        workflow_data = {
            "Comments": "Test Workflow",
            "NumTasks": 2,
            "RequestNumEvents": 10000,
            "Taskset1": {
                "Memory": 2000,
                "Multicore": 1,
                "TimePerEvent": 10,
                "SizePerEvent": 200,
                "GroupName": "group_1",
            },
            "Taskset2": {
                "Memory": 4000,
                "Multicore": 2,
                "TimePerEvent": 20,
                "SizePerEvent": 300,
                "InputTaskset": "Taskset1",
                "GroupName": "group_1",
            },
            "CompositionNumber": 1
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(workflow_data, f)
            temp_file = f.name

        try:
            simulator = WorkflowSimulator(ResourceConfig())
            result = simulator.simulate_workflow(temp_file)

            assert result.success is True
            assert result.workflow_id == temp_file
            assert result.composition_number == 1
            assert result.total_events == 10000
            assert result.total_groups == 1
            assert result.total_jobs == 7  # Based on target wallclock time calculation
            assert result.total_wall_time > 0
            assert result.total_turnaround_time > 0
            assert len(result.groups) == 1
            assert len(result.jobs) == 7
            # execution_log removed

        finally:
            Path(temp_file).unlink()

    def test_simulate_workflow_file_not_found(self):
        """Test simulation with non-existent file."""
        simulator = WorkflowSimulator()
        result = simulator.simulate_workflow('nonexistent_file.json')

        assert result.success is False
        assert result.workflow_id == 'unknown'
        assert result.error_message is not None

    def test_simulate_workflow_invalid_json(self):
        """Test simulation with invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_file = f.name

        try:
            simulator = WorkflowSimulator()
            result = simulator.simulate_workflow(temp_file)

            assert result.success is False
            assert result.error_message is not None

        finally:
            Path(temp_file).unlink()

    def test_calculate_batch_size(self):
        """Test batch size calculation."""
        simulator = WorkflowSimulator(ResourceConfig(target_wallclock_time=3600.0))  # 1 hour

        # Create a group with tasksets
        taskset1 = TasksetInfo(
            taskset_id="Taskset1",
            group_name="group_1",
            input_taskset=None,
            time_per_event=10.0,
            memory=2000,
            multicore=1,
            size_per_event=200,
            group_input_events=100,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=False
        )

        taskset2 = TasksetInfo(
            taskset_id="Taskset2",
            group_name="group_1",
            input_taskset="Taskset1",
            time_per_event=20.0,
            memory=4000,
            multicore=2,
            size_per_event=300,
            group_input_events=100,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=True
        )

        group = GroupInfo(
            group_id="group_1",
            tasksets=[taskset1, taskset2],
            input_events=1000,
            job_count=0,
            exact_job_count=0.0,
            total_execution_time=0.0,
            dependencies=[]
        )

        # Total time per event = 10 + 20 = 30 seconds
        # Max events per job = 3600 / 30 = 120 events
        # But group input events is 1000, so should return min(120, 1000) = 120
        batch_size = simulator._calculate_batch_size(group)
        assert batch_size == 120

    def test_calculate_batch_size_empty_group(self):
        """Test batch size calculation with empty group."""
        simulator = WorkflowSimulator()
        group = GroupInfo(
            group_id="group_1",
            tasksets=[],
            input_events=1000,
            job_count=0,
            exact_job_count=0.0,
            total_execution_time=0.0,
            dependencies=[]
        )

        batch_size = simulator._calculate_batch_size(group)
        assert batch_size == 1000  # Should return input_events

    def test_calculate_job_wallclock_time(self):
        """Test job wallclock time calculation."""
        simulator = WorkflowSimulator()

        taskset1 = TasksetInfo(
            taskset_id="Taskset1",
            group_name="group_1",
            input_taskset=None,
            time_per_event=10.0,
            memory=2000,
            multicore=1,
            size_per_event=200,
            group_input_events=100,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=False
        )

        taskset2 = TasksetInfo(
            taskset_id="Taskset2",
            group_name="group_1",
            input_taskset="Taskset1",
            time_per_event=20.0,
            memory=4000,
            multicore=2,
            size_per_event=300,
            group_input_events=100,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=True
        )

        group = GroupInfo(
            group_id="group_1",
            tasksets=[taskset1, taskset2],
            input_events=1000,
            job_count=0,
            exact_job_count=0.0,
            total_execution_time=0.0,
            dependencies=[]
        )

        # Batch size as defined in the group configuration
        wallclock_time = simulator._calculate_job_wallclock_time(group, 1000)
        assert wallclock_time == 30000.0
        # Simulate a last job of the group with fractional batch size (hence,
        # less than 1000 events per job - make it 200 in this example)
        wallclock_time = simulator._calculate_job_wallclock_time(group, 200)
        assert wallclock_time == 6000.0

    def test_calculate_job_wallclock_time_empty_group(self):
        """Test job wallclock time calculation with empty group."""
        simulator = WorkflowSimulator()
        group = GroupInfo(
            group_id="group_1",
            tasksets=[],
            input_events=1000,
            job_count=0,
            exact_job_count=0.0,
            total_execution_time=0.0,
            dependencies=[]
        )

        wallclock_time = simulator._calculate_job_wallclock_time(group, 100)
        assert wallclock_time == 0.0

    def test_print_simulation_summary(self, capsys):
        """Test simulation summary printing."""
        # Create a mock simulation result
        taskset = TasksetInfo(
            taskset_id="Taskset1",
            group_name="group_1",
            input_taskset=None,
            time_per_event=10.0,
            memory=2000,
            multicore=1,
            size_per_event=200,
            group_input_events=100,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=False
        )

        group = GroupInfo(
            group_id="group_1",
            tasksets=[taskset],
            input_events=1000,
            job_count=10,
            exact_job_count=10.0,
            total_execution_time=10000.0,
            dependencies=[]
        )

        job = JobInfo(
            job_id="group_1_job_1",
            group_id="group_1",
            batch_size=1000,
            wallclock_time=10000.0,
            start_time=0.0,
            end_time=10000.0,
            status="completed"
        )

        result = SimulationResult(
            workflow_id="test_workflow",
            composition_number=1,
            total_events=10000,
            total_groups=1,
            total_jobs=10,
            total_wall_time=100000.0,
            total_turnaround_time=10000.0,
            groups=[group],
            jobs=[job],
            success=True
        )

        simulator = WorkflowSimulator()
        simulator.print_simulation_summary(result)

        captured = capsys.readouterr()
        assert "WORKFLOW SIMULATION SUMMARY" in captured.out
        assert "test_workflow" in captured.out
        assert "Total Events: 10,000" in captured.out
        assert "Total Groups: 1" in captured.out
        assert "Total Jobs: 10" in captured.out

    def test_write_simulation_result(self, tmp_path):
        """Test writing simulation result to file."""
        # Create a mock simulation result
        taskset = TasksetInfo(
            taskset_id="Taskset1",
            group_name="group_1",
            input_taskset=None,
            time_per_event=10.0,
            memory=2000,
            multicore=1,
            size_per_event=200,
            group_input_events=100,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=False
        )

        group = GroupInfo(
            group_id="group_1",
            tasksets=[taskset],
            input_events=1000,
            job_count=10,
            exact_job_count=10.0,
            total_execution_time=10000.0,
            dependencies=[]
        )

        result = SimulationResult(
            workflow_id="test_workflow",
            composition_number=1,
            total_events=10000,
            total_groups=1,
            total_jobs=10,
            total_wall_time=100000.0,
            total_turnaround_time=10000.0,
            groups=[group],
            jobs=[],
            success=True
        )

        simulator = WorkflowSimulator()
        output_file = tmp_path / "test_simulation.json"
        simulator.write_simulation_result(result, output_file)

        assert output_file.exists()

        # Verify file content
        with open(output_file, 'r') as f:
            data = json.load(f)

        assert data['workflow_id'] == "test_workflow"
        assert data['total_events'] == 10000
        assert data['total_groups'] == 1
        assert data['total_jobs'] == 10
        assert data['success'] is True

    def test_simulate_workflow_multigroup_jobs_persisted(self):
        """Ensure jobs from multiple independent groups are all persisted."""
        # Two independent groups (no dependencies), both should process full RequestNumEvents
        # Configure time such that batch size is predictable (10 events/job)
        workflow_data = {
            "Comments": "Multi-group test",
            "NumTasks": 2,
            "RequestNumEvents": 2000,
            # Group A
            "Taskset1": {
                "Memory": 1000,
                "Multicore": 1,
                "TimePerEvent": 100,  # seconds
                "SizePerEvent": 100,
                "GroupName": "group_A"
            },
            # Group B
            "Taskset2": {
                "Memory": 1000,
                "Multicore": 4,
                "TimePerEvent": 150,  # seconds
                "SizePerEvent": 100,
                "GroupName": "group_B"
            },
            "CompositionNumber": 2
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(workflow_data, f)
            temp_file = f.name

        try:
            # Use default ResourceConfig (target_wallclock_time=43200s) to minimize job count
            simulator = WorkflowSimulator(ResourceConfig())
            result = simulator.simulate_workflow(temp_file)

            assert result.success is True
            assert result.total_groups == 2
            # Default target 43200s:
            # Group A: 100 s/event -> batch size 432 -> ceil(2000/432)=5 jobs
            # Group B: 150 s/event -> batch size 288 -> ceil(2000/288)=7 jobs
            expected_jobs_group_a = 5
            expected_jobs_group_b = 7
            assert result.total_groups == 2
            assert sum(g.job_count for g in result.groups) == expected_jobs_group_a + expected_jobs_group_b
            assert result.total_jobs == expected_jobs_group_a + expected_jobs_group_b
            assert len(result.jobs) == expected_jobs_group_a + expected_jobs_group_b

            # Verify batch sizes per group (all but the final job have full batch size)
            jobs_group_a = [j for j in result.jobs if j.group_id == "group_A"]
            jobs_group_b = [j for j in result.jobs if j.group_id == "group_B"]

            # Expected batch sizes
            expected_batch_a = 432  # 43200 / 100 s per event
            expected_batch_b = 288  # 43200 / 150 s per event

            # Group A: 4 full-size jobs and 1 remainder of 272
            assert sum(1 for j in jobs_group_a if j.batch_size == expected_batch_a) == 4
            assert sum(1 for j in jobs_group_a if j.batch_size == 272) == 1

            # Group B: 6 full-size jobs and 1 remainder of 272
            assert sum(1 for j in jobs_group_b if j.batch_size == expected_batch_b) == 6
            assert sum(1 for j in jobs_group_b if j.batch_size == 272) == 1

            # Ensure jobs from both groups exist
            group_ids = {job.group_id for job in result.jobs}
            assert "group_A" in group_ids and "group_B" in group_ids
        finally:
            Path(temp_file).unlink()


if __name__ == "__main__":
    pytest.main([__file__])
