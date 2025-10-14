"""
Unit tests for workflow_metrics.py module.
Tests the WorkflowMetricsCalculator class and its methods.
"""

import json
import pytest
from pathlib import Path
import sys

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from workflow_metrics import WorkflowMetricsCalculator
from workflow_simulator import SimulationResult, GroupInfo, TasksetInfo, JobInfo


class TestWorkflowMetricsCalculator:
    """Test cases for WorkflowMetricsCalculator."""

    def test_initialization(self):
        """Test calculator initialization."""
        calculator = WorkflowMetricsCalculator()
        assert calculator.metrics is None

    def test_calculate_metrics_from_simulation(self):
        """Test metrics calculation from simulation result."""
        # Create a mock simulation result
        taskset1 = TasksetInfo(
            taskset_id="Taskset1",
            group_name="group_1",
            input_taskset=None,
            time_per_event=10.0,
            memory=2000,
            multicore=1,
            size_per_event=200,
            group_input_events=1080,
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
            group_input_events=1080,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=True
        )

        group = GroupInfo(
            group_id="group_1",
            tasksets=[taskset1, taskset2],
            input_events=1080,
            job_count=926,
            exact_job_count=925.9259259259259,
            total_execution_time=32400.0,
            dependencies=[]
        )

        job = JobInfo(
            job_id="group_1_job_1",
            group_id="group_1",
            batch_size=1080,
            wallclock_time=32400.0,
            start_time=0.0,
            end_time=32400.0,
            status="completed"
        )

        simulation_result = SimulationResult(
            workflow_id="test_workflow",
            composition_number=1,
            total_events=1000000,
            total_groups=1,
            total_jobs=926,
            total_wall_time=30024000.0,
            total_turnaround_time=32400.0,
            groups=[group],
            jobs=[job],
            execution_log=[],
            success=True
        )

        # Test metrics calculation from simulation
        calculator = WorkflowMetricsCalculator()
        metrics = calculator.calculate_metrics(simulation_result)

        assert metrics.workflow_id == "test_workflow"
        assert metrics.composition_number == 1
        assert metrics.total_tasksets == 2
        assert metrics.total_groups == 1
        assert metrics.total_jobs == 926
        assert metrics.total_wall_time == 30024000.0
        assert metrics.total_turnaround_time == 32400.0
        assert metrics.success_rate == 1.0
        assert len(metrics.group_metrics) == 1
        assert metrics.group_metrics[0].group_id == "group_1"
        assert metrics.group_metrics[0].job_count == 926

    def test_calculate_job_statistics(self):
        """Test job statistics calculation."""
        # Create mock simulation result with jobs
        taskset = TasksetInfo(
            taskset_id="Taskset1",
            group_name="group_1",
            input_taskset=None,
            time_per_event=10.0,
            memory=2000,
            multicore=1,
            size_per_event=200,
            group_input_events=1000,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=False
        )

        group = GroupInfo(
            group_id="group_1",
            tasksets=[taskset],
            input_events=1000,
            job_count=100,
            exact_job_count=100.0,
            total_execution_time=10000.0,
            dependencies=[]
        )

        # Create multiple jobs with different wall times and batch sizes
        jobs = []
        for i in range(3):
            job = JobInfo(
                job_id=f"group_1_job_{i+1}",
                group_id="group_1",
                batch_size=1000 + i * 100,  # Different batch sizes
                wallclock_time=10000.0 + i * 1000,  # Different wall times
                start_time=0.0,
                end_time=10000.0 + i * 1000,
                status="completed"
            )
            jobs.append(job)

        simulation_result = SimulationResult(
            workflow_id="test_workflow",
            composition_number=1,
            total_events=100000,
            total_groups=1,
            total_jobs=3,
            total_wall_time=33000.0,
            total_turnaround_time=12000.0,
            groups=[group],
            jobs=jobs,
            execution_log=[],
            success=True
        )

        calculator = WorkflowMetricsCalculator()
        job_stats = calculator.calculate_job_statistics(simulation_result)

        assert job_stats['total_jobs'] == 3
        assert job_stats['average_wall_time'] == 11000.0  # (10000 + 11000 + 12000) / 3
        assert job_stats['min_wall_time'] == 10000.0
        assert job_stats['max_wall_time'] == 12000.0
        assert job_stats['average_batch_size'] == 1100.0  # (1000 + 1100 + 1200) / 3
        assert job_stats['min_batch_size'] == 1000
        assert job_stats['max_batch_size'] == 1200

    def test_calculate_job_statistics_empty(self):
        """Test job statistics calculation with no jobs."""
        simulation_result = SimulationResult(
            workflow_id="test_workflow",
            composition_number=1,
            total_events=0,
            total_groups=0,
            total_jobs=0,
            total_wall_time=0.0,
            total_turnaround_time=0.0,
            groups=[],
            jobs=[],
            execution_log=[],
            success=True
        )

        calculator = WorkflowMetricsCalculator()
        job_stats = calculator.calculate_job_statistics(simulation_result)

        assert job_stats['total_jobs'] == 0
        assert job_stats['average_wall_time'] == 0.0
        assert job_stats['min_wall_time'] == 0.0
        assert job_stats['max_wall_time'] == 0.0

    def test_calculate_group_statistics(self):
        """Test group statistics calculation."""
        taskset1 = TasksetInfo(
            taskset_id="Taskset1",
            group_name="group_1",
            input_taskset=None,
            time_per_event=10.0,
            memory=2000,
            multicore=1,
            size_per_event=200,
            group_input_events=1000,
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
            group_input_events=1000,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=True
        )

        group = GroupInfo(
            group_id="group_1",
            tasksets=[taskset1, taskset2],
            input_events=1000,
            job_count=100,
            exact_job_count=100.0,
            total_execution_time=30000.0,
            dependencies=[]
        )

        simulation_result = SimulationResult(
            workflow_id="test_workflow",
            composition_number=1,
            total_events=100000,
            total_groups=1,
            total_jobs=100,
            total_wall_time=3000000.0,
            total_turnaround_time=30000.0,
            groups=[group],
            jobs=[],
            execution_log=[],
            success=True
        )

        calculator = WorkflowMetricsCalculator()
        group_stats = calculator.calculate_group_statistics(simulation_result)

        assert "group_1" in group_stats
        stats = group_stats["group_1"]
        assert stats['job_count'] == 100
        assert stats['input_events'] == 1000
        assert stats['total_execution_time'] == 30000.0
        assert stats['taskset_count'] == 2
        assert len(stats['tasksets']) == 2

        # Check taskset details
        taskset_data = stats['tasksets']
        assert taskset_data[0]['taskset_id'] == "Taskset1"
        assert taskset_data[0]['time_per_event'] == 10.0
        assert taskset_data[0]['memory'] == 2000
        assert taskset_data[1]['taskset_id'] == "Taskset2"
        assert taskset_data[1]['time_per_event'] == 20.0
        assert taskset_data[1]['memory'] == 4000

    def test_get_metrics_summary(self):
        """Test metrics summary generation."""
        # Create a simple simulation result
        taskset = TasksetInfo(
            taskset_id="Taskset1",
            group_name="group_1",
            input_taskset=None,
            time_per_event=10.0,
            memory=2000,
            multicore=1,
            size_per_event=200,
            group_input_events=1000,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=False
        )

        group = GroupInfo(
            group_id="group_1",
            tasksets=[taskset],
            input_events=1000,
            job_count=100,
            exact_job_count=100.0,
            total_execution_time=10000.0,
            dependencies=[]
        )

        simulation_result = SimulationResult(
            workflow_id="test_workflow",
            composition_number=1,
            total_events=100000,
            total_groups=1,
            total_jobs=100,
            total_wall_time=1000000.0,
            total_turnaround_time=10000.0,
            groups=[group],
            jobs=[],
            execution_log=[],
            success=True
        )

        calculator = WorkflowMetricsCalculator()
        calculator.calculate_metrics(simulation_result)
        summary = calculator.get_metrics_summary()

        required_keys = [
            'workflow_id', 'total_tasksets', 'total_groups', 'total_jobs',
            'total_wall_time', 'total_turnaround_time', 'resource_efficiency',
            'throughput', 'success_rate'
        ]

        for key in required_keys:
            assert key in summary

    def test_print_metrics(self, capsys):
        """Test metrics printing."""
        # Create a simple simulation result
        taskset = TasksetInfo(
            taskset_id="Taskset1",
            group_name="group_1",
            input_taskset=None,
            time_per_event=10.0,
            memory=2000,
            multicore=1,
            size_per_event=200,
            group_input_events=1000,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=False
        )

        group = GroupInfo(
            group_id="group_1",
            tasksets=[taskset],
            input_events=1000,
            job_count=100,
            exact_job_count=100.0,
            total_execution_time=10000.0,
            dependencies=[]
        )

        simulation_result = SimulationResult(
            workflow_id="test_workflow",
            composition_number=1,
            total_events=100000,
            total_groups=1,
            total_jobs=100,
            total_wall_time=1000000.0,
            total_turnaround_time=10000.0,
            groups=[group],
            jobs=[],
            execution_log=[],
            success=True
        )

        calculator = WorkflowMetricsCalculator()
        calculator.calculate_metrics(simulation_result)
        calculator.print_metrics()

        captured = capsys.readouterr()
        assert "WORKFLOW EXECUTION METRICS" in captured.out
        assert "Total Tasksets: 1" in captured.out
        assert "Total Groups: 1" in captured.out

    def test_write_metrics_to_file(self, tmp_path):
        """Test writing metrics to file."""
        # Create a simple simulation result
        taskset = TasksetInfo(
            taskset_id="Taskset1",
            group_name="group_1",
            input_taskset=None,
            time_per_event=10.0,
            memory=2000,
            multicore=1,
            size_per_event=200,
            group_input_events=1000,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=False
        )

        group = GroupInfo(
            group_id="group_1",
            tasksets=[taskset],
            input_events=1000,
            job_count=100,
            exact_job_count=100.0,
            total_execution_time=10000.0,
            dependencies=[]
        )

        simulation_result = SimulationResult(
            workflow_id="test_workflow",
            composition_number=1,
            total_events=100000,
            total_groups=1,
            total_jobs=100,
            total_wall_time=1000000.0,
            total_turnaround_time=10000.0,
            groups=[group],
            jobs=[],
            execution_log=[],
            success=True
        )

        calculator = WorkflowMetricsCalculator()
        calculator.calculate_metrics(simulation_result)

        output_file = tmp_path / "test_metrics.json"
        calculator.write_metrics_to_file(output_file)

        assert output_file.exists()

        # Verify file content
        with open(output_file, 'r') as f:
            data = json.load(f)

        assert data['total_tasksets'] == 1
        assert data['total_groups'] == 1
        assert data['total_jobs'] == 100


if __name__ == "__main__":
    pytest.main([__file__])
