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
            status="completed",
            total_cpu_time=32400.0,
            total_write_local_mb=216.0,  # (200 + 300) * 1080 / 1024 / 2
            total_write_remote_mb=158.2,  # 300 * 1080 / 1024 / 2 (only keep_output=True)
            total_read_remote_mb=0.0,  # No input taskset from other groups
            total_network_transfer_mb=158.2  # remote_write + remote_read
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

        # Test per-event metrics
        assert metrics.wall_time_per_event == 30.024  # 30024000.0 / 1000000
        assert metrics.cpu_time_per_event == 0.0324  # 32400.0 / 1000000
        assert metrics.network_transfer_mb_per_event == 0.0001582  # 158.2 / 1000000

        # Test aggregated job-level metrics
        assert metrics.total_cpu_time == 32400.0
        assert metrics.total_write_local_mb == 216.0
        assert metrics.total_write_remote_mb == 158.2
        assert metrics.total_read_remote_mb == 0.0
        assert metrics.total_network_transfer_mb == 158.2

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
                status="completed",
                total_cpu_time=10000.0 + i * 1000,  # Same as wallclock time for simplicity
                total_write_local_mb=195.3 + i * 19.5,  # 200 * batch_size / 1024
                total_write_remote_mb=0.0,  # keep_output=False
                total_read_remote_mb=0.0,  # No input taskset
                total_network_transfer_mb=0.0  # No remote operations
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

        # Test aggregated job-level metrics
        assert job_stats['total_cpu_time'] == 33000.0  # 10000 + 11000 + 12000
        assert abs(job_stats['total_write_local_mb'] - 644.4) < 0.1  # 195.3 + 214.8 + 234.4
        assert job_stats['total_write_remote_mb'] == 0.0
        assert job_stats['total_read_remote_mb'] == 0.0
        assert job_stats['total_network_transfer_mb'] == 0.0

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
            success=True
        )

        calculator = WorkflowMetricsCalculator()
        job_stats = calculator.calculate_job_statistics(simulation_result)

        assert job_stats['total_jobs'] == 0
        assert job_stats['average_wall_time'] == 0.0
        assert job_stats['min_wall_time'] == 0.0
        assert job_stats['max_wall_time'] == 0.0

        # Test aggregated job-level metrics for empty case
        assert job_stats['total_cpu_time'] == 0.0
        assert job_stats['total_write_local_mb'] == 0.0
        assert job_stats['total_write_remote_mb'] == 0.0
        assert job_stats['total_read_remote_mb'] == 0.0
        assert job_stats['total_network_transfer_mb'] == 0.0

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
            success=True
        )

        calculator = WorkflowMetricsCalculator()
        calculator.calculate_metrics(simulation_result)
        summary = calculator.get_metrics_summary()

        required_keys = [
            'workflow_id', 'total_tasksets', 'total_groups', 'total_jobs',
            'total_wall_time', 'total_turnaround_time', 'wall_time_per_event', 'cpu_time_per_event', 'network_transfer_mb_per_event',
 'event_throughput', 'success_rate', 'total_cpu_time', 'total_write_local_mb',
            'total_write_remote_mb', 'total_read_remote_mb', 'total_network_transfer_mb'
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
            success=True
        )

        calculator = WorkflowMetricsCalculator()
        calculator.calculate_metrics(simulation_result)
        calculator.print_metrics()

        captured = capsys.readouterr()
        assert "WORKFLOW EXECUTION METRICS" in captured.out
        assert "Total Tasksets: 1" in captured.out
        assert "Total Groups: 1" in captured.out
        assert "Wall Time per Event:" in captured.out
        assert "CPU Time per Event:" in captured.out
        assert "Network Transfer per Event:" in captured.out
        assert "AGGREGATED JOB METRICS" in captured.out
        assert "Total CPU Time:" in captured.out
        assert "Total Write Local:" in captured.out
        assert "Total Write Remote:" in captured.out
        assert "Total Read Remote:" in captured.out
        assert "Total Network Transfer:" in captured.out

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

    def test_aggregated_job_level_metrics(self):
        """Test comprehensive aggregated job-level metrics calculation."""
        # Create tasksets with different characteristics
        taskset1 = TasksetInfo(
            taskset_id="Taskset1",
            group_name="group_1",
            input_taskset=None,
            time_per_event=5.0,
            memory=1000,
            multicore=2,
            size_per_event=100,  # 100 KB
            group_input_events=1000,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=False  # Local write only
        )

        taskset2 = TasksetInfo(
            taskset_id="Taskset2",
            group_name="group_1",
            input_taskset="Taskset1",
            time_per_event=10.0,
            memory=2000,
            multicore=1,
            size_per_event=200,  # 200 KB
            group_input_events=1000,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=True  # Remote write
        )

        group = GroupInfo(
            group_id="group_1",
            tasksets=[taskset1, taskset2],
            input_events=500,
            job_count=2,
            exact_job_count=2.0,
            total_execution_time=7500.0,  # (5 + 10) * 500
            dependencies=[]
        )

        # Create jobs with different batch sizes
        jobs = []
        for i in range(2):
            batch_size = 500 + i * 100  # 500, 600
            job = JobInfo(
                job_id=f"group_1_job_{i+1}",
                group_id="group_1",
                batch_size=batch_size,
                wallclock_time=15.0 * batch_size,  # (5 + 10) * batch_size
                start_time=0.0,
                end_time=15.0 * batch_size,
                status="completed",
                # Calculate expected metrics
                total_cpu_time=15.0 * batch_size,  # (5 * 2 + 10 * 1) * batch_size
                total_write_local_mb=(100 + 200) * batch_size / 1024.0,  # Both tasksets write locally
                total_write_remote_mb=200 * batch_size / 1024.0,  # Only taskset2 (keep_output=True)
                total_read_remote_mb=0.0,  # No input from other groups
                total_network_transfer_mb=200 * batch_size / 1024.0  # remote_write + remote_read
            )
            jobs.append(job)

        simulation_result = SimulationResult(
            workflow_id="test_workflow",
            composition_number=1,
            total_events=1000,
            total_groups=1,
            total_jobs=2,
            total_wall_time=16500.0,  # 15 * 500 + 15 * 600
            total_turnaround_time=9000.0,  # Max of the two jobs
            groups=[group],
            jobs=jobs,
            success=True
        )

        calculator = WorkflowMetricsCalculator()
        metrics = calculator.calculate_metrics(simulation_result)

        # Test aggregated job-level metrics
        expected_cpu_time = 15.0 * 500 + 15.0 * 600  # 16500.0
        expected_write_local = (100 + 200) * 500 / 1024.0 + (100 + 200) * 600 / 1024.0  # ~439.45
        expected_write_remote = 200 * 500 / 1024.0 + 200 * 600 / 1024.0  # ~214.84
        expected_read_remote = 0.0
        expected_network_transfer = expected_write_remote + expected_read_remote  # ~214.84

        assert abs(metrics.total_cpu_time - expected_cpu_time) < 0.01
        assert abs(metrics.total_write_local_mb - expected_write_local) < 0.01
        assert abs(metrics.total_write_remote_mb - expected_write_remote) < 0.01
        assert metrics.total_read_remote_mb == expected_read_remote
        assert abs(metrics.total_network_transfer_mb - expected_network_transfer) < 0.01

        # Test that metrics are included in summary
        summary = calculator.get_metrics_summary()
        assert 'total_cpu_time' in summary
        assert 'total_write_local_mb' in summary
        assert 'total_write_remote_mb' in summary
        assert 'total_read_remote_mb' in summary
        assert 'total_network_transfer_mb' in summary

        # Test that values match
        assert abs(summary['total_cpu_time'] - expected_cpu_time) < 0.01
        assert abs(summary['total_write_local_mb'] - expected_write_local) < 0.01
        assert abs(summary['total_write_remote_mb'] - expected_write_remote) < 0.01
        assert summary['total_read_remote_mb'] == expected_read_remote
        assert abs(summary['total_network_transfer_mb'] - expected_network_transfer) < 0.01

    def test_calculate_resource_utilization_from_simulation(self):
        """Test resource utilization calculation from simulation result."""
        # Create mock tasksets with different resource requirements
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

        taskset3 = TasksetInfo(
            taskset_id="Taskset3",
            group_name="group_1",
            input_taskset="Taskset2",
            time_per_event=10.0,
            memory=1000,
            multicore=2,
            size_per_event=150,
            group_input_events=1080,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=False
        )

        group = GroupInfo(
            group_id="group_1",
            tasksets=[taskset1, taskset2, taskset3],
            input_events=1080,
            job_count=2,
            exact_job_count=2.0,
            total_execution_time=43200.0,  # (10 + 20 + 10) * 1080
            dependencies=[]
        )

        # Create mock jobs with different batch sizes
        job1 = JobInfo(
            job_id="group_1_job_1",
            group_id="group_1",
            batch_size=1080,
            wallclock_time=43200.0,  # (10 + 20 + 10) * 1080
            start_time=0.0,
            end_time=43200.0,
            status="completed",
            total_cpu_time=75600.0,  # (1080*10*1) + (1080*20*2) + (1080*10*2) = 10800 + 43200 + 21600
            total_write_local_mb=216.0,
            total_write_remote_mb=158.2,
            total_read_remote_mb=0.0,
            total_read_local_mb=0.0,
            total_network_transfer_mb=158.2
        )

        job2 = JobInfo(
            job_id="group_1_job_2",
            group_id="group_1",
            batch_size=1000,
            wallclock_time=40000.0,  # (10 + 20 + 10) * 1000
            start_time=43200.0,
            end_time=83200.0,
            status="completed",
            total_cpu_time=70000.0,  # (1000*10*1) + (1000*20*2) + (1000*10*2) = 10000 + 40000 + 20000
            total_write_local_mb=200.0,
            total_write_remote_mb=146.5,
            total_read_remote_mb=0.0,
            total_read_local_mb=0.0,
            total_network_transfer_mb=146.5
        )

        simulation_result = SimulationResult(
            workflow_id="test_workflow",
            composition_number=1,
            total_events=2080,  # 1080 + 1000
            total_groups=1,
            total_jobs=2,
            total_wall_time=83200.0,
            total_turnaround_time=43200.0,
            groups=[group],
            jobs=[job1, job2],
            success=True
        )

        # Test resource utilization calculation
        calculator = WorkflowMetricsCalculator()
        resource_usage = calculator._calculate_resource_utilization_from_simulation(simulation_result)

        # Verify resource usage object is created
        assert resource_usage is not None
        assert hasattr(resource_usage, 'cpu_usage')
        assert hasattr(resource_usage, 'memory_usage')
        assert hasattr(resource_usage, 'storage_usage')
        assert hasattr(resource_usage, 'network_usage')
        assert hasattr(resource_usage, 'cpu_utilization')
        assert hasattr(resource_usage, 'memory_occupancy')

        # Test CPU utilization calculation
        # Expected: average of job utilizations
        # Job 1: CPU used = 75600, CPU allocated = 1080 * 40 * 2 = 86400, utilization = 75600/86400 = 0.875
        # Job 2: CPU used = 70000, CPU allocated = 1000 * 40 * 2 = 80000, utilization = 70000/80000 = 0.875
        # Average: (0.875 + 0.875) / 2 = 0.875
        expected_cpu_utilization = 0.875
        assert abs(resource_usage.cpu_utilization - expected_cpu_utilization) < 0.001

        # Test memory occupancy calculation
        # Expected: based on the actual implementation formula
        # Memory used = sum(group.input_events * taskset.time_per_event * taskset.memory)
        #            = 1080 * (10*2000 + 20*4000 + 10*1000) = 1080 * (20000 + 80000 + 10000) = 1080 * 110000 = 118800000
        # Memory allocated = group.input_events * group_time_per_event * max_group_memory
        #                  = 1080 * 40 * 4000 = 172800000
        # Utilization = 118800000 / 172800000 = 0.6875
        expected_memory_occupancy = 0.6875
        assert abs(resource_usage.memory_occupancy - expected_memory_occupancy) < 0.001

        # Test resource usage totals
        # Total CPU cores used = max_cores * num_jobs = 2 * 2 = 4
        expected_cpu_cores = 4
        assert resource_usage.cpu_usage == expected_cpu_cores

        # Total memory used = max_memory * num_jobs = 4000 * 2 = 8000
        expected_memory = 8000
        assert resource_usage.memory_usage == expected_memory

        # Network usage should match total network transfer
        expected_network = 158.2 + 146.5  # 304.7
        assert abs(resource_usage.network_usage - expected_network) < 0.01

    def test_calculate_resource_utilization_empty_simulation(self):
        """Test resource utilization calculation with empty simulation."""
        # Create empty simulation result
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
            success=True
        )

        calculator = WorkflowMetricsCalculator()
        resource_usage = calculator._calculate_resource_utilization_from_simulation(simulation_result)

        # Should return default values
        assert resource_usage.cpu_usage == 0.0
        assert resource_usage.memory_usage == 0.0
        assert resource_usage.storage_usage == 0.0
        assert resource_usage.network_usage == 0.0
        assert resource_usage.cpu_utilization == 0.0
        assert resource_usage.memory_occupancy == 0.0

    def test_calculate_resource_utilization_multiple_groups(self):
        """Test resource utilization calculation with multiple groups."""
        # Create two groups with different resource requirements
        taskset1 = TasksetInfo(
            taskset_id="Taskset1",
            group_name="group_1",
            input_taskset=None,
            time_per_event=5.0,
            memory=1000,
            multicore=1,
            size_per_event=100,
            group_input_events=500,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=False
        )

        taskset2 = TasksetInfo(
            taskset_id="Taskset2",
            group_name="group_2",
            input_taskset=None,
            time_per_event=15.0,
            memory=3000,
            multicore=3,
            size_per_event=200,
            group_input_events=300,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=False
        )

        group1 = GroupInfo(
            group_id="group_1",
            tasksets=[taskset1],
            input_events=500,
            job_count=1,
            exact_job_count=1.0,
            total_execution_time=2500.0,  # 5 * 500
            dependencies=[]
        )

        group2 = GroupInfo(
            group_id="group_2",
            tasksets=[taskset2],
            input_events=300,
            job_count=1,
            exact_job_count=1.0,
            total_execution_time=4500.0,  # 15 * 300
            dependencies=[]
        )

        job1 = JobInfo(
            job_id="group_1_job_1",
            group_id="group_1",
            batch_size=500,
            wallclock_time=2500.0,
            start_time=0.0,
            end_time=2500.0,
            status="completed",
            total_cpu_time=2500.0,  # 500 * 5 * 1
            total_write_local_mb=50.0,
            total_write_remote_mb=0.0,
            total_read_remote_mb=0.0,
            total_read_local_mb=0.0,
            total_network_transfer_mb=0.0
        )

        job2 = JobInfo(
            job_id="group_2_job_1",
            group_id="group_2",
            batch_size=300,
            wallclock_time=4500.0,
            start_time=2500.0,
            end_time=7000.0,
            status="completed",
            total_cpu_time=13500.0,  # 300 * 15 * 3
            total_write_local_mb=60.0,
            total_write_remote_mb=0.0,
            total_read_remote_mb=0.0,
            total_read_local_mb=0.0,
            total_network_transfer_mb=0.0
        )

        simulation_result = SimulationResult(
            workflow_id="test_workflow",
            composition_number=1,
            total_events=800,  # 500 + 300
            total_groups=2,
            total_jobs=2,
            total_wall_time=7000.0,
            total_turnaround_time=4500.0,
            groups=[group1, group2],
            jobs=[job1, job2],
            success=True
        )

        calculator = WorkflowMetricsCalculator()
        resource_usage = calculator._calculate_resource_utilization_from_simulation(simulation_result)

        # Test CPU utilization calculation
        # Group 1: CPU used = 2500, CPU allocated = 500 * 5 * 1 = 2500, utilization = 1.0
        # Group 2: CPU used = 13500, CPU allocated = 300 * 15 * 3 = 13500, utilization = 1.0
        # Average: (1.0 + 1.0) / 2 = 1.0
        expected_cpu_utilization = 1.0
        assert abs(resource_usage.cpu_utilization - expected_cpu_utilization) < 0.001

        # Test memory occupancy calculation
        # Group 1: Memory used = 1000 * 2500 = 2500000, Memory allocated = 1000 * 2500 = 2500000, utilization = 1.0
        # Group 2: Memory used = 3000 * 4500 = 13500000, Memory allocated = 3000 * 4500 = 13500000, utilization = 1.0
        # Average: (1.0 + 1.0) / 2 = 1.0
        expected_memory_occupancy = 1.0
        assert abs(resource_usage.memory_occupancy - expected_memory_occupancy) < 0.001

        # Test resource usage totals
        # Total CPU cores = 1 + 3 = 4
        expected_cpu_cores = 4
        assert resource_usage.cpu_usage == expected_cpu_cores

        # Total memory = 1000 + 3000 = 4000
        expected_memory = 4000
        assert resource_usage.memory_usage == expected_memory


if __name__ == "__main__":
    pytest.main([__file__])
