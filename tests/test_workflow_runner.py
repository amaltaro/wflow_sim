"""
Unit tests for workflow_runner.py module.
Tests the WorkflowRunner class and its methods.
"""

import json
import pytest
from pathlib import Path
import sys
import tempfile

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from workflow_runner import WorkflowRunner
from workflow_simulator import ResourceConfig, SimulationResult, GroupInfo, TasksetInfo, JobInfo


class TestWorkflowRunner:
    """Test cases for WorkflowRunner."""

    def test_initialization(self):
        """Test runner initialization."""
        runner = WorkflowRunner()
        assert runner.resource_config is not None
        assert runner.simulator is not None
        assert runner.logger is not None
        # test default resource configs
        assert runner.resource_config.target_wallclock_time == 43200.0
        assert runner.resource_config.max_job_slots == -1
        assert runner.resource_config.cpu_per_slot == 1
        assert runner.resource_config.memory_per_slot == 1000

    def test_initialization_with_config(self):
        """Test runner initialization with custom config."""
        config = ResourceConfig(
            target_wallclock_time=21600.0,  # 6 hours
            max_job_slots=50,
            cpu_per_slot=2,
            memory_per_slot=2000
        )
        runner = WorkflowRunner(config)
        assert runner.resource_config.target_wallclock_time == 21600.0
        assert runner.resource_config.max_job_slots == 50

    def test_run_workflow_success(self):
        """Test successful workflow execution."""
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
            runner = WorkflowRunner(ResourceConfig())
            results = runner.run_workflow(temp_file)

            assert results['success'] is True
            assert results['error_message'] is None
            assert results['simulation_result'] is not None
            assert results['metrics'] is not None

            # Check simulation result
            simulation = results['simulation_result']
            assert simulation.success is True
            assert simulation.total_events == 10000
            assert simulation.total_groups == 1
            assert simulation.total_jobs == 7

            # Check metrics
            metrics = results['metrics']
            assert metrics.total_tasksets == 2
            assert metrics.total_groups == 1
            assert metrics.total_jobs == 7
            assert metrics.success_rate == 1.0

        finally:
            Path(temp_file).unlink()

    def test_run_workflow_file_not_found(self):
        """Test workflow execution with non-existent file."""
        runner = WorkflowRunner()
        results = runner.run_workflow('nonexistent_file.json')

        assert results['success'] is False
        assert results['error_message'] is not None
        assert results['simulation_result'] is not None
        assert results['metrics'] is None

    def test_print_complete_summary_success(self, capsys):
        """Test printing complete summary for successful workflow."""
        # Create mock results
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
            exact_job_count=9.8,
            total_execution_time=9800.0,
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

        simulation = SimulationResult(
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

        # Create mock metrics (simplified)
        class MockMetrics:
            def __init__(self):
                self.resource_efficiency = 0.5
                self.event_throughput = 1.0
                self.success_rate = 1.0
                self.wall_time_per_event = 10.0
                self.cpu_time_per_event = 5.0
                self.network_transfer_mb_per_event = 0.1

        results = {
            'success': True,
            'error_message': None,
            'simulation_result': simulation,
            'metrics': MockMetrics()
        }

        runner = WorkflowRunner()
        runner.print_complete_summary(results)

        captured = capsys.readouterr()
        assert "COMPLETE WORKFLOW EXECUTION SUMMARY" in captured.out
        assert "test_workflow" in captured.out
        assert "Total Events: 10,000" in captured.out
        assert "Total Groups: 1" in captured.out
        assert "Total Jobs: 10" in captured.out
        assert "Resource Efficiency: 0.50" in captured.out
        assert "Event Throughput: 1.000000 events/CPU-second" in captured.out
        assert "Network Transfer per Event: 0.100000 MB/event" in captured.out

    def test_print_complete_summary_failure(self, capsys):
        """Test printing complete summary for failed workflow."""
        results = {
            'success': False,
            'error_message': 'Test error message',
            'simulation_result': None,
            'metrics': None
        }

        runner = WorkflowRunner()
        runner.print_complete_summary(results)

        captured = capsys.readouterr()
        assert "❌ Workflow execution failed: Test error message" in captured.out

    def test_write_complete_results(self, tmp_path):
        """Test writing complete results to file."""
        # Create mock results
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

        simulation = SimulationResult(
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

        # Create mock metrics
        class MockMetrics:
            def __init__(self):
                self.workflow_id = "test_workflow"
                self.composition_number = 1
                self.total_events = 10000
                self.total_tasksets = 1
                self.total_groups = 1
                self.total_jobs = 10
                self.total_wall_time = 100000.0
                self.total_turnaround_time = 10000.0
                self.wall_time_per_event = 10.0
                self.cpu_time_per_event = 5.0
                self.network_transfer_mb_per_event = 0.1
                self.resource_efficiency = 0.5
                self.event_throughput = 1.0
                self.success_rate = 1.0
                self.total_cpu_time = 50000.0
                self.total_write_local_mb = 1000.0
                self.total_write_remote_mb = 500.0
                self.total_read_remote_mb = 200.0
                self.total_read_local_mb = 300.0
                self.total_network_transfer_mb = 700.0
                self.total_write_local_mb_per_event = 0.1
                self.total_write_remote_mb_per_event = 0.05
                self.total_read_remote_mb_per_event = 0.02
                self.total_read_local_mb_per_event = 0.03

        results = {
            'success': True,
            'error_message': None,
            'simulation_result': simulation,
            'metrics': MockMetrics()
        }

        runner = WorkflowRunner()
        output_file = tmp_path / "test_complete_results.json"
        runner.write_complete_results(results, output_file)

        assert output_file.exists()

        # Verify file content
        with open(output_file, 'r') as f:
            data = json.load(f)

        assert 'simulation_result' in data
        assert 'metrics' in data

        # Check simulation result structure
        sim_result = data['simulation_result']
        assert sim_result['success'] is True
        assert sim_result['error_message'] is None
        assert sim_result['groups'][0]['group_id'] == "group_1"
        assert len(sim_result['groups']) == 1
        assert len(sim_result['jobs']) == 1

        # Check metrics structure
        metrics = data['metrics']
        assert metrics['workflow_id'] == "test_workflow"
        assert metrics['total_tasksets'] == 1
        assert metrics['total_groups'] == 1
        assert metrics['total_jobs'] == 10
        assert metrics['resource_efficiency'] == 0.5
        assert metrics['event_throughput'] == 1.0
        assert metrics['network_transfer_mb_per_event'] == 0.1
        assert metrics['success_rate'] == 1.0

        # Check new aggregated job-level metrics
        assert metrics['total_cpu_time'] == 50000.0
        assert metrics['total_write_local_mb'] == 1000.0
        assert metrics['total_write_remote_mb'] == 500.0
        assert metrics['total_read_remote_mb'] == 200.0
        assert metrics['total_network_transfer_mb'] == 700.0


if __name__ == "__main__":
    pytest.main([__file__])
