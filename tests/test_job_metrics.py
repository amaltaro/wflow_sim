"""
Unit tests for job_metrics.py module.
Tests the JobMetricsCalculator class and JobMetrics dataclass.
"""

import pytest
from pathlib import Path
import sys

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from job_metrics import JobMetrics, JobMetricsCalculator
from workflow_simulator import TasksetInfo


class TestJobMetrics:
    """Test cases for JobMetrics dataclass."""

    def test_job_metrics_initialization(self):
        """Test JobMetrics dataclass initialization."""
        metrics = JobMetrics(
            total_cpu_time=1000.0,
            total_write_local_mb=500.0,
            total_write_remote_mb=200.0,
            total_read_remote_mb=150.0,
            total_read_local_mb=100.0,
            total_network_transfer_mb=350.0
        )

        assert metrics.total_cpu_time == 1000.0
        assert metrics.total_write_local_mb == 500.0
        assert metrics.total_write_remote_mb == 200.0
        assert metrics.total_read_remote_mb == 150.0
        assert metrics.total_read_local_mb == 100.0
        assert metrics.total_network_transfer_mb == 350.0

    def test_job_metrics_default_values(self):
        """Test JobMetrics with default values."""
        metrics = JobMetrics(
            total_cpu_time=0.0,
            total_write_local_mb=0.0,
            total_write_remote_mb=0.0,
            total_read_remote_mb=0.0,
            total_read_local_mb=0.0,
            total_network_transfer_mb=0.0
        )

        assert metrics.total_cpu_time == 0.0
        assert metrics.total_write_local_mb == 0.0
        assert metrics.total_write_remote_mb == 0.0
        assert metrics.total_read_remote_mb == 0.0
        assert metrics.total_read_local_mb == 0.0
        assert metrics.total_network_transfer_mb == 0.0


class TestJobMetricsCalculator:
    """Test cases for JobMetricsCalculator class."""

    def test_initialization(self):
        """Test calculator initialization."""
        calculator = JobMetricsCalculator()
        assert calculator.logger is not None

    def test_calculate_job_metrics_basic(self):
        """Test basic job metrics calculation."""
        calculator = JobMetricsCalculator()

        # Create a simple taskset
        taskset = TasksetInfo(
            taskset_id="Taskset1",
            group_name="group_1",
            input_taskset=None,
            time_per_event=10.0,
            memory=2000,
            multicore=2,
            size_per_event=200,  # 200 KB
            group_input_events=1000,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=False
        )

        job_metrics = calculator.calculate_job_metrics(
            tasksets=[taskset],
            batch_size=1000
        )

        # CPU time: 10.0 * 1000 * 2 = 20000.0
        assert job_metrics.total_cpu_time == 20000.0
        # Local write: (200 * 1000) / 1024 = 195.31 MB
        assert abs(job_metrics.total_write_local_mb - 195.31) < 0.01
        # Remote write: 0 (keep_output=False and not input for other groups)
        assert job_metrics.total_write_remote_mb == 0.0
        # Remote read: 0 (no input taskset)
        assert job_metrics.total_read_remote_mb == 0.0
        # Network transfer: 0 + 0 = 0
        assert job_metrics.total_network_transfer_mb == 0.0

    def test_calculate_job_metrics_with_keep_output(self):
        """Test job metrics with keep_output=True."""
        calculator = JobMetricsCalculator()

        taskset = TasksetInfo(
            taskset_id="Taskset1",
            group_name="group_1",
            input_taskset=None,
            time_per_event=5.0,
            memory=1000,
            multicore=1,
            size_per_event=300,  # 300 KB
            group_input_events=500,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=True  # This should trigger remote write
        )

        job_metrics = calculator.calculate_job_metrics(
            tasksets=[taskset],
            batch_size=500
        )

        # CPU time: 5.0 * 500 * 1 = 2500.0
        assert job_metrics.total_cpu_time == 2500.0
        # Local write: (300 * 500) / 1024 = 146.48 MB
        assert abs(job_metrics.total_write_local_mb - 146.48) < 0.01
        # Remote write: same as local write (keep_output=True)
        assert abs(job_metrics.total_write_remote_mb - 146.48) < 0.01
        # Remote read: 0 (no input taskset)
        assert job_metrics.total_read_remote_mb == 0.0
        # Network transfer: 146.48 + 0 = 146.48
        assert abs(job_metrics.total_network_transfer_mb - 146.48) < 0.01

    def test_calculate_job_metrics_with_remote_read(self):
        """Test job metrics with remote read (input taskset from other group)."""
        calculator = JobMetricsCalculator()

        taskset = TasksetInfo(
            taskset_id="Taskset2",
            group_name="group_2",
            input_taskset="Taskset1",
            time_per_event=15.0,
            memory=3000,
            multicore=3,
            size_per_event=400,  # 400 KB
            group_input_events=800,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=False
        )

        job_metrics = calculator.calculate_job_metrics(
            tasksets=[taskset],
            batch_size=800,
            input_tasksets_for_other_groups=set(),  # Not input for other groups
            input_taskset_size_per_event=250  # 250 KB from input taskset
        )

        # CPU time: 15.0 * 800 * 3 = 36000.0
        assert job_metrics.total_cpu_time == 36000.0
        # Local write: (400 * 800) / 1024 = 312.5 MB
        assert job_metrics.total_write_local_mb == 312.5
        # Remote write: 0 (keep_output=False and not input for other groups)
        assert job_metrics.total_write_remote_mb == 0.0
        # Remote read: (250 * 800) / 1024 = 195.31 MB
        assert abs(job_metrics.total_read_remote_mb - 195.31) < 0.01
        # Network transfer: 0 + 195.31 = 195.31
        assert abs(job_metrics.total_network_transfer_mb - 195.31) < 0.01

    def test_calculate_job_metrics_input_for_other_groups(self):
        """Test job metrics with taskset that is input for other groups."""
        calculator = JobMetricsCalculator()

        taskset = TasksetInfo(
            taskset_id="Taskset1",
            group_name="group_1",
            input_taskset=None,
            time_per_event=8.0,
            memory=1500,
            multicore=2,
            size_per_event=350,  # 350 KB
            group_input_events=600,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=False
        )

        job_metrics = calculator.calculate_job_metrics(
            tasksets=[taskset],
            batch_size=600,
            input_tasksets_for_other_groups={"Taskset1"},  # This taskset is input for other groups
            input_taskset_size_per_event=None
        )

        # CPU time: 8.0 * 600 * 2 = 9600.0
        assert job_metrics.total_cpu_time == 9600.0
        # Local write: (350 * 600) / 1024 = 205.08 MB
        assert abs(job_metrics.total_write_local_mb - 205.08) < 0.01
        # Remote write: same as local write (input for other groups)
        assert abs(job_metrics.total_write_remote_mb - 205.08) < 0.01
        # Remote read: 0 (no input taskset size provided)
        assert job_metrics.total_read_remote_mb == 0.0
        # Network transfer: 205.08 + 0 = 205.08
        assert abs(job_metrics.total_network_transfer_mb - 205.08) < 0.01

    def test_calculate_job_metrics_multiple_tasksets(self):
        """Test job metrics with multiple tasksets."""
        calculator = JobMetricsCalculator()

        taskset1 = TasksetInfo(
            taskset_id="Taskset1",
            group_name="group_1",
            input_taskset=None,
            time_per_event=5.0,
            memory=1000,
            multicore=1,
            size_per_event=100,  # 100 KB
            group_input_events=200,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=False
        )

        taskset2 = TasksetInfo(
            taskset_id="Taskset2",
            group_name="group_1",
            input_taskset="Taskset1",
            time_per_event=10.0,
            memory=2000,
            multicore=2,
            size_per_event=200,  # 200 KB
            group_input_events=200,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=True
        )

        job_metrics = calculator.calculate_job_metrics(
            tasksets=[taskset1, taskset2],
            batch_size=200,
            input_tasksets_for_other_groups=set(),
            input_taskset_size_per_event=None
        )

        # CPU time: (5.0 * 200 * 1) + (10.0 * 200 * 2) = 1000 + 4000 = 5000.0
        assert job_metrics.total_cpu_time == 5000.0
        # Local write: (100 * 200) / 1024 + (200 * 200) / 1024 = 19.53 + 39.06 = 58.59 MB
        assert abs(job_metrics.total_write_local_mb - 58.59) < 0.01
        # Remote write: only taskset2 (keep_output=True) = 39.06 MB
        assert abs(job_metrics.total_write_remote_mb - 39.06) < 0.01
        # Remote read: 0 (no input taskset size provided)
        assert job_metrics.total_read_remote_mb == 0.0
        # Network transfer: 39.06 + 0 = 39.06
        assert abs(job_metrics.total_network_transfer_mb - 39.06) < 0.01

    def test_calculate_job_metrics_complete_scenario(self):
        """Test complete scenario with remote read and remote write."""
        calculator = JobMetricsCalculator()

        taskset = TasksetInfo(
            taskset_id="Taskset2",
            group_name="group_2",
            input_taskset="Taskset1",
            time_per_event=12.0,
            memory=2500,
            multicore=2,
            size_per_event=500,  # 500 KB
            group_input_events=1000,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=True
        )

        job_metrics = calculator.calculate_job_metrics(
            tasksets=[taskset],
            batch_size=1000,
            input_tasksets_for_other_groups=set(),
            input_taskset_size_per_event=300  # 300 KB from input taskset
        )

        # CPU time: 12.0 * 1000 * 2 = 24000.0
        assert job_metrics.total_cpu_time == 24000.0
        # Local write: (500 * 1000) / 1024 = 488.28 MB
        assert abs(job_metrics.total_write_local_mb - 488.28) < 0.01
        # Remote write: same as local write (keep_output=True)
        assert abs(job_metrics.total_write_remote_mb - 488.28) < 0.01
        # Remote read: (300 * 1000) / 1024 = 292.97 MB
        assert abs(job_metrics.total_read_remote_mb - 292.97) < 0.01
        # Network transfer: 488.28 + 292.97 = 781.25 MB
        assert abs(job_metrics.total_network_transfer_mb - 781.25) < 0.01

    def test_calculate_job_metrics_empty_tasksets(self):
        """Test job metrics with empty tasksets list."""
        calculator = JobMetricsCalculator()

        job_metrics = calculator.calculate_job_metrics(
            tasksets=[],
            batch_size=1000
        )

        assert job_metrics.total_cpu_time == 0.0
        assert job_metrics.total_write_local_mb == 0.0
        assert job_metrics.total_write_remote_mb == 0.0
        assert job_metrics.total_read_remote_mb == 0.0
        assert job_metrics.total_network_transfer_mb == 0.0

    def test_calculate_job_metrics_zero_batch_size(self):
        """Test job metrics with zero batch size."""
        calculator = JobMetricsCalculator()

        taskset = TasksetInfo(
            taskset_id="Taskset1",
            group_name="group_1",
            input_taskset=None,
            time_per_event=10.0,
            memory=2000,
            multicore=2,
            size_per_event=200,
            group_input_events=1000,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=True
        )

        job_metrics = calculator.calculate_job_metrics(
            tasksets=[taskset],
            batch_size=0
        )

        assert job_metrics.total_cpu_time == 0.0
        assert job_metrics.total_write_local_mb == 0.0
        assert job_metrics.total_write_remote_mb == 0.0
        assert job_metrics.total_read_remote_mb == 0.0
        assert job_metrics.total_network_transfer_mb == 0.0

    def test_calculate_job_metrics_size_conversion(self):
        """Test that SizePerEvent is correctly converted from KB to MB."""
        calculator = JobMetricsCalculator()

        taskset = TasksetInfo(
            taskset_id="Taskset1",
            group_name="group_1",
            input_taskset=None,
            time_per_event=1.0,
            memory=1000,
            multicore=1,
            size_per_event=1024,  # 1024 KB = 1 MB
            group_input_events=1000,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=True
        )

        job_metrics = calculator.calculate_job_metrics(
            tasksets=[taskset],
            batch_size=1000
        )

        # Local write: (1024 * 1000) / 1024 = 1000.0 MB
        assert job_metrics.total_write_local_mb == 1000.0
        # Remote write: same as local write
        assert job_metrics.total_write_remote_mb == 1000.0

    def test_calculate_job_metrics_remote_read_size_conversion(self):
        """Test that remote read SizePerEvent is correctly converted from KB to MB."""
        calculator = JobMetricsCalculator()

        taskset = TasksetInfo(
            taskset_id="Taskset1",
            group_name="group_1",
            input_taskset="InputTaskset",
            time_per_event=1.0,
            memory=1000,
            multicore=1,
            size_per_event=100,  # 100 KB
            group_input_events=1000,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=False
        )

        job_metrics = calculator.calculate_job_metrics(
            tasksets=[taskset],
            batch_size=1000,
            input_tasksets_for_other_groups=set(),
            input_taskset_size_per_event=2048  # 2048 KB = 2 MB
        )

        # Remote read: (2048 * 1000) / 1024 = 2000.0 MB
        assert job_metrics.total_read_remote_mb == 2000.0
        # Network transfer: 0 + 2000.0 = 2000.0
        assert job_metrics.total_network_transfer_mb == 2000.0

    def test_calculate_job_metrics_none_input_tasksets_for_other_groups(self):
        """Test that None input_tasksets_for_other_groups is handled correctly."""
        calculator = JobMetricsCalculator()

        taskset = TasksetInfo(
            taskset_id="Taskset1",
            group_name="group_1",
            input_taskset=None,
            time_per_event=5.0,
            memory=1000,
            multicore=1,
            size_per_event=200,
            group_input_events=500,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=False
        )

        # Test with None input_tasksets_for_other_groups
        job_metrics = calculator.calculate_job_metrics(
            tasksets=[taskset],
            batch_size=500,
            input_tasksets_for_other_groups=None
        )

        assert job_metrics.total_cpu_time == 2500.0
        assert job_metrics.total_write_remote_mb == 0.0  # Should not be input for other groups

    def test_calculate_job_metrics_none_input_taskset_size(self):
        """Test that None input_taskset_size_per_event is handled correctly."""
        calculator = JobMetricsCalculator()

        taskset = TasksetInfo(
            taskset_id="Taskset1",
            group_name="group_1",
            input_taskset="InputTaskset",
            time_per_event=5.0,
            memory=1000,
            multicore=1,
            size_per_event=200,
            group_input_events=500,
            scram_arch=["el9_amd64_gcc11"],
            requires_gpu="forbidden",
            keep_output=False
        )

        # Test with None input_taskset_size_per_event
        job_metrics = calculator.calculate_job_metrics(
            tasksets=[taskset],
            batch_size=500,
            input_tasksets_for_other_groups=set(),
            input_taskset_size_per_event=None
        )

        assert job_metrics.total_cpu_time == 2500.0
        assert job_metrics.total_read_remote_mb == 0.0  # Should not have remote read


if __name__ == "__main__":
    pytest.main([__file__])
