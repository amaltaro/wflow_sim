"""
Job Metrics Calculator

This module provides job-level metrics calculation for workflow simulation,
handling individual job resource usage, I/O operations, and complete network transfer metrics.
"""

import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass


@dataclass
class JobMetrics:
    """Job-level metrics for a single job execution."""
    total_cpu_used_time: float  # Actual CPU time used from event processing
    total_cpu_allocated_time: float  # CPU time allocated for the whole job
    total_execution_time: float  # Total sequential execution time (tasksets execute one after another)
    total_write_local_mb: float
    total_write_remote_mb: float
    total_read_remote_mb: float
    total_read_local_mb: float
    total_network_transfer_mb: float  # remote_write + remote_read
    job_overhead_secs: float  # Total job overhead time in seconds
    job_overhead_cpu_time: float  # Allocated CPU time caused by job overhead (job_overhead_secs * max_multicore)


class JobMetricsCalculator:
    """
    Calculator for job-level metrics from taskset information.

    This class handles the calculation of individual job metrics including
    CPU time, I/O operations, and network transfers based on taskset properties.
    """

    def __init__(self, taskset_overhead_seconds: float = 60.0,
                 data_transfer_rate_mb_per_s: float = 100.0):
        """
        Initialize the job metrics calculator.

        Args:
            taskset_overhead_seconds: Overhead per taskset in seconds (default: 60.0)
            data_transfer_rate_mb_per_s: Data transfer rate in MB/s for overhead calculation (default: 100.0)
        """
        self.logger = logging.getLogger(__name__)
        self.taskset_overhead_seconds = taskset_overhead_seconds
        self.data_transfer_rate_mb_per_s = data_transfer_rate_mb_per_s

    def calculate_job_metrics(self, tasksets: List[Any], batch_size: int,
                             input_tasksets_for_other_groups: Optional[Set[str]] = None,
                             input_taskset_size_per_event: Optional[int] = None) -> JobMetrics:
        """
        Calculate job-level metrics for a given set of tasksets and batch size.

        Args:
            tasksets: List of TasksetInfo objects for the job
            batch_size: Number of events to process in the job
            input_tasksets_for_other_groups: Set of taskset IDs that are input tasksets for other groups
            input_taskset_size_per_event: SizePerEvent of the input taskset (in KB) if job has input taskset

        Returns:
            JobMetrics object containing calculated metrics
        """
        if input_tasksets_for_other_groups is None:
            input_tasksets_for_other_groups = set()

        total_cpu_used_time = 0.0
        total_write_local_mb = 0.0
        total_write_remote_mb = 0.0
        total_read_remote_mb = 0.0
        total_read_local_mb = 0.0

        # Calculate remote read: if job has input taskset, read data from shared storage
        if input_taskset_size_per_event is not None:
            # SizePerEvent is in KB, convert to MB
            total_read_remote_mb = (input_taskset_size_per_event * batch_size) / 1024.0

        # Find max multicore needed for the job (all tasksets share the same allocated resources)
        max_multicore = max(taskset.multicore for taskset in tasksets) if tasksets else 1

        # Calculate total sequential execution time (tasksets execute one after another)
        total_execution_time = sum(taskset.time_per_event * batch_size for taskset in tasksets)

        # Calculate CPU allocated time: total execution time × max multicore (allocated resources)
        total_cpu_allocated_time = total_execution_time * max_multicore

        taskset_size_map = {ts.taskset_id: ts.size_per_event for ts in tasksets}

        for taskset in tasksets:
            # Calculate CPU used time: actual CPU time used from event processing
            cpu_time = taskset.time_per_event * batch_size * taskset.multicore
            total_cpu_used_time += cpu_time
            # Calculate write operations (SizePerEvent is in KB, convert to MB)
            write_mb = (taskset.size_per_event * batch_size) / 1024.0

            # All data is written to local disk first
            total_write_local_mb += write_mb

            # Data is written remotely if:
            # 1. keep_output=True (explicitly marked for remote storage), OR
            # 2. This taskset is an input taskset for another group (needs to be available for other groups)
            is_remote_write = (taskset.keep_output or
                             taskset.taskset_id in input_tasksets_for_other_groups)

            if is_remote_write:
                total_write_remote_mb += write_mb

            # Calculate local read: if taskset has input_taskset within the same group
            if taskset.input_taskset and taskset.input_taskset in taskset_size_map:
                # This is a local read from another taskset in the same group
                input_size_per_event = taskset_size_map[taskset.input_taskset]
                local_read_mb = (input_size_per_event * batch_size) / 1024.0
                total_read_local_mb += local_read_mb

        # Network transfer includes both remote write and remote read
        total_network_transfer_mb = total_write_remote_mb + total_read_remote_mb

        # Calculate job overhead:
        # 1. Taskset overhead: 60 seconds per taskset
        taskset_overhead = len(tasksets) * self.taskset_overhead_seconds
        # 2. Network transfer overhead: 1 second per 100MB/s of data
        # Handle division by zero when data_transfer_rate is 0 (overhead disabled)
        if self.data_transfer_rate_mb_per_s > 0:
            network_transfer_overhead = total_network_transfer_mb / self.data_transfer_rate_mb_per_s
        else:
            network_transfer_overhead = 0.0
        # Total job overhead
        job_overhead_secs = taskset_overhead + network_transfer_overhead
        # Calculate allocated CPU time caused by job overhead
        job_overhead_cpu_time = job_overhead_secs * max_multicore
        # Add overhead to CPU metrics (overhead consumes CPU resources)
        total_cpu_used_time += (job_overhead_secs * 1)
        total_cpu_allocated_time += job_overhead_cpu_time
        # Add overhead to total execution time (overhead is part of job execution time)
        total_execution_time += job_overhead_secs

        return JobMetrics(
            total_cpu_used_time=total_cpu_used_time,
            total_cpu_allocated_time=total_cpu_allocated_time,
            total_execution_time=total_execution_time,
            total_write_local_mb=total_write_local_mb,
            total_write_remote_mb=total_write_remote_mb,
            total_read_remote_mb=total_read_remote_mb,
            total_read_local_mb=total_read_local_mb,
            total_network_transfer_mb=total_network_transfer_mb,
            job_overhead_secs=job_overhead_secs,
            job_overhead_cpu_time=job_overhead_cpu_time
        )
