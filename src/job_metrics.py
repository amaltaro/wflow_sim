"""
Job Metrics Calculator

This module provides job-level metrics calculation for workflow simulation,
handling individual job resource usage, I/O operations, and performance metrics.
"""

import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass


@dataclass
class JobMetrics:
    """Job-level metrics for a single job execution."""
    total_cpu_time: float
    total_write_local_mb: float
    total_write_remote_mb: float
    total_read_remote_mb: float
    total_network_transfer_mb: float


class JobMetricsCalculator:
    """
    Calculator for job-level metrics from taskset information.

    This class handles the calculation of individual job metrics including
    CPU time, I/O operations, and network transfers based on taskset properties.
    """

    def __init__(self):
        """Initialize the job metrics calculator."""
        self.logger = logging.getLogger(__name__)

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

        total_cpu_time = 0.0
        total_write_local_mb = 0.0
        total_write_remote_mb = 0.0
        total_read_remote_mb = 0.0

        # Calculate remote read: if job has input taskset, read data from shared storage
        if input_taskset_size_per_event is not None:
            # SizePerEvent is in KB, convert to MB
            total_read_remote_mb = (input_taskset_size_per_event * batch_size) / 1024.0

        for taskset in tasksets:
            # Calculate CPU time: time_per_event * input_events * multicore
            cpu_time = taskset.time_per_event * batch_size * taskset.multicore
            total_cpu_time += cpu_time

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

        # For now, network transfer equals remote write
        total_network_transfer_mb = total_write_remote_mb

        return JobMetrics(
            total_cpu_time=total_cpu_time,
            total_write_local_mb=total_write_local_mb,
            total_write_remote_mb=total_write_remote_mb,
            total_read_remote_mb=total_read_remote_mb,
            total_network_transfer_mb=total_network_transfer_mb
        )

    def calculate_job_statistics(self, jobs: List[Any]) -> Dict[str, Any]:
        """
        Calculate aggregated statistics across all jobs.

        Args:
            jobs: List of JobInfo objects

        Returns:
            Dictionary containing aggregated job statistics
        """
        if not jobs:
            return {
                'total_cpu_time': 0.0,
                'total_write_local_mb': 0.0,
                'total_write_remote_mb': 0.0,
                'total_read_remote_mb': 0.0,
                'total_network_transfer_mb': 0.0,
                'average_cpu_time_per_job': 0.0,
                'average_write_local_mb_per_job': 0.0,
                'average_write_remote_mb_per_job': 0.0,
                'average_read_remote_mb_per_job': 0.0,
                'average_network_transfer_mb_per_job': 0.0
            }

        total_cpu_time = sum(job.total_cpu_time for job in jobs)
        total_write_local_mb = sum(job.total_write_local_mb for job in jobs)
        total_write_remote_mb = sum(job.total_write_remote_mb for job in jobs)
        total_read_remote_mb = sum(job.total_read_remote_mb for job in jobs)
        total_network_transfer_mb = sum(job.total_network_transfer_mb for job in jobs)

        job_count = len(jobs)

        return {
            'total_cpu_time': total_cpu_time,
            'total_write_local_mb': total_write_local_mb,
            'total_write_remote_mb': total_write_remote_mb,
            'total_read_remote_mb': total_read_remote_mb,
            'total_network_transfer_mb': total_network_transfer_mb,
            'average_cpu_time_per_job': total_cpu_time / job_count,
            'average_write_local_mb_per_job': total_write_local_mb / job_count,
            'average_write_remote_mb_per_job': total_write_remote_mb / job_count,
            'average_read_remote_mb_per_job': total_read_remote_mb / job_count,
            'average_network_transfer_mb_per_job': total_network_transfer_mb / job_count
        }
