"""
Job Metrics Calculator

This module provides job-level metrics calculation for workflow simulation,
handling individual job resource usage, I/O operations, and performance metrics.
"""

import logging
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class JobMetrics:
    """Job-level metrics for a single job execution."""
    total_cpu_time: float
    total_write_local_mb: float
    total_write_remote_mb: float
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

    def calculate_job_metrics(self, tasksets: List[Any], batch_size: int) -> JobMetrics:
        """
        Calculate job-level metrics for a given set of tasksets and batch size.
        
        Args:
            tasksets: List of TasksetInfo objects for the job
            batch_size: Number of events to process in the job
            
        Returns:
            JobMetrics object containing calculated metrics
        """
        total_cpu_time = 0.0
        total_write_local_mb = 0.0
        total_write_remote_mb = 0.0
        
        for taskset in tasksets:
            # Calculate CPU time: time_per_event * input_events * multicore
            cpu_time = taskset.time_per_event * batch_size * taskset.multicore
            total_cpu_time += cpu_time
            
            # Calculate write operations based on keep_output flag
            write_mb = taskset.size_per_event * batch_size
            if taskset.keep_output:
                total_write_remote_mb += write_mb
            else:
                total_write_local_mb += write_mb
        
        # For now, network transfer equals remote write
        total_network_transfer_mb = total_write_remote_mb
        
        return JobMetrics(
            total_cpu_time=total_cpu_time,
            total_write_local_mb=total_write_local_mb,
            total_write_remote_mb=total_write_remote_mb,
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
                'total_network_transfer_mb': 0.0,
                'average_cpu_time_per_job': 0.0,
                'average_write_local_mb_per_job': 0.0,
                'average_write_remote_mb_per_job': 0.0,
                'average_network_transfer_mb_per_job': 0.0
            }

        total_cpu_time = sum(job.total_cpu_time for job in jobs)
        total_write_local_mb = sum(job.total_write_local_mb for job in jobs)
        total_write_remote_mb = sum(job.total_write_remote_mb for job in jobs)
        total_network_transfer_mb = sum(job.total_network_transfer_mb for job in jobs)
        
        job_count = len(jobs)
        
        return {
            'total_cpu_time': total_cpu_time,
            'total_write_local_mb': total_write_local_mb,
            'total_write_remote_mb': total_write_remote_mb,
            'total_network_transfer_mb': total_network_transfer_mb,
            'average_cpu_time_per_job': total_cpu_time / job_count,
            'average_write_local_mb_per_job': total_write_local_mb / job_count,
            'average_write_remote_mb_per_job': total_write_remote_mb / job_count,
            'average_network_transfer_mb_per_job': total_network_transfer_mb / job_count
        }
