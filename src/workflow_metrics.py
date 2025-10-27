"""
Workflow Metrics Calculator

This module provides a comprehensive metrics calculation class for workflow simulation
results, supporting performance analysis and comparison of different workflow compositions.
"""

import json
import logging
import math
import statistics
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import time

try:
    from .job_metrics import JobMetricsCalculator
except ImportError:
    from job_metrics import JobMetricsCalculator


@dataclass
class ResourceUsage:
    """Resource usage metrics for a single execution unit."""
    cpu_usage: float  # CPU cores used
    memory_usage: float  # Memory used in MB
    storage_usage: float  # Storage used in MB
    network_usage: float = 0.0  # Network usage in MB
    # Utilization ratios (0.0 to 1.0)
    cpu_utilization: float = 0.0  # Ratio of CPU used vs allocated
    memory_occupancy: float = 0.0  # Ratio of memory used vs allocated


@dataclass
class ExecutionMetrics:
    """Execution timing and performance metrics."""
    start_time: float
    end_time: float
    execution_time: float
    queue_time: float = 0.0
    setup_time: float = 0.0
    teardown_time: float = 0.0


@dataclass
class TasksetMetrics:
    """Metrics for individual tasksets."""
    taskset_id: str
    execution_metrics: ExecutionMetrics
    resource_usage: ResourceUsage
    input_events: int
    output_events: int
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class GroupMetrics:
    """Metrics for workflow groups."""
    group_id: str
    taskset_metrics: List[TasksetMetrics]
    total_execution_time: float
    total_resource_usage: ResourceUsage
    job_count: int
    success: bool = True


@dataclass
class WorkflowMetrics:
    """Comprehensive workflow execution metrics."""
    workflow_id: str
    composition_number: int
    total_events: int
    total_tasksets: int
    total_groups: int
    total_jobs: int
    total_wall_time: float
    total_turnaround_time: float
    wall_time_per_event: float
    cpu_time_per_event: float
    network_transfer_mb_per_event: float
    group_metrics: List[GroupMetrics]
    event_throughput: float
    success_rate: float
    # Aggregated job-level metrics
    total_cpu_time: float = 0.0
    total_write_local_mb: float = 0.0
    total_write_remote_mb: float = 0.0
    total_read_remote_mb: float = 0.0
    total_read_local_mb: float = 0.0
    total_network_transfer_mb: float = 0.0
    # Per-event metrics
    total_write_local_mb_per_event: float = 0.0
    total_write_remote_mb_per_event: float = 0.0
    total_read_remote_mb_per_event: float = 0.0
    total_read_local_mb_per_event: float = 0.0
    # Resource utilization metrics (aggregated from all jobs)
    resource_utilization: ResourceUsage = None


class WorkflowMetricsCalculator:
    """
    Calculator for workflow performance metrics from simulation results.

    This class provides comprehensive metrics calculation for workflow simulations,
    including execution times, resource usage, throughput, and efficiency measures.

    This is the authoritative module for all metrics calculations across the project.
    Only works with simulation results, not raw workflow composition data.
    """

    def __init__(self):
        """
        Initialize the metrics calculator.

        No workflow data needed since we only work with simulation results.
        """
        self.metrics: Optional[WorkflowMetrics] = None
        self.job_metrics_calculator = JobMetricsCalculator()
        self.logger = logging.getLogger(__name__)
        self._job_metrics_cache: Optional[Dict[str, Any]] = None

    def calculate_metrics(self, simulation_result: 'SimulationResult') -> WorkflowMetrics:
        """
        Calculate metrics directly from a SimulationResult object.

        This method provides a direct interface for calculating metrics from
        simulation results without requiring data conversion.

        Args:
            simulation_result: SimulationResult object from WorkflowSimulator

        Returns:
            WorkflowMetrics object containing all calculated metrics
        """
        self.logger.info("Starting workflow metrics calculation from simulation result")

        # Extract basic information from simulation result
        workflow_id = simulation_result.workflow_id
        composition_number = simulation_result.composition_number
        total_events = simulation_result.total_events

        # Use simulation results directly
        total_tasksets = self._count_tasksets_from_simulation(simulation_result)
        total_groups = simulation_result.total_groups
        total_jobs = simulation_result.total_jobs
        total_wall_time = simulation_result.total_wall_time
        total_turnaround_time = simulation_result.total_turnaround_time

        # Cache aggregated job metrics once (used by multiple methods below)
        self._job_metrics_cache = self._aggregate_job_metrics(simulation_result.jobs)

        # Calculate group-level metrics from simulation
        group_metrics = self._calculate_group_metrics_from_simulation(simulation_result)

        # Calculate efficiency metrics
        event_throughput = self._calculate_event_throughput_from_simulation(simulation_result)
        success_rate = self._calculate_success_rate_from_simulation(simulation_result)
        wall_time_per_event = self._calculate_wall_time_per_event_from_simulation(simulation_result)
        cpu_time_per_event = self._calculate_cpu_time_per_event_from_simulation(simulation_result)
        network_transfer_mb_per_event = self._calculate_network_transfer_per_event_from_simulation(simulation_result)
        total_write_local_mb_per_event = self._calculate_write_local_per_event_from_simulation(simulation_result)
        total_write_remote_mb_per_event = self._calculate_write_remote_per_event_from_simulation(simulation_result)
        total_read_remote_mb_per_event = self._calculate_read_remote_per_event_from_simulation(simulation_result)
        total_read_local_mb_per_event = self._calculate_read_local_per_event_from_simulation(simulation_result)

        # Calculate resource utilization metrics
        resource_utilization = self._calculate_resource_utilization_from_simulation(simulation_result)

        # Use cached job metrics
        job_metrics_stats = self._job_metrics_cache

        self.metrics = WorkflowMetrics(
            workflow_id=workflow_id,
            composition_number=composition_number,
            total_events=total_events,
            total_tasksets=total_tasksets,
            total_groups=total_groups,
            total_jobs=total_jobs,
            total_wall_time=total_wall_time,
            total_turnaround_time=total_turnaround_time,
            wall_time_per_event=wall_time_per_event,
            cpu_time_per_event=cpu_time_per_event,
            network_transfer_mb_per_event=network_transfer_mb_per_event,
            group_metrics=group_metrics,
            event_throughput=event_throughput,
            success_rate=success_rate,
            total_cpu_time=job_metrics_stats['total_cpu_time'],
            total_write_local_mb=job_metrics_stats['total_write_local_mb'],
            total_write_remote_mb=job_metrics_stats['total_write_remote_mb'],
            total_read_remote_mb=job_metrics_stats['total_read_remote_mb'],
            total_read_local_mb=job_metrics_stats['total_read_local_mb'],
            total_network_transfer_mb=job_metrics_stats['total_network_transfer_mb'],
            total_write_local_mb_per_event=total_write_local_mb_per_event,
            total_write_remote_mb_per_event=total_write_remote_mb_per_event,
            total_read_remote_mb_per_event=total_read_remote_mb_per_event,
            total_read_local_mb_per_event=total_read_local_mb_per_event,
            resource_utilization=resource_utilization
        )

        self.logger.info("Workflow metrics calculation from simulation completed")
        return self.metrics

    def _calculate_total_resource_usage(self, taskset_metrics: List[TasksetMetrics]) -> ResourceUsage:
        """Calculate total resource usage across tasksets."""
        total_cpu = sum(ts.resource_usage.cpu_usage for ts in taskset_metrics)
        total_memory = sum(ts.resource_usage.memory_usage for ts in taskset_metrics)
        total_storage = sum(ts.resource_usage.storage_usage for ts in taskset_metrics)
        total_network = sum(ts.resource_usage.network_usage for ts in taskset_metrics)

        return ResourceUsage(
            cpu_usage=total_cpu,
            memory_usage=total_memory,
            storage_usage=total_storage,
            network_usage=total_network
        )


    def _count_tasksets_from_simulation(self, simulation_result: 'SimulationResult') -> int:
        """Count total tasksets from simulation result."""
        total_tasksets = 0
        for group in simulation_result.groups:
            total_tasksets += len(group.tasksets)
        return total_tasksets

    def _aggregate_job_metrics(self, jobs: List[Any]) -> Dict[str, float]:
        """
        Aggregate job metrics across all jobs (workflow-level calculation).

        This is a workflow-level operation since it aggregates across multiple jobs.
        Returns aggregated totals and statistics for both job metrics and execution metadata.

        Args:
            jobs: List of JobInfo objects

        Returns:
            Dictionary containing aggregated metrics and statistics
        """
        if not jobs:
            return {
                'total_cpu_time': 0.0,
                'total_write_local_mb': 0.0,
                'total_write_remote_mb': 0.0,
                'total_read_remote_mb': 0.0,
                'total_read_local_mb': 0.0,
                'total_network_transfer_mb': 0.0,
                'total_jobs': 0
            }

        return {
            'total_cpu_time': sum(job.total_cpu_time for job in jobs),
            'total_write_local_mb': sum(job.total_write_local_mb for job in jobs),
            'total_write_remote_mb': sum(job.total_write_remote_mb for job in jobs),
            'total_read_remote_mb': sum(job.total_read_remote_mb for job in jobs),
            'total_read_local_mb': sum(job.total_read_local_mb for job in jobs),
            'total_network_transfer_mb': sum(job.total_network_transfer_mb for job in jobs),
            'total_jobs': len(jobs),
            '_wall_times': [job.wallclock_time for job in jobs],
            '_batch_sizes': [job.batch_size for job in jobs]
        }

    def _calculate_group_metrics_from_simulation(self, simulation_result: 'SimulationResult') -> List[GroupMetrics]:
        """Calculate group metrics from simulation result."""
        group_metrics = []

        for group in simulation_result.groups:
            # Calculate taskset metrics for this group
            taskset_metrics = []
            for taskset in group.tasksets:
                # Use group input_events (batch size) for calculations
                batch_size = group.input_events
                execution_time = taskset.time_per_event * batch_size
                execution_metrics = ExecutionMetrics(
                    start_time=0.0,
                    end_time=execution_time,
                    execution_time=execution_time
                )

                resource_usage = ResourceUsage(
                    cpu_usage=taskset.multicore * 100.0,
                    memory_usage=taskset.memory,
                    storage_usage=taskset.size_per_event * batch_size
                )

                taskset_metric = TasksetMetrics(
                    taskset_id=taskset.taskset_id,
                    execution_metrics=execution_metrics,
                    resource_usage=resource_usage,
                    input_events=batch_size,
                    output_events=batch_size
                )
                taskset_metrics.append(taskset_metric)

            # Calculate total execution time and resource usage for group
            total_execution_time = sum(ts.execution_metrics.execution_time for ts in taskset_metrics)
            total_resource_usage = self._calculate_total_resource_usage(taskset_metrics)

            group_metric = GroupMetrics(
                group_id=group.group_id,
                taskset_metrics=taskset_metrics,
                total_execution_time=total_execution_time,
                total_resource_usage=total_resource_usage,
                job_count=group.job_count
            )
            group_metrics.append(group_metric)

        return group_metrics


    def _calculate_event_throughput_from_simulation(self, simulation_result: 'SimulationResult') -> float:
        """Calculate event throughput from simulation result."""
        if simulation_result.total_events > 0:
            # Use cached job metrics
            total_cpu_time = self._job_metrics_cache['total_cpu_time']
            if total_cpu_time > 0:
                return simulation_result.total_events / total_cpu_time
        return 0.0

    def _calculate_success_rate_from_simulation(self, simulation_result: 'SimulationResult') -> float:
        """Calculate success rate from simulation result."""
        if simulation_result.success:
            return 1.0

    def _calculate_wall_time_per_event_from_simulation(self, simulation_result: 'SimulationResult') -> float:
        """Calculate wall time per event from simulation result."""
        if simulation_result.total_events > 0:
            return simulation_result.total_wall_time / simulation_result.total_events
        return 0.0

    def _calculate_cpu_time_per_event_from_simulation(self, simulation_result: 'SimulationResult') -> float:
        """Calculate CPU time per event from simulation result."""
        if simulation_result.total_events > 0:
            # Use cached job metrics
            total_cpu_time = self._job_metrics_cache['total_cpu_time']
            return total_cpu_time / simulation_result.total_events
        return 0.0

    def _calculate_network_transfer_per_event_from_simulation(self, simulation_result: 'SimulationResult') -> float:
        """Calculate network transfer per event from simulation result."""
        if simulation_result.total_events > 0:
            # Use cached job metrics
            total_network_transfer_mb = self._job_metrics_cache['total_network_transfer_mb']
            return total_network_transfer_mb / simulation_result.total_events
        return 0.0

    def _calculate_write_local_per_event_from_simulation(self, simulation_result: 'SimulationResult') -> float:
        """Calculate write local per event from simulation result."""
        if simulation_result.total_events > 0:
            # Use cached job metrics
            total_write_local_mb = self._job_metrics_cache['total_write_local_mb']
            return total_write_local_mb / simulation_result.total_events
        return 0.0

    def _calculate_write_remote_per_event_from_simulation(self, simulation_result: 'SimulationResult') -> float:
        """Calculate write remote per event from simulation result."""
        if simulation_result.total_events > 0:
            # Use cached job metrics
            total_write_remote_mb = self._job_metrics_cache['total_write_remote_mb']
            return total_write_remote_mb / simulation_result.total_events
        return 0.0

    def _calculate_read_remote_per_event_from_simulation(self, simulation_result: 'SimulationResult') -> float:
        """Calculate read remote per event from simulation result."""
        if simulation_result.total_events > 0:
            # Use cached job metrics
            total_read_remote_mb = self._job_metrics_cache['total_read_remote_mb']
            return total_read_remote_mb / simulation_result.total_events
        return 0.0

    def _calculate_read_local_per_event_from_simulation(self, simulation_result: 'SimulationResult') -> float:
        """Calculate read local per event from simulation result."""
        if simulation_result.total_events > 0:
            # Use cached job metrics
            total_read_local_mb = self._job_metrics_cache['total_read_local_mb']
            return total_read_local_mb / simulation_result.total_events
        return 0.0

    def _calculate_resource_utilization_from_simulation(self, simulation_result: 'SimulationResult') -> ResourceUsage:
        """Calculate resource utilization from simulation result."""
        if simulation_result.total_wall_time <= 0:
            return ResourceUsage(cpu_usage=0.0, memory_usage=0.0, storage_usage=0.0,
                               cpu_utilization=0.0, memory_occupancy=0.0)

        # Calculate utilization per group and then average across groups
        group_cpu_utilizations = []
        group_memory_occupancies = []
        total_cpu_cores_used = 0
        total_memory_used = 0

        # Use cached job metrics for total usage (cache available when called through calculate_metrics)
        # If cache not available, populate it by calling _aggregate_job_metrics
        if self._job_metrics_cache is None:
            self._job_metrics_cache = self._aggregate_job_metrics(simulation_result.jobs)

        total_cpu_time_used = self._job_metrics_cache['total_cpu_time']
        network_usage = self._job_metrics_cache['total_network_transfer_mb']

        for group in simulation_result.groups:
            # find allocated resources per job/group
            max_group_cores = max(taskset.multicore for taskset in group.tasksets)
            max_group_memory = max(taskset.memory for taskset in group.tasksets)

            # calculate actual CPU utilization ratio
            group_time_per_event = sum(taskset.time_per_event for taskset in group.tasksets)
            group_cpu_utilization = sum(group.input_events * taskset.time_per_event * taskset.multicore for taskset in group.tasksets)
            group_cpu_utilization /= group.input_events * group_time_per_event * max_group_cores
            # calculate actual Memory utilization ratio
            group_memory_utilization = sum(group.input_events * taskset.time_per_event * taskset.memory for taskset in group.tasksets)
            group_memory_utilization /= group.input_events * group_time_per_event * max_group_memory

            # append to overall workflow calculation
            group_cpu_utilizations.append(group_cpu_utilization)
            group_memory_occupancies.append(group_memory_utilization)

            # Find jobs for this group
            for job in simulation_result.jobs:
                if job.group_id == group.group_id:
                    total_cpu_cores_used += max_group_cores
                    total_memory_used += max_group_memory

        # Calculate average utilization across all groups
        cpu_utilization = statistics.mean(group_cpu_utilizations) if group_cpu_utilizations else 0.0
        memory_occupancy = statistics.mean(group_memory_occupancies) if group_memory_occupancies else 0.0

        return ResourceUsage(
            cpu_usage=total_cpu_cores_used,
            memory_usage=total_memory_used,
            storage_usage=0.0,  # Could be calculated from I/O metrics if needed
            network_usage=network_usage,
            cpu_utilization=cpu_utilization,
            memory_occupancy=memory_occupancy
        )

    def calculate_job_statistics(self, simulation_result: 'SimulationResult') -> Dict[str, Any]:
        """
        Calculate comprehensive job statistics from simulation results.

        Args:
            simulation_result: SimulationResult object from WorkflowSimulator

        Returns:
            Dictionary containing job statistics
        """
        if not simulation_result.jobs:
            return {
                'average_wall_time': 0.0,
                'min_wall_time': 0.0,
                'max_wall_time': 0.0,
                'average_batch_size': 0.0,
                'min_batch_size': 0,
                'max_batch_size': 0,
                'total_jobs': 0,
                'total_cpu_time': 0.0,
                'total_write_local_mb': 0.0,
                'total_write_remote_mb': 0.0,
                'total_read_local_mb': 0.0,
                'total_read_remote_mb': 0.0,
                'total_network_transfer_mb': 0.0
            }

        # Aggregate metrics across all jobs (workflow-level aggregation)
        job_metrics = self._aggregate_job_metrics(simulation_result.jobs)

        # Extract wall times and batch sizes for statistics
        job_wall_times = job_metrics['_wall_times']
        batch_sizes = job_metrics['_batch_sizes']

        return {
            'average_wall_time': sum(job_wall_times) / len(job_wall_times) if job_wall_times else 0.0,
            'min_wall_time': min(job_wall_times) if job_wall_times else 0.0,
            'max_wall_time': max(job_wall_times) if job_wall_times else 0.0,
            'average_batch_size': sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0.0,
            'min_batch_size': min(batch_sizes) if batch_sizes else 0,
            'max_batch_size': max(batch_sizes) if batch_sizes else 0,
            'total_jobs': job_metrics['total_jobs'],
            'total_cpu_time': job_metrics['total_cpu_time'],
            'total_write_local_mb': job_metrics['total_write_local_mb'],
            'total_write_remote_mb': job_metrics['total_write_remote_mb'],
            'total_read_local_mb': job_metrics['total_read_local_mb'],
            'total_read_remote_mb': job_metrics['total_read_remote_mb'],
            'total_network_transfer_mb': job_metrics['total_network_transfer_mb']
        }

    def calculate_group_statistics(self, simulation_result: 'SimulationResult') -> Dict[str, Any]:
        """
        Calculate comprehensive group statistics from simulation results.

        Args:
            simulation_result: SimulationResult object from WorkflowSimulator

        Returns:
            Dictionary containing group statistics
        """
        group_stats = {}

        for group in simulation_result.groups:
            group_stats[group.group_id] = {
                'job_count': group.job_count,
                'input_events': group.input_events,
                'total_execution_time': group.total_execution_time,
                'taskset_count': len(group.tasksets),
                'tasksets': [
                    {
                        'taskset_id': ts.taskset_id,
                        'time_per_event': ts.time_per_event,
                        'memory': ts.memory,
                        'multicore': ts.multicore,
                        'size_per_event': ts.size_per_event
                    }
                    for ts in group.tasksets
                ]
            }

        return group_stats

    def print_metrics(self, detailed: bool = False) -> None:
        """
        Print calculated metrics in a readable format.

        Args:
            detailed: If True, print detailed metrics for each group and taskset
        """
        if not self.metrics:
            self.calculate_metrics()

        print("\n" + "="*60)
        print("WORKFLOW EXECUTION METRICS")
        print("="*60)

        print(f"Workflow ID: {self.metrics.workflow_id}")
        print(f"Composition Number: {self.metrics.composition_number}")
        print(f"Total Tasksets: {self.metrics.total_tasksets}")
        print(f"Total Groups: {self.metrics.total_groups}")
        print(f"Total Jobs: {self.metrics.total_jobs}")
        print(f"Total Turnaround Time: {self.metrics.total_turnaround_time:.2f} seconds")
        print(f"Total Wall Time: {self.metrics.total_wall_time:.2f} seconds")
        print(f"Wall Time per Event: {self.metrics.wall_time_per_event:.6f} seconds")
        print(f"CPU Time per Event: {self.metrics.cpu_time_per_event:.6f} seconds")
        print(f"Network Transfer per Event: {self.metrics.network_transfer_mb_per_event:.6f} MB")
        print(f"Event Throughput: {self.metrics.event_throughput:.6f} events/CPU-second")
        print(f"Success Rate: {self.metrics.success_rate:.2f}")
        # Print total resource usage metrics
        if self.metrics.resource_utilization:
            print(f"\n" + "-"*40)
            print("TOTAL RESOURCE USAGE")
            print("-"*40)
            print(f"Total CPU Cores Used: {self.metrics.resource_utilization.cpu_usage:.0f} cores")
            print(f"Total Memory Used: {self.metrics.resource_utilization.memory_usage:.0f} MB")
            print(f"CPU Cores per Event: {self.metrics.resource_utilization.cpu_usage / self.metrics.total_events:.6f} cores/event")
            print(f"Memory per Event: {self.metrics.resource_utilization.memory_usage / self.metrics.total_events:.6f} MB/event")
            print(f"CPU Utilization: {self.metrics.resource_utilization.cpu_utilization:.2%}")
            print(f"Memory Occupancy: {self.metrics.resource_utilization.memory_occupancy:.2%}")
        # Print aggregated job-level metrics
        print(f"\n" + "-"*40)
        print("AGGREGATED JOB METRICS")
        print("-"*40)
        print(f"Total CPU Time: {self.metrics.total_cpu_time:.2f} seconds")
        print(f"Total Write Local: {self.metrics.total_write_local_mb:.2f} MB")
        print(f"Total Write Remote: {self.metrics.total_write_remote_mb:.2f} MB")
        print(f"Total Read Local: {self.metrics.total_read_local_mb:.2f} MB")
        print(f"Total Read Remote: {self.metrics.total_read_remote_mb:.2f} MB")
        print(f"Total Network Transfer: {self.metrics.total_network_transfer_mb:.2f} MB")

        if detailed:
            print("\n" + "-"*40)
            print("GROUP DETAILS")
            print("-"*40)

            for group in self.metrics.group_metrics:
                print(f"\nGroup: {group.group_id}")
                print(f"  Jobs: {group.job_count}")
                print(f"  Execution Time: {group.total_execution_time:.2f}s")
                print(f"  CPU Usage: {group.total_resource_usage.cpu_usage:.2f}%")
                print(f"  Memory Usage: {group.total_resource_usage.memory_usage:.2f} MB")
                print(f"  Storage Usage: {group.total_resource_usage.storage_usage:.2f} MB")

                for taskset in group.taskset_metrics:
                    print(f"    {taskset.taskset_id}: {taskset.execution_metrics.execution_time:.2f}s")

    def write_metrics_to_file(self, filepath: Union[str, Path]) -> None:
        """
        Write metrics to a JSON file.

        Args:
            filepath: Path to output file
        """
        if not self.metrics:
            self.calculate_metrics()

        # Convert metrics to dictionary for JSON serialization
        metrics_dict = asdict(self.metrics)

        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

        self.logger.info(f"Metrics written to {filepath}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of key metrics.

        Returns:
            Dictionary containing key metrics summary
        """
        if not self.metrics:
            self.calculate_metrics()

        return {
            'workflow_id': self.metrics.workflow_id,
            'total_tasksets': self.metrics.total_tasksets,
            'total_groups': self.metrics.total_groups,
            'total_jobs': self.metrics.total_jobs,
            'total_wall_time': self.metrics.total_wall_time,
            'total_turnaround_time': self.metrics.total_turnaround_time,
            'wall_time_per_event': self.metrics.wall_time_per_event,
            'cpu_time_per_event': self.metrics.cpu_time_per_event,
            'network_transfer_mb_per_event': self.metrics.network_transfer_mb_per_event,
            'event_throughput': self.metrics.event_throughput,
            'success_rate': self.metrics.success_rate,
            'total_cpu_time': self.metrics.total_cpu_time,
            'total_write_local_mb': self.metrics.total_write_local_mb,
            'total_write_remote_mb': self.metrics.total_write_remote_mb,
            'total_read_remote_mb': self.metrics.total_read_remote_mb,
            'total_read_local_mb': self.metrics.total_read_local_mb,
            'total_network_transfer_mb': self.metrics.total_network_transfer_mb,
            'total_write_local_mb_per_event': self.metrics.total_write_local_mb_per_event,
            'total_write_remote_mb_per_event': self.metrics.total_write_remote_mb_per_event,
            'total_read_remote_mb_per_event': self.metrics.total_read_remote_mb_per_event,
            'total_read_local_mb_per_event': self.metrics.total_read_local_mb_per_event,
            'cpu_utilization': self.metrics.resource_utilization.cpu_utilization if self.metrics.resource_utilization else 0.0,
            'memory_occupancy': self.metrics.resource_utilization.memory_occupancy if self.metrics.resource_utilization else 0.0,
            'total_cpu_cores_used': self.metrics.resource_utilization.cpu_usage if self.metrics.resource_utilization else 0.0,
            'total_memory_used_mb': self.metrics.resource_utilization.memory_usage if self.metrics.resource_utilization else 0.0,
            'cpu_cores_per_event': (self.metrics.resource_utilization.cpu_usage / self.metrics.total_events) if self.metrics.resource_utilization and self.metrics.total_events > 0 else 0.0,
            'memory_mb_per_event': (self.metrics.resource_utilization.memory_usage / self.metrics.total_events) if self.metrics.resource_utilization and self.metrics.total_events > 0 else 0.0
        }
