"""
Workflow Metrics Calculator

This module provides a comprehensive metrics calculation class for workflow simulation
results, supporting performance analysis and comparison of different workflow compositions.
"""

import json
import logging
import math
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import time


@dataclass
class ResourceUsage:
    """Resource usage metrics for a single execution unit."""
    cpu_usage: float
    memory_usage: float
    storage_usage: float
    network_usage: float = 0.0


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
    total_tasksets: int
    total_groups: int
    total_jobs: int
    total_wall_time: float
    total_turnaround_time: float
    group_metrics: List[GroupMetrics]
    resource_efficiency: float
    throughput: float
    success_rate: float
    timestamp: float = time.time()


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
        self.logger = logging.getLogger(__name__)

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

        # Use simulation results directly
        total_tasksets = self._count_tasksets_from_simulation(simulation_result)
        total_groups = simulation_result.total_groups
        total_jobs = simulation_result.total_jobs
        total_wall_time = simulation_result.total_wall_time
        total_turnaround_time = simulation_result.total_turnaround_time

        # Calculate group-level metrics from simulation
        group_metrics = self._calculate_group_metrics_from_simulation(simulation_result)

        # Calculate efficiency metrics
        resource_efficiency = self._calculate_resource_efficiency_from_simulation(simulation_result)
        throughput = self._calculate_throughput_from_simulation(simulation_result)
        success_rate = self._calculate_success_rate_from_simulation(simulation_result)

        self.metrics = WorkflowMetrics(
            workflow_id=workflow_id,
            composition_number=composition_number,
            total_tasksets=total_tasksets,
            total_groups=total_groups,
            total_jobs=total_jobs,
            total_wall_time=total_wall_time,
            total_turnaround_time=total_turnaround_time,
            group_metrics=group_metrics,
            resource_efficiency=resource_efficiency,
            throughput=throughput,
            success_rate=success_rate
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

    def _calculate_group_metrics_from_simulation(self, simulation_result: 'SimulationResult') -> List[GroupMetrics]:
        """Calculate group metrics from simulation result."""
        group_metrics = []

        for group in simulation_result.groups:
            # Calculate taskset metrics for this group
            taskset_metrics = []
            for taskset in group.tasksets:
                execution_time = taskset.time_per_event * taskset.group_input_events
                execution_metrics = ExecutionMetrics(
                    start_time=0.0,
                    end_time=execution_time,
                    execution_time=execution_time
                )

                resource_usage = ResourceUsage(
                    cpu_usage=taskset.multicore * 100.0,
                    memory_usage=taskset.memory,
                    storage_usage=taskset.size_per_event * taskset.group_input_events
                )

                taskset_metric = TasksetMetrics(
                    taskset_id=taskset.taskset_id,
                    execution_metrics=execution_metrics,
                    resource_usage=resource_usage,
                    input_events=taskset.group_input_events,
                    output_events=taskset.group_input_events
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

    def _calculate_resource_efficiency_from_simulation(self, simulation_result: 'SimulationResult') -> float:
        """Calculate resource efficiency from simulation result."""
        if simulation_result.total_jobs == 0:
            return 0.0

        # Calculate total resource usage across all groups
        total_cpu = 0
        total_memory = 0

        for group in simulation_result.groups:
            for taskset in group.tasksets:
                total_cpu += taskset.multicore
                total_memory += taskset.memory

        # Simple efficiency metric based on resource utilization
        if total_memory > 0:
            return min(1.0, (total_cpu * 100) / total_memory)
        return 0.0

    def _calculate_throughput_from_simulation(self, simulation_result: 'SimulationResult') -> float:
        """Calculate throughput from simulation result."""
        if simulation_result.total_turnaround_time > 0:
            return simulation_result.total_events / simulation_result.total_turnaround_time
        return 0.0

    def _calculate_success_rate_from_simulation(self, simulation_result: 'SimulationResult') -> float:
        """Calculate success rate from simulation result."""
        if simulation_result.success:
            return 1.0

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
                'total_jobs': 0
            }

        job_wall_times = [job.wallclock_time for job in simulation_result.jobs]
        batch_sizes = [job.batch_size for job in simulation_result.jobs]

        return {
            'average_wall_time': sum(job_wall_times) / len(job_wall_times),
            'min_wall_time': min(job_wall_times),
            'max_wall_time': max(job_wall_times),
            'average_batch_size': sum(batch_sizes) / len(batch_sizes),
            'min_batch_size': min(batch_sizes),
            'max_batch_size': max(batch_sizes),
            'total_jobs': len(simulation_result.jobs)
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
        print(f"Resource Efficiency: {self.metrics.resource_efficiency:.2f}")
        print(f"Throughput: {self.metrics.throughput:.2f} events/second")
        print(f"Success Rate: {self.metrics.success_rate:.2f}")

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
            'resource_efficiency': self.metrics.resource_efficiency,
            'throughput': self.metrics.throughput,
            'success_rate': self.metrics.success_rate
        }
