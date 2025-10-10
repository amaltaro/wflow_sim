"""
Workflow Metrics Calculator

This module provides a comprehensive metrics calculation class for workflow simulation
results, supporting performance analysis and comparison of different workflow compositions.
"""

import json
import logging
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
    Calculator for workflow performance metrics.

    This class provides comprehensive metrics calculation for workflow simulations,
    including execution times, resource usage, throughput, and efficiency measures.
    """

    def __init__(self, workflow_data: Dict[str, Any]):
        """
        Initialize the metrics calculator with workflow data.

        Args:
            workflow_data: Dictionary containing workflow definition and execution results
        """
        self.workflow_data = workflow_data
        self.metrics: Optional[WorkflowMetrics] = None
        self.logger = logging.getLogger(__name__)

    def calculate_metrics(self) -> WorkflowMetrics:
        """
        Calculate comprehensive workflow metrics.

        Returns:
            WorkflowMetrics object containing all calculated metrics
        """
        self.logger.info("Starting workflow metrics calculation")

        # Extract basic workflow information
        workflow_id = self.workflow_data.get('workflow_id', 'unknown')
        composition_number = self.workflow_data.get('CompositionNumber', 0)

        # Calculate core metrics
        total_tasksets = self._calculate_total_tasksets()
        total_groups = self._calculate_total_groups()
        total_jobs = self._calculate_total_jobs()

        # Calculate timing metrics
        total_wall_time = self._calculate_total_wall_time()

        # Get turnaround time from simulation results if available
        simulation_results = self.workflow_data.get('simulation_results', {})
        total_turnaround_time = simulation_results.get('total_turnaround_time', self._calculate_total_execution_time())

        # Calculate group-level metrics
        group_metrics = self._calculate_group_metrics()

        # Calculate efficiency metrics
        resource_efficiency = self._calculate_resource_efficiency()
        throughput = self._calculate_throughput()
        success_rate = self._calculate_success_rate()

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

        self.logger.info("Workflow metrics calculation completed")
        return self.metrics

    def _calculate_total_tasksets(self) -> int:
        """Calculate total number of tasksets in the workflow."""
        taskset_count = 0
        for key in self.workflow_data.keys():
            if key.startswith('Taskset') and key[7:].isdigit():
                taskset_count += 1
        return taskset_count

    def _calculate_total_groups(self) -> int:
        """Calculate total number of groups in the workflow."""
        groups = set()
        for key, value in self.workflow_data.items():
            if key.startswith('Taskset') and isinstance(value, dict):
                group_name = value.get('GroupName')
                if group_name:
                    groups.add(group_name)
        return len(groups)

    def _calculate_total_jobs(self) -> int:
        """Calculate total number of jobs based on group events and scaling."""
        total_jobs = 0
        groups = self._get_groups_info()

        for group_id, group_info in groups.items():
            requested_events = self.workflow_data.get('RequestNumEvents', 0)
            group_input_events = group_info.get('GroupInputEvents', requested_events)

            # Calculate job scaling based on requested vs actual events
            if group_input_events > 0:
                jobs_for_group = max(1, requested_events // group_input_events)
                total_jobs += jobs_for_group

        return total_jobs

    def _calculate_total_execution_time(self) -> float:
        """Calculate total execution time across all tasksets."""
        # This would be calculated from actual execution results
        # For now, return a placeholder calculation
        total_time = 0.0
        for key, value in self.workflow_data.items():
            if key.startswith('Taskset') and isinstance(value, dict):
                time_per_event = value.get('TimePerEvent', 0)
                input_events = value.get('GroupInputEvents', 0)
                total_time += time_per_event * input_events
        return total_time

    def _calculate_total_wall_time(self) -> float:
        """Calculate total wall time (real elapsed time)."""
        # Use simulation results if available, otherwise calculate from workflow data
        simulation_results = self.workflow_data.get('simulation_results', {})
        if 'total_wall_time' in simulation_results:
            return simulation_results['total_wall_time']

        # Fallback to calculation from workflow data
        return self._calculate_total_execution_time()

    def _calculate_group_metrics(self) -> List[GroupMetrics]:
        """Calculate metrics for each group."""
        group_metrics = []
        groups = self._get_groups_info()

        for group_id, group_info in groups.items():
            # Calculate group-level metrics
            taskset_metrics = self._calculate_taskset_metrics_for_group(group_id)
            total_execution_time = sum(ts.execution_metrics.execution_time
                                    for ts in taskset_metrics)

            # Calculate total resource usage
            total_resource_usage = self._calculate_total_resource_usage(taskset_metrics)

            # Calculate job count for this group
            requested_events = self.workflow_data.get('RequestNumEvents', 0)
            group_input_events = group_info.get('GroupInputEvents', requested_events)
            job_count = max(1, requested_events // group_input_events) if group_input_events > 0 else 1

            group_metric = GroupMetrics(
                group_id=group_id,
                taskset_metrics=taskset_metrics,
                total_execution_time=total_execution_time,
                total_resource_usage=total_resource_usage,
                job_count=job_count
            )
            group_metrics.append(group_metric)

        return group_metrics

    def _calculate_taskset_metrics_for_group(self, group_id: str) -> List[TasksetMetrics]:
        """Calculate metrics for tasksets in a specific group."""
        taskset_metrics = []

        for key, value in self.workflow_data.items():
            if (key.startswith('Taskset') and isinstance(value, dict)
                and value.get('GroupName') == group_id):

                # Extract taskset information
                taskset_id = key
                time_per_event = value.get('TimePerEvent', 0)
                input_events = value.get('GroupInputEvents', 0)
                memory = value.get('Memory', 0)
                multicore = value.get('Multicore', 1)

                # Calculate execution metrics
                execution_time = time_per_event * input_events
                execution_metrics = ExecutionMetrics(
                    start_time=0.0,  # Would be actual start time
                    end_time=execution_time,
                    execution_time=execution_time
                )

                # Calculate resource usage
                resource_usage = ResourceUsage(
                    cpu_usage=multicore * 100.0,  # Percentage
                    memory_usage=memory,
                    storage_usage=value.get('SizePerEvent', 0) * input_events
                )

                taskset_metric = TasksetMetrics(
                    taskset_id=taskset_id,
                    execution_metrics=execution_metrics,
                    resource_usage=resource_usage,
                    input_events=input_events,
                    output_events=input_events  # Assuming 1:1 for now
                )
                taskset_metrics.append(taskset_metric)

        return taskset_metrics

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

    def _calculate_resource_efficiency(self) -> float:
        """Calculate overall resource efficiency."""
        # This is a simplified calculation - would be more complex in practice
        total_tasksets = self._calculate_total_tasksets()
        if total_tasksets == 0:
            return 0.0

        # Calculate efficiency based on resource utilization
        total_memory = 0
        total_cpu = 0

        for key, value in self.workflow_data.items():
            if key.startswith('Taskset') and isinstance(value, dict):
                total_memory += value.get('Memory', 0)
                total_cpu += value.get('Multicore', 1)

        # Simple efficiency metric (would be more sophisticated in practice)
        return min(1.0, (total_cpu * 100) / (total_memory + 1))

    def _calculate_throughput(self) -> float:
        """Calculate workflow throughput (events per second)."""
        total_events = self.workflow_data.get('RequestNumEvents', 0)
        total_time = self._calculate_total_execution_time()

        if total_time > 0:
            return total_events / total_time
        return 0.0

    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate."""
        # For now, assume 100% success - would be calculated from actual results
        return 1.0

    def _get_groups_info(self) -> Dict[str, Dict[str, Any]]:
        """Extract group information from workflow data."""
        groups = {}

        for key, value in self.workflow_data.items():
            if key.startswith('Taskset') and isinstance(value, dict):
                group_name = value.get('GroupName')
                if group_name:
                    if group_name not in groups:
                        groups[group_name] = {
                            'GroupInputEvents': value.get('GroupInputEvents', 0),
                            'tasksets': []
                        }
                    groups[group_name]['tasksets'].append(key)

        return groups

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
