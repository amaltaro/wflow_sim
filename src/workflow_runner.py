"""
Workflow Runner

This module provides a high-level interface that combines workflow simulation
with metrics calculation, offering a complete workflow execution and analysis pipeline.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import time

try:
    from .workflow_simulator import WorkflowSimulator, ResourceConfig, SimulationResult
    from .workflow_metrics import WorkflowMetricsCalculator, WorkflowMetrics
except ImportError:
    from workflow_simulator import WorkflowSimulator, ResourceConfig, SimulationResult
    from workflow_metrics import WorkflowMetricsCalculator, WorkflowMetrics


class WorkflowRunner:
    """
    High-level workflow execution and analysis pipeline.

    This class combines workflow simulation with metrics calculation to provide
    a complete workflow execution and analysis solution.
    """

    def __init__(self, resource_config: Optional[ResourceConfig] = None):
        """
        Initialize the workflow runner.

        Args:
            resource_config: Resource configuration for simulation
        """
        self.resource_config = resource_config or ResourceConfig()
        self.simulator = WorkflowSimulator(self.resource_config)
        self.logger = logging.getLogger(__name__)

    def run_workflow(self, workflow_filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Run a complete workflow simulation and analysis.

        Args:
            workflow_filepath: Path to JSON file containing workflow definition

        Returns:
            Dictionary containing simulation results and metrics
        """
        self.logger.info("Starting complete workflow execution and analysis")

        # Step 1: Run simulation
        simulation_result = self.simulator.simulate_workflow(workflow_filepath)

        if not simulation_result.success:
            self.logger.error(f"Workflow simulation failed: {simulation_result.error_message}")
            return {
                'simulation_result': simulation_result,
                'metrics': None,
                'success': False,
                'error_message': simulation_result.error_message
            }

        # Step 2: Calculate metrics directly from simulation result
        metrics_calculator = WorkflowMetricsCalculator()
        metrics = metrics_calculator.calculate_metrics(simulation_result)

        self.logger.info("Workflow execution and analysis completed successfully")

        return {
            'simulation_result': simulation_result,
            'metrics': metrics,
            'success': True,
            'error_message': None
        }


    def run_workflow_from_file(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Run workflow simulation and analysis from a JSON file.

        Args:
            filepath: Path to workflow JSON file

        Returns:
            Dictionary containing simulation results and metrics
        """
        return self.run_workflow(filepath)

    def print_complete_summary(self, results: Dict[str, Any]) -> None:
        """Print a complete summary of simulation and metrics."""
        if not results['success']:
            print(f"\n❌ Workflow execution failed: {results['error_message']}")
            return

        simulation = results['simulation_result']
        metrics = results['metrics']

        print("\n" + "="*80)
        print("COMPLETE WORKFLOW EXECUTION SUMMARY")
        print("="*80)

        # Simulation summary
        print(f"\n📊 SIMULATION RESULTS:")
        print(f"  Workflow ID: {simulation.workflow_id}")
        print(f"  Composition: {simulation.composition_number}")
        print(f"  Total Events: {simulation.total_events:,}")
        print(f"  Total Groups: {simulation.total_groups}")
        print(f"  Total Jobs: {simulation.total_jobs}")
        print(f"  Total Wall Time: {simulation.total_wall_time:.2f}s ({simulation.total_wall_time/3600:.2f}h)")
        print(f"  Total Turnaround Time: {simulation.total_turnaround_time:.2f}s ({simulation.total_turnaround_time/3600:.2f}h)")

        # Metrics summary
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"  Resource Efficiency: {metrics.resource_efficiency:.2f}")
        print(f"  Throughput: {metrics.throughput:.2f} events/second")
        print(f"  Success Rate: {metrics.success_rate:.2f}")
        print(f"  Total Execution Time: {simulation.total_turnaround_time:.2f}s")

        # Group details
        print(f"\n🏗️  GROUP BREAKDOWN:")
        for group in simulation.groups:
            print(f"  Group {group.group_id}:")
            print(f"    Jobs: {group.job_count}")
            print(f"    Events per Job: {group.input_events:,}")
            print(f"    Wall Time per Job: {self.resource_config.target_wallclock_time:.2f}s")
            print(f"    Total Execution Time: {group.total_execution_time:.2f}s")
            print(f"    Tasksets: {len(group.tasksets)}")

            for taskset in group.tasksets:
                print(f"      {taskset.taskset_id}: {taskset.time_per_event}s/event, "
                      f"{taskset.memory}MB, {taskset.multicore} cores")

        # Job statistics using consolidated metrics calculator
        print(f"\n⚡ JOB STATISTICS:")
        metrics_calculator = WorkflowMetricsCalculator()
        job_stats = metrics_calculator.calculate_job_statistics(simulation)
        print(f"  Average Job Wall Time: {job_stats['average_wall_time']:.2f}s")
        print(f"  Min Job Wall Time: {job_stats['min_wall_time']:.2f}s")
        print(f"  Max Job Wall Time: {job_stats['max_wall_time']:.2f}s")
        print(f"  Average Batch Size: {job_stats['average_batch_size']:.0f} events")
        print(f"  Min Batch Size: {job_stats['min_batch_size']} events")
        print(f"  Max Batch Size: {job_stats['max_batch_size']} events")

    def write_complete_results(self, results: Dict[str, Any],
                              filepath: Union[str, Path]) -> None:
        """Write complete results (simulation + metrics) to a JSON file."""
        output_data = {
            'simulation_result': {
                'workflow_id': results['simulation_result'].workflow_id,
                'composition_number': results['simulation_result'].composition_number,
                'total_events': results['simulation_result'].total_events,
                'total_groups': results['simulation_result'].total_groups,
                'total_jobs': results['simulation_result'].total_jobs,
                'total_wall_time': results['simulation_result'].total_wall_time,
                'total_turnaround_time': results['simulation_result'].total_turnaround_time,
                'success': results['simulation_result'].success,
                'error_message': results['simulation_result'].error_message,
                'groups': [
                    {
                        'group_id': group.group_id,
                        'job_count': group.job_count,
                        'input_events': group.input_events,
                        'total_execution_time': group.total_execution_time,
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
                    for group in results['simulation_result'].groups
                ],
                'jobs': [
                    {
                        'job_id': job.job_id,
                        'group_id': job.group_id,
                        'batch_size': job.batch_size,
                        'wallclock_time': job.wallclock_time,
                        'start_time': job.start_time,
                        'end_time': job.end_time,
                        'status': job.status
                    }
                    for job in results['simulation_result'].jobs
                ]
            },
            'metrics': {
                'workflow_id': results['metrics'].workflow_id,
                'composition_number': results['metrics'].composition_number,
                'total_tasksets': results['metrics'].total_tasksets,
                'total_groups': results['metrics'].total_groups,
                'total_jobs': results['metrics'].total_jobs,
                'total_wall_time': results['metrics'].total_wall_time,
                'total_turnaround_time': results['metrics'].total_turnaround_time,
                'resource_efficiency': results['metrics'].resource_efficiency,
                'throughput': results['metrics'].throughput,
                'success_rate': results['metrics'].success_rate,
                'timestamp': results['metrics'].timestamp
            },
            'success': results['success'],
            'error_message': results['error_message']
        }

        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)

        self.logger.info(f"Complete results written to {filepath}")


def main():
    """Example usage of the WorkflowRunner."""
    # Configure resources
    resource_config = ResourceConfig(
        target_wallclock_time=43200.0,  # 12 hours
        max_job_slots=-1  # Infinite slots
    )

    # Create runner and execute workflow
    runner = WorkflowRunner(resource_config)
    results = runner.run_workflow('templates/3tasks_composition_001.json')

    # Print complete summary
    runner.print_complete_summary(results)

    # Write results to file
    runner.write_complete_results(results, 'results/complete_workflow_results.json')


if __name__ == "__main__":
    main()
