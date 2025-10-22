"""
Workflow Runner

This module provides a high-level interface that combines workflow simulation
with metrics calculation, offering a complete workflow execution and analysis pipeline.
"""

import json
import logging
import argparse
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

        # Display wall time per event from metrics
        print(f"  Wall Time per Event: {metrics.wall_time_per_event:.6f}s/event")
        print(f"  CPU Time per Event: {metrics.cpu_time_per_event:.6f}s/event")
        print(f"  Network Transfer per Event: {metrics.network_transfer_per_event_mb:.6f} MB/event")

        # Metrics summary
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"  Resource Efficiency: {metrics.resource_efficiency:.2f}")
        print(f"  Event Throughput: {metrics.event_throughput:.6f} events/CPU-second")
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
        print(f"  Total CPU Time: {job_stats['total_cpu_time']:.2f}s")
        print(f"  Total Write Local: {job_stats['total_write_local_mb']:.2f} MB")
        print(f"  Total Write Remote: {job_stats['total_write_remote_mb']:.2f} MB")
        print(f"  Total Read Remote: {job_stats['total_read_remote_mb']:.2f} MB")
        print(f"  Total Network Transfer: {job_stats['total_network_transfer_mb']:.2f} MB")

    def write_complete_results(self, results: Dict[str, Any],
                              filepath: Union[str, Path]) -> None:
        """Write complete results (simulation + metrics) to a JSON file."""
        simulation = results['simulation_result']
        metrics = results['metrics']

        output_data = {
            'metrics': {
                'workflow_id': metrics.workflow_id,
                'composition_number': metrics.composition_number,
                'total_events': metrics.total_events,
                'total_tasksets': metrics.total_tasksets,
                'total_groups': metrics.total_groups,
                'total_jobs': metrics.total_jobs,
                'total_wall_time': metrics.total_wall_time,
                'total_turnaround_time': metrics.total_turnaround_time,
                'wall_time_per_event': metrics.wall_time_per_event,
                'cpu_time_per_event': metrics.cpu_time_per_event,
                'network_transfer_per_event_mb': metrics.network_transfer_per_event_mb,
                'resource_efficiency': metrics.resource_efficiency,
                'event_throughput': metrics.event_throughput,
                'success_rate': metrics.success_rate,
                'total_cpu_time': metrics.total_cpu_time,
                'total_write_local_mb': metrics.total_write_local_mb,
                'total_write_remote_mb': metrics.total_write_remote_mb,
                'total_read_remote_mb': metrics.total_read_remote_mb,
                'total_network_transfer_mb': metrics.total_network_transfer_mb
            },
            'simulation_result': {
                # Only include raw simulation data not available in metrics
                'success': simulation.success,
                'error_message': simulation.error_message,
                'groups': [
                    {
                        'group_id': group.group_id,
                        'job_count': group.job_count,
                        'input_events': group.input_events,
                        'total_execution_time': group.total_execution_time,
                        'exact_job_count': group.exact_job_count,
                        'dependencies': list(sorted(group.dependencies)),
                        'tasksets': [
                            {
                                'taskset_id': ts.taskset_id,
                                'group_name': ts.group_name,
                                'input_taskset': ts.input_taskset,
                                'time_per_event': ts.time_per_event,
                                'memory': ts.memory,
                                'multicore': ts.multicore,
                                'size_per_event': ts.size_per_event,
                                'group_input_events': ts.group_input_events,
                                'scram_arch': ts.scram_arch,
                                'requires_gpu': ts.requires_gpu,
                                'keep_output': ts.keep_output
                            }
                            for ts in group.tasksets
                        ]
                    }
                    for group in simulation.groups
                ],
                'jobs': [
                    {
                        'job_id': job.job_id,
                        'group_id': job.group_id,
                        'batch_size': job.batch_size,
                        'wallclock_time': job.wallclock_time,
                        'start_time': job.start_time,
                        'end_time': job.end_time,
                        'status': job.status,
                        'total_cpu_time': job.total_cpu_time,
                        'total_write_local_mb': job.total_write_local_mb,
                        'total_write_remote_mb': job.total_write_remote_mb,
                        'total_read_remote_mb': job.total_read_remote_mb,
                        'total_network_transfer_mb': job.total_network_transfer_mb
                    }
                    for job in simulation.jobs
                ],
            }
        }

        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)

        self.logger.info(f"Complete results written to {filepath}")


def _get_output_path(input_path: str) -> str:
    """
    Generate output path based on input path structure.

    Args:
        input_path: Path to input workflow file

    Returns:
        Output path in results/ directory with same structure (excluding templates/ prefix)
    """
    input_path_obj = Path(input_path)

    # Remove 'templates/' prefix if present
    if input_path_obj.parts[0] == 'templates':
        relative_path = input_path_obj.relative_to('templates')
    else:
        relative_path = input_path_obj

    # Create output path: results/ + relative path
    output_path = Path("results") / relative_path

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return str(output_path)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Workflow Runner - Complete workflow execution and analysis pipeline'
    )
    parser.add_argument(
        '--target-wallclock-time',
        type=int,
        default=43200,
        help='Target wallclock time in seconds (default: 43200 = 12 hours)'
    )
    parser.add_argument(
        '--max-job-slots',
        type=int,
        default=-1,
        help='Maximum number of job slots (-1 for infinite, default: -1)'
    )
    parser.add_argument(
        '--input-workflow-path',
        type=str,
        default='templates/3tasks_composition_001.json',
        help='Path to input workflow JSON file (default: templates/3tasks_composition_001.json)'
    )
    return parser.parse_args()


def main():
    """Main function with command line argument support."""
    args = parse_arguments()

    # Configure resources from command line arguments
    resource_config = ResourceConfig(
        target_wallclock_time=args.target_wallclock_time,
        max_job_slots=args.max_job_slots
    )

    # Create runner and execute workflow
    runner = WorkflowRunner(resource_config)
    results = runner.run_workflow(args.input_workflow_path)

    # Print complete summary
    runner.print_complete_summary(results)

    # Write results to file with same structure as input
    output_path = _get_output_path(args.input_workflow_path)
    runner.write_complete_results(results, output_path)


if __name__ == "__main__":
    main()
