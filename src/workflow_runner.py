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

# Limit job dumps per group in JSON output to reduce disk usage
MAX_JOBS_PER_GROUP_IN_OUTPUT = 10


def _jobs_for_output(all_jobs: List[Any], max_per_group: int = MAX_JOBS_PER_GROUP_IN_OUTPUT) -> List[Any]:
    """Return at most max_per_group jobs per group_id, preserving original order."""
    counts: Dict[str, int] = {}
    out: List[Any] = []
    for job in all_jobs:
        gid = getattr(job, 'group_id', None)
        if gid is not None:
            n = counts.get(gid, 0)
            if n >= max_per_group:
                continue
            counts[gid] = n + 1
        out.append(job)
    return out


class WorkflowRunner:
    """
    High-level workflow execution and analysis pipeline.

    This class combines workflow simulation with metrics calculation to provide
    a complete workflow execution and analysis solution.
    """

    def __init__(self, resource_config: Optional[ResourceConfig] = None,
                 *,
                 failure_rate: int,
                 data_transfer_rate_mb_per_s: float):
        """
        Initialize the workflow runner.

        Args:
            resource_config: Resource configuration for simulation
            failure_rate: Job failure rate as percentage (0-99); required, no default
                (supply from CLI parser or caller).
            data_transfer_rate_mb_per_s: Network data transfer rate in MB/s for
                overhead; required, no default (supply from CLI parser or caller).
        """
        self.resource_config = resource_config or ResourceConfig()
        self.failure_rate = failure_rate
        self.simulator = WorkflowSimulator(
            self.resource_config,
            failure_rate=failure_rate,
            data_transfer_rate_mb_per_s=data_transfer_rate_mb_per_s
        )
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
        print(f"  Overhead Enabled: {simulation.overhead_enabled}")
        print(f"  Failure Rate: {simulation.failure_rate:.1f}%")
        print(f"  Total Events: {simulation.total_events:,}")
        print(f"  Total Groups: {simulation.total_groups}")
        total_logical_jobs = simulation.total_jobs - simulation.total_job_retries
        print(f"  Total Jobs: {simulation.total_jobs} (Logical: {total_logical_jobs}, Retries: {simulation.total_job_retries})")
        print(f"  Total Wall Time: {simulation.total_wall_time:.2f}s ({simulation.total_wall_time/3600:.2f}h)")
        print(f"  Total Turnaround Time: {simulation.total_turnaround_time:.2f}s ({simulation.total_turnaround_time/3600:.2f}h)")

        # Display wall time per event from metrics
        print(f"  Wall Time per Event: {metrics.wall_time_per_event:.6f}s/event")
        print(f"  CPU Time per Event: {metrics.cpu_time_per_event:.6f}s/event")
        print(f"  Network Transfer per Event: {metrics.network_transfer_mb_per_event:.6f} MB/event")

        # Metrics summary
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"  Event Throughput: {metrics.event_throughput:.6f} events/CPU-second")
        print(f"  Success Rate: {metrics.success_rate:.2f}")
        print(f"  Total Execution Time: {simulation.total_turnaround_time:.2f}s")

        # Resource usage summary
        if metrics.resource_utilization:
            print(f"\n💻 RESOURCE USAGE:")
            print(f"  Total CPU Cores Used: {metrics.resource_utilization.cpu_usage:.0f} cores")
            print(f"  Total Memory Used: {metrics.resource_utilization.memory_usage:.0f} MB")
            print(f"  CPU Cores per Event: {metrics.resource_utilization.cpu_usage / metrics.total_events:.6f} cores/event")
            print(f"  Memory per Event: {metrics.resource_utilization.memory_usage / metrics.total_events:.6f} MB/event")
            print(f"  CPU Utilization: {metrics.resource_utilization.cpu_utilization:.2%}")
            print(f"  Memory Occupancy: {metrics.resource_utilization.memory_occupancy:.2%}")

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
        print(f"  Total CPU Used Time: {job_stats['total_cpu_used_time']:.2f}s")
        print(f"  Total CPU Allocated Time: {job_stats['total_cpu_allocated_time']:.2f}s")
        print(f"  Total Write Local: {job_stats['total_write_local_mb']:.2f} MB")
        print(f"  Total Write Remote: {job_stats['total_write_remote_mb']:.2f} MB")
        print(f"  Total Read Local: {job_stats['total_read_local_mb']:.2f} MB")
        print(f"  Total Read Remote: {job_stats['total_read_remote_mb']:.2f} MB")
        print(f"  Total Network Transfer: {job_stats['total_network_transfer_mb']:.2f} MB")

    def write_complete_results(self, results: Dict[str, Any],
                              filepath: Union[str, Path]) -> None:
        """Write complete results (simulation + metrics) to a JSON file."""
        simulation = results['simulation_result']
        metrics = results['metrics']

        # Handle case where simulation failed and metrics is None
        if metrics is None:
            self.logger.warning(f"Simulation failed, writing partial results without metrics to {filepath}")
            output_data = {
                'simulation_result': {
                    'success': simulation.success,
                    'error_message': simulation.error_message,
                    'overhead_enabled': simulation.overhead_enabled,
                    'failure_rate': simulation.failure_rate,
                    'total_job_retries': simulation.total_job_retries,
                    'jobs_per_group_limit': MAX_JOBS_PER_GROUP_IN_OUTPUT,
                    'groups': [],
                    'jobs': []
                }
            }
            with open(filepath, 'w') as f:
                json.dump(output_data, f, indent=2)
            self.logger.info(f"Partial results written to {filepath}")
            return

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
                'network_transfer_mb_per_event': metrics.network_transfer_mb_per_event,
                'event_throughput': metrics.event_throughput,
                'success_rate': metrics.success_rate,
                'total_cpu_used_time': metrics.total_cpu_used_time,
                'total_cpu_allocated_time': metrics.total_cpu_allocated_time,
                'total_write_local_mb': metrics.total_write_local_mb,
                'total_write_remote_mb': metrics.total_write_remote_mb,
                'total_read_remote_mb': metrics.total_read_remote_mb,
                'total_read_local_mb': metrics.total_read_local_mb,
                'total_network_transfer_mb': metrics.total_network_transfer_mb,
                'total_job_overhead_secs': metrics.total_job_overhead_secs,
                'total_job_overhead_cpu_time': metrics.total_job_overhead_cpu_time,
                'total_write_local_mb_per_event': metrics.total_write_local_mb_per_event,
                'total_write_remote_mb_per_event': metrics.total_write_remote_mb_per_event,
                'total_read_remote_mb_per_event': metrics.total_read_remote_mb_per_event,
                'total_read_local_mb_per_event': metrics.total_read_local_mb_per_event,
                'cpu_utilization': metrics.resource_utilization.cpu_utilization if metrics.resource_utilization else 0.0,
                'memory_occupancy': metrics.resource_utilization.memory_occupancy if metrics.resource_utilization else 0.0,
                'total_cpu_cores_used': metrics.resource_utilization.cpu_usage if metrics.resource_utilization else 0.0,
                'total_memory_used_mb': metrics.resource_utilization.memory_usage if metrics.resource_utilization else 0.0,
                'cpu_cores_per_event': (metrics.resource_utilization.cpu_usage / metrics.total_events) if metrics.resource_utilization and metrics.total_events > 0 else 0.0,
                'memory_mb_per_event': (metrics.resource_utilization.memory_usage / metrics.total_events) if metrics.resource_utilization and metrics.total_events > 0 else 0.0
            },
            'simulation_result': {
                # Only include raw simulation data not available in metrics
                'success': simulation.success,
                'error_message': simulation.error_message,
                'overhead_enabled': simulation.overhead_enabled,
                'failure_rate': simulation.failure_rate,
                'total_job_retries': simulation.total_job_retries,
                'jobs_per_group_limit': MAX_JOBS_PER_GROUP_IN_OUTPUT,
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
                        'total_cpu_used_time': job.total_cpu_used_time,
                        'total_cpu_allocated_time': job.total_cpu_allocated_time,
                        'total_write_local_mb': job.total_write_local_mb,
                        'total_write_remote_mb': job.total_write_remote_mb,
                        'total_read_local_mb': job.total_read_local_mb,
                        'total_read_remote_mb': job.total_read_remote_mb,
                        'total_network_transfer_mb': job.total_network_transfer_mb,
                        'total_execution_time': job.total_execution_time,
                        'job_overhead_secs': job.job_overhead_secs,
                        'job_overhead_cpu_time': job.job_overhead_cpu_time,
                        'retry_count': job.retry_count,
                        'original_job_id': job.original_job_id
                    }
                    for job in _jobs_for_output(simulation.jobs)
                ],
            }
        }

        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)

        self.logger.info(f"Complete results written to {filepath}")


def _data_rate_dir_from_mbps(data_transfer_rate_mb_per_s: float) -> str:
    """
    Map network data transfer rate (MB/s) to directory name (MBps/GBps = bytes per second).

    Args:
        data_transfer_rate_mb_per_s: Rate in MB/s (e.g. 10, 100, 1000, 10000).

    Returns:
        Directory name: 10MBps, 100MBps, 1GBps, 10GBps, or {int}MBps for other values.
    """
    rate = int(data_transfer_rate_mb_per_s)
    canonical = {10: "10MBps", 100: "100MBps", 1000: "1GBps", 10000: "10GBps"}
    if rate in canonical:
        return canonical[rate]
    if rate <= 0:
        return "100MBps"
    return f"{int(round(rate))}MBps"


def _get_output_path(input_path: str,
                     target_wallclock_time: float = 43200.0,
                     failure_rate: float = 0.0,
                     data_transfer_rate_mb_per_s: float = 100.0,
                     output_base: str = "results/sim") -> str:
    """
    Generate output path based on input path structure with nested organization.

    Creates nested structure:
    {output_base}/{intermediate}/{case_name}/{time_dir}/fr{failure_rate}/{data_rate}/
    (time_dir is e.g. 15m, 30m, 1h, 2h, 4h, 8h, 12h, 24h; data_rate is e.g. 10MBps, 100MBps)

    Args:
        input_path: Path to input workflow file
        target_wallclock_time: Target wallclock time in seconds (default: 43200.0 = 12 hours)
        failure_rate: Job failure rate as percentage (default: 0.0)
        data_transfer_rate_mb_per_s: Network data transfer rate in MB/s (default: 100.0)
        output_base: Base directory for output (default: results/sim)

    Returns:
        Output path: {output_base}/.../fr{failure_rate}/{data_rate}/{filename}.json
    """
    input_path_obj = Path(input_path)
    base = Path(output_base)

    # Remove 'templates/' prefix if present
    if input_path_obj.parts[0] == 'templates':
        relative_path = input_path_obj.relative_to('templates')
    else:
        relative_path = input_path_obj

    # Format time directory: "15m", "30m" for <1h; "1h", "2h", ... for hours
    if target_wallclock_time < 3600:
        time_dir = f"{int(target_wallclock_time // 60)}m"
    else:
        time_dir = f"{int(target_wallclock_time // 3600)}h"

    # Format failure rate directory (e.g., fr0, fr1, fr5, fr10, fr25)
    failure_rate_int = int(round(failure_rate))
    fr_dir = f"fr{failure_rate_int}"

    # Data rate directory (e.g., 10MBps, 100MBps, 1GBps, 10GBps)
    data_rate_dir = _data_rate_dir_from_mbps(data_transfer_rate_mb_per_s)

    # Extract case name and preserve intermediate directories (e.g., "others")
    if len(relative_path.parts) >= 2:
        case_name = relative_path.parts[-2]
        filename = relative_path.name
        if len(relative_path.parts) >= 3:
            intermediate_dirs = relative_path.parts[:-2]
            output_dir = (
                base / Path(*intermediate_dirs) / case_name / time_dir / fr_dir / data_rate_dir
            )
        else:
            output_dir = base / case_name / time_dir / fr_dir / data_rate_dir
    else:
        case_name = relative_path.stem.split('_')[0] if '_' in relative_path.stem else relative_path.stem
        filename = relative_path.name
        output_dir = base / case_name / time_dir / fr_dir / data_rate_dir
    output_path = output_dir / filename

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
    parser.add_argument(
        '--failure-rate',
        type=int,
        default=0,
        help='Job failure rate as percentage (0-99, default: 0). Note: 100%% is not allowed as it prevents workflow convergence.'
    )
    parser.add_argument(
        '--data-transfer-rate',
        type=float,
        default=100.0,
        help='Network data transfer rate in MB/s for overhead calculation (default: 100.0)'
    )
    parser.add_argument(
        '--output-base',
        type=str,
        default='results/sim',
        help='Base directory for simulation output (default: results/sim).'
    )
    return parser.parse_args()


def main():
    """Main function with command line argument support."""
    args = parse_arguments()

    # Validate failure rate (protect against 100% which prevents convergence)
    if args.failure_rate >= 100:
        print("ERROR: Failure rate must be less than 100% to allow workflow convergence.")
        print("Please specify a failure rate between 0 and 99.")
        return

    if args.failure_rate < 0:
        print("ERROR: Failure rate cannot be negative.")
        return

    # Configure resources from command line arguments
    resource_config = ResourceConfig(
        target_wallclock_time=args.target_wallclock_time,
        max_job_slots=args.max_job_slots
    )

    # Create runner and execute workflow
    runner = WorkflowRunner(
        resource_config,
        failure_rate=args.failure_rate,
        data_transfer_rate_mb_per_s=args.data_transfer_rate
    )
    results = runner.run_workflow(args.input_workflow_path)

    # Print complete summary
    runner.print_complete_summary(results)

    # Write results to file with nested structure
    output_path = _get_output_path(
        args.input_workflow_path,
        target_wallclock_time=args.target_wallclock_time,
        failure_rate=args.failure_rate,
        data_transfer_rate_mb_per_s=args.data_transfer_rate,
        output_base=args.output_base
    )
    runner.write_complete_results(results, output_path)


if __name__ == "__main__":
    main()
