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
        
    def run_workflow(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a complete workflow simulation and analysis.
        
        Args:
            workflow_data: Dictionary containing workflow definition
            
        Returns:
            Dictionary containing simulation results and metrics
        """
        self.logger.info("Starting complete workflow execution and analysis")
        
        # Step 1: Run simulation
        simulation_result = self.simulator.simulate_workflow(workflow_data)
        
        if not simulation_result.success:
            self.logger.error(f"Workflow simulation failed: {simulation_result.error_message}")
            return {
                'simulation_result': simulation_result,
                'metrics': None,
                'success': False,
                'error_message': simulation_result.error_message
            }
        
        # Step 2: Convert simulation result to metrics-compatible format
        metrics_data = self._convert_simulation_to_metrics_data(simulation_result, workflow_data)
        
        # Step 3: Calculate metrics
        metrics_calculator = WorkflowMetricsCalculator(metrics_data)
        metrics = metrics_calculator.calculate_metrics()
        
        self.logger.info("Workflow execution and analysis completed successfully")
        
        return {
            'simulation_result': simulation_result,
            'metrics': metrics,
            'success': True,
            'error_message': None
        }
    
    def _convert_simulation_to_metrics_data(self, simulation_result: SimulationResult, 
                                          workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert simulation result to format compatible with metrics calculator."""
        # Start with original workflow data
        metrics_data = workflow_data.copy()
        
        # Add simulation-specific data
        metrics_data['workflow_id'] = simulation_result.workflow_id
        metrics_data['simulation_results'] = {
            'total_wall_time': simulation_result.total_wall_time,
            'total_jobs': simulation_result.total_jobs,
            'groups': []
        }
        
        # Convert group information
        for group in simulation_result.groups:
            group_data = {
                'group_id': group.group_id,
                'job_count': group.job_count,
                'total_execution_time': group.total_execution_time,
                'input_events': group.input_events,
                'tasksets': []
            }
            
            # Convert taskset information
            for taskset in group.tasksets:
                taskset_data = {
                    'taskset_id': taskset.taskset_id,
                    'group_name': taskset.group_name,
                    'time_per_event': taskset.time_per_event,
                    'memory': taskset.memory,
                    'multicore': taskset.multicore,
                    'size_per_event': taskset.size_per_event,
                    'group_input_events': taskset.group_input_events,
                    'execution_time': taskset.time_per_event * taskset.group_input_events
                }
                group_data['tasksets'].append(taskset_data)
            
            metrics_data['simulation_results']['groups'].append(group_data)
        
        return metrics_data
    
    def run_workflow_from_file(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Run workflow simulation and analysis from a JSON file.
        
        Args:
            filepath: Path to workflow JSON file
            
        Returns:
            Dictionary containing simulation results and metrics
        """
        with open(filepath, 'r') as f:
            workflow_data = json.load(f)
        
        return self.run_workflow(workflow_data)
    
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
        
        # Metrics summary
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"  Resource Efficiency: {metrics.resource_efficiency:.2f}")
        print(f"  Throughput: {metrics.throughput:.2f} events/second")
        print(f"  Success Rate: {metrics.success_rate:.2f}")
        print(f"  Total Execution Time: {metrics.total_execution_time:.2f}s")
        
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
        
        # Job statistics
        print(f"\n⚡ JOB STATISTICS:")
        job_wall_times = [job.wallclock_time for job in simulation.jobs]
        if job_wall_times:
            print(f"  Average Job Wall Time: {sum(job_wall_times)/len(job_wall_times):.2f}s")
            print(f"  Min Job Wall Time: {min(job_wall_times):.2f}s")
            print(f"  Max Job Wall Time: {max(job_wall_times):.2f}s")
        
        batch_sizes = [job.batch_size for job in simulation.jobs]
        if batch_sizes:
            print(f"  Average Batch Size: {sum(batch_sizes)/len(batch_sizes):.0f} events")
            print(f"  Min Batch Size: {min(batch_sizes)} events")
            print(f"  Max Batch Size: {max(batch_sizes)} events")
    
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
                'total_execution_time': results['metrics'].total_execution_time,
                'total_wall_time': results['metrics'].total_wall_time,
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
    # Load workflow data
    workflow_data = load_workflow_from_file('templates/3tasks_composition_001.json')
    
    # Configure resources
    resource_config = ResourceConfig(
        target_wallclock_time=43200.0,  # 12 hours
        max_job_slots=-1  # Infinite slots
    )
    
    # Create runner and execute workflow
    runner = WorkflowRunner(resource_config)
    results = runner.run_workflow(workflow_data)
    
    # Print complete summary
    runner.print_complete_summary(results)
    
    # Write results to file
    runner.write_complete_results(results, 'results/complete_workflow_results.json')


if __name__ == "__main__":
    from .workflow_simulator import load_workflow_from_file
    main()
