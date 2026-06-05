#!/usr/bin/env python3
"""
Workflow Simulation Example

This example demonstrates how to use the WorkflowSimulator and WorkflowRunner
to simulate workflow execution with group-based job scheduling.

For command line usage, see the main scripts:
- python src/workflow_runner.py --help
- python src/workflow_simulator.py --help
"""

import sys
import os
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from workflow_runner import WorkflowRunner
from workflow_simulator import ResourceConfig


def main():
    """Run workflow simulation example."""
    print("="*80)
    print("WORKFLOW SIMULATION EXAMPLE")
    print("="*80)
    print("Note: For command line usage, run:")
    print("  python src/workflow_runner.py --help")
    print("  python src/workflow_simulator.py --help")
    print()
    
    # Configure logging (optional - set to WARNING to reduce output)
    logging.basicConfig(level=logging.WARNING)
    
    # Set workflow file path
    workflow_file = (
        Path(__file__).parent.parent / 'templates' / 'others' / 'seq_real' / 'seq_real_const_001.json'
    )
    print(f"Using workflow file: {workflow_file}")
    
    if not workflow_file.exists():
        print(f"❌ Error: Workflow file not found: {workflow_file}")
        return 1
    
    # Configure resources
    resource_config = ResourceConfig(
        target_wallclock_time=43200.0,  # 12 hours in seconds
        max_job_slots=-1  # -1 means infinite job slots
    )
    
    print(f"Resource Configuration:")
    print(f"  Target Wallclock Time: {resource_config.target_wallclock_time/3600:.1f} hours")
    print(f"  Max Job Slots: {'Infinite' if resource_config.max_job_slots == -1 else resource_config.max_job_slots}")
    print()
    
    # Create workflow runner (failure_rate and data_transfer_rate passed explicitly)
    runner = WorkflowRunner(
        resource_config,
        job_failure_rate=0,
        data_transfer_rate_mb_per_s=100.0,
    )
    
    # Run simulation
    print("🚀 Starting workflow simulation...")
    results = runner.run_workflow(workflow_file)
    
    if not results['success']:
        print(f"❌ Simulation failed: {results['error_message']}")
        return 1
    
    # Print results
    runner.print_complete_summary(results)
    
    # Save results to file
    output_file = Path(__file__).parent.parent / 'results' / 'simulation_example_results.json'
    output_file.parent.mkdir(exist_ok=True)
    
    print(f"\n💾 Saving results to: {output_file}")
    runner.write_complete_results(results, output_file)
    
    print("\n✅ Workflow simulation completed successfully!")
    print("\n💡 Tip: Use command line interface for easier usage:")
    print("  python src/workflow_runner.py --target-wallclock-time 3600")
    return 0


if __name__ == "__main__":
    sys.exit(main())
