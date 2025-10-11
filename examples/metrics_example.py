#!/usr/bin/env python3
"""
Example usage of the WorkflowMetricsCalculator class.

This script demonstrates how to calculate, display, and save workflow metrics
from a workflow composition JSON file.
"""

import json
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from workflow_metrics import WorkflowMetricsCalculator
from workflow_runner import WorkflowRunner


def main():
    """Main example function."""
    # Load the workflow template
    template_path = Path(__file__).parent.parent / "templates" / "3tasks_composition_001.json"

    if not template_path.exists():
        print(f"Template file not found: {template_path}")
        return

    # Load workflow data
    with open(template_path, 'r') as f:
        workflow_data = json.load(f)

    print("Loading workflow data from:", template_path)
    print(f"Workflow: {workflow_data.get('Comments', 'Unknown')}")
    print(f"Number of tasks: {workflow_data.get('NumTasks', 0)}")
    print(f"Requested events: {workflow_data.get('RequestNumEvents', 0)}")

    # Run simulation to get accurate results
    print("\nRunning workflow simulation...")
    runner = WorkflowRunner()
    results = runner.run_workflow(template_path)

    if not results['success']:
        print(f"Simulation failed: {results['error_message']}")
        return

    # Calculate metrics from simulation results
    print("\nCalculating workflow metrics...")
    calculator = WorkflowMetricsCalculator()
    metrics = calculator.calculate_metrics(results['simulation_result'])

    # Print metrics summary
    calculator.print_metrics()

    # Save metrics to file
    output_path = Path(__file__).parent.parent / "results" / "3tasks_composition_001_metrics.json"
    output_path.parent.mkdir(exist_ok=True)

    calculator.write_metrics_to_file(output_path)
    print(f"\nMetrics saved to: {output_path}")

    # Get summary for further processing
    summary = calculator.get_metrics_summary()
    print(f"\nMetrics Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
