import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple
from pathlib import Path
import matplotlib
# Set non-interactive backend to avoid display issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pformat

# Add scripts directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Import plotting functions from the original visualization script
from workflow_visualization import (
    plot_io_patterns,
    plot_resource_utilization,
    plot_performance_metrics,
    plot_turnaround_time_comparison,
    generate_summary_table
)


def extract_construction_number(file_name: str) -> int:
    """Extract construction number from filenames like summary_const001 or seq_real_const_001."""
    match = re.search(r"const[_]?(\d+)", file_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0


def transform_real_data_to_simulation_format(real_data: Dict[str, Any], 
                                            file_name: str) -> Dict[str, Any]:
    """Transform real data JSON format to simulation data format expected by plotting functions.
    
    Args:
        real_data: Real data dictionary with structure:
            - document_name
            - time_metrics
            - cpu_metrics
            - memory_metrics
            - io_metrics
            - event_metrics
        file_name: Name of the source file
        
    Returns:
        Dictionary in simulation format with metrics that plotting functions expect
    """
    document_counts = real_data.get('document_counts', {})
    time_metrics = real_data.get('time_metrics', {})
    cpu_metrics = real_data.get('cpu_metrics', {})
    memory_metrics = real_data.get('memory_metrics', {})
    io_metrics = real_data.get('io_metrics', {})
    event_metrics = real_data.get('event_metrics', {})
    
    total_events = event_metrics.get('total_events', 0)
    total_groups = document_counts.get('total_groups', 0)
    
    # Calculate per-event metrics
    total_write_remote_mb = io_metrics.get('total_write_remote_mb', 0.0)
    total_read_remote_mb = io_metrics.get('total_read_remote_mb', 0.0)
    total_write_local_mb = io_metrics.get('total_write_local_mb', 0.0)
    total_read_local_mb = io_metrics.get('total_read_local_mb', 0.0)
    
    # Calculate per-event values
    total_write_remote_mb_per_event = (total_write_remote_mb / total_events 
                                     if total_events > 0 else 0.0)
    total_read_remote_mb_per_event = (total_read_remote_mb / total_events 
                                     if total_events > 0 else 0.0)
    total_write_local_mb_per_event = (total_write_local_mb / total_events 
                                     if total_events > 0 else 0.0)
    total_read_local_mb_per_event = (total_read_local_mb / total_events 
                                    if total_events > 0 else 0.0)
    
    # Network transfer = remote read + remote write
    network_transfer_mb_per_event = (total_read_remote_mb_per_event + 
                                     total_write_remote_mb_per_event)
    total_network_transfer_mb = total_read_remote_mb + total_write_remote_mb
    
    # Get actual CPU cores allocated from real data (sum of OriginalCpus across all jobs)
    total_cpu_cores_used = cpu_metrics.get('total_cpu_cores_used', 0.0)

    # Get total wallclock time and CPU allocated time
    total_cpu_allocated_time = cpu_metrics.get('total_cpu_allocated_time_sec', 0.0)
    total_wallclock_time = time_metrics.get('total_wallclock_time_with_overhead_sec', 0.0)

    # Extract construction number from filename
    composition_number = extract_construction_number(file_name)
    
    # Build simulation format dictionary
    sim_format = {
        '_file_name': file_name,
        'composition_number': composition_number,
        
        # Event metrics
        'event_throughput': event_metrics.get('event_throughput_with_overhead_events_per_sec', 0.0),
        'total_events_processed': total_events,
        
        # Time metrics
        'total_wall_time': total_wallclock_time,
        'total_turnaround_time': time_metrics.get('workflow_turnaround_time_sec', 0.0),
        'wall_time_per_event': time_metrics.get('wallclock_time_per_event_overhead_sec', 0.0),
        
        # CPU metrics
        'cpu_time_per_event': event_metrics.get('cpu_time_per_event_sec', 0.0),
        'cpu_utilization': cpu_metrics.get('cpu_utilization', 0.0),
        'total_cpu_allocated_time': total_cpu_allocated_time,
        'total_cpu_cores_used': total_cpu_cores_used,
        
        # Memory metrics
        # Use allocated memory for resource cost analysis (what was allocated, not what was used)
        'total_memory_used_mb': memory_metrics.get('total_memory_allocated_mb', 0.0),
        'memory_occupancy': memory_metrics.get('memory_utilization', 0.0),
        
        # I/O metrics - per event
        'total_write_remote_mb_per_event': total_write_remote_mb_per_event,
        'total_read_remote_mb_per_event': total_read_remote_mb_per_event,
        'total_write_local_mb_per_event': total_write_local_mb_per_event,
        'total_read_local_mb_per_event': total_read_local_mb_per_event,
        'network_transfer_mb_per_event': network_transfer_mb_per_event,
        
        # I/O metrics - totals
        'total_write_remote_mb': total_write_remote_mb,
        'total_read_remote_mb': total_read_remote_mb,
        'total_write_local_mb': total_write_local_mb,
        'total_read_local_mb': total_read_local_mb,
        'total_network_transfer_mb': total_network_transfer_mb,
        
        # Per-event resource metrics (use pre-calculated values from JSON for consistency)
        'cpu_cores_per_event': event_metrics.get('cpu_cores_per_event', 0.0),
        'memory_mb_per_event': event_metrics.get('memory_mb_per_event', 0.0),

        # Group information (from real data)
        'total_groups': total_groups,

        # Overhead flag (real data includes overhead)
        '_overhead_enabled': True
    }
    
    return sim_format


def find_real_data_files(directory_path: str) -> List[str]:
    """Find all summary JSON files in a directory.
    
    Looks for files matching pattern like 'summary_const*.json'
    """
    directory = Path(directory_path)
    if not directory.exists():
        raise FileNotFoundError(f"Directory '{directory_path}' not found")
    
    if not directory.is_dir():
        raise NotADirectoryError(f"'{directory_path}' is not a directory")
    
    # Find all JSON files matching summary pattern
    json_files = list(directory.glob("summary_*.json"))
    
    if not json_files:
        raise FileNotFoundError(
            f"No summary JSON files (summary_*.json) found in directory '{directory_path}'"
        )
    
    # Convert to strings and sort for consistent ordering
    return sorted([str(f) for f in json_files])


def process_real_data_directory(directory_path: str) -> tuple:
    """Process real data summary files in a directory.
    
    Args:
        directory_path: Path to directory containing real data summary JSON files
        
    Returns:
        tuple: (all_groups, all_jobs, all_simulation_data)
        - all_groups: List of minimal group entries (one per group) for plotting compatibility
        - all_jobs: Empty list (real data doesn't have job-level detail)
        - all_simulation_data: List of transformed simulation format dictionaries
    """
    summary_files = find_real_data_files(directory_path)
    
    if not summary_files:
        print(f"Warning: No summary files found in '{directory_path}'")
        return [], [], []
    
    all_simulation_data = []
    all_groups = []
    files_processed = 0
    
    print(f"Processing {len(summary_files)} real data summary files")
    
    for file_path in summary_files:
        try:
            print(f"  Loading and processing: {Path(file_path).name}")
            
            # Load the file
            with open(file_path, 'r') as f:
                real_data = json.load(f)
            file_name = Path(file_path).name
            
            # Transform to simulation format
            sim_format = transform_real_data_to_simulation_format(real_data, file_name)
            all_simulation_data.append(sim_format)
            
            # Create minimal group entries for plotting compatibility
            # The plotting functions use len(sim_groups_for_file) to get num_groups
            # We create one entry per group so the count is correct
            total_groups = sim_format.get('total_groups', 0)
            for group_idx in range(total_groups):
                group_entry = {
                    'group_id': f'Group{group_idx + 1}',
                    'file_name': file_name,
                }
                all_groups.append(group_entry)

            files_processed += 1
            
        except Exception as e:
            print(f"  Warning: Failed to process {Path(file_path).name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if files_processed == 0:
        raise ValueError(
            f"No valid real data processed from directory '{directory_path}'"
        )
    
    print(f"Successfully processed {files_processed} summary files")
    if all_simulation_data:
        print(f"Sample transformed metrics: {pformat(all_simulation_data[0])}")
    
    return all_groups, [], all_simulation_data


# Map composition numbers to canonical workflow type names for real execution plots
REAL_WORKFLOW_LABELS = {1: "StepChain", 16: "TaskChain"}


def _sort_workflows_by_composition(
    all_simulation_data: List[Dict],
    sim_groups: List[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """Order workflows by composition_number so plots match StepChain → TaskChain."""
    order = sorted(
        range(len(all_simulation_data)),
        key=lambda i: all_simulation_data[i].get("composition_number", 0),
    )
    sorted_data = [all_simulation_data[i] for i in order]
    sorted_groups: List[Dict] = []
    for i in order:
        file_name = all_simulation_data[i].get("_file_name", "")
        sorted_groups.extend(g for g in sim_groups if g.get("file_name") == file_name)
    return sorted_data, sorted_groups


def _build_display_labels(all_simulation_data: List[Dict]) -> List[str]:
    """Build X-axis labels for real workflow plots (StepChain/TaskChain or Const N)."""
    labels = []
    for sim_data in all_simulation_data:
        comp = sim_data.get("composition_number", 0)
        labels.append(REAL_WORKFLOW_LABELS.get(comp, f"Const {comp}"))
    return labels


def generate_workflow_visualizations(all_simulation_data: List[Dict],
                                    sim_groups: List[Dict],
                                    jobs: List[Dict],
                                    output_dir: str) -> None:
    """Generate all workflow comparison visualizations for real data.

    Args:
        all_simulation_data: List of simulation data dictionaries (transformed from real data)
        sim_groups: List of group metrics dictionaries (empty for real data)
        jobs: List of job metrics dictionaries (empty for real data)
        output_dir: Output directory for visualization files
    """
    if len(all_simulation_data) == 0:
        print("\nNo real data files found, skipping visualizations")
        return
    
    all_simulation_data, sim_groups = _sort_workflows_by_composition(
        all_simulation_data, sim_groups
    )
    print(f"\nGenerating workflow comparison for {len(all_simulation_data)} real data workflow(s)...")
    display_labels = _build_display_labels(all_simulation_data)
    try:
        # Generate summary table (works for any number of workflows)
        generate_summary_table(
            all_simulation_data=all_simulation_data,
            sim_groups=sim_groups,
            output_dir=output_dir
        )
        
        # Generate plots (only if more than one workflow)
        if len(all_simulation_data) > 1:
            plot_io_patterns(
                all_simulation_data=all_simulation_data,
                sim_groups=sim_groups,
                jobs=jobs,
                output_dir=output_dir,
                custom_labels=display_labels
            )
            
            plot_resource_utilization(
                all_simulation_data=all_simulation_data,
                sim_groups=sim_groups,
                jobs=jobs,
                output_dir=output_dir,
                custom_labels=display_labels
            )
            
            plot_performance_metrics(
                all_simulation_data=all_simulation_data,
                sim_groups=sim_groups,
                jobs=jobs,
                output_dir=output_dir,
                custom_labels=display_labels
            )

            plot_turnaround_time_comparison(
                all_simulation_data=all_simulation_data,
                sim_groups=sim_groups,
                jobs=jobs,
                output_dir=output_dir,
                custom_labels=display_labels
            )
        else:
            print(f"  => Skipping comparison plots (only {len(all_simulation_data)} workflow found)")
            print(f"     (Comparison plots require at least 2 workflows)")
    except Exception as e:
        print(f"Warning: Could not generate workflow comparison: {e}")
        import traceback
        traceback.print_exc()


def _run_visualization_pass(input_dir: str, output_dir: str) -> None:
    """Load summaries from ``input_dir`` and write comparison plots to ``output_dir``."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Processing real data from directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    groups, jobs, all_simulation_data = process_real_data_directory(input_dir)
    generate_workflow_visualizations(
        all_simulation_data=all_simulation_data,
        sim_groups=groups,
        jobs=jobs,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create visualizations for real workflow data analysis results",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="results/real",
        help="Directory with summary_*.json files (default: results/real)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for PNGs (default: same as --input-dir)",
    )
    parser.add_argument(
        "--also-normalized",
        action="store_true",
        help="Also run for results/real_norm (raw pass still uses --input-dir/--output-dir)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    passes = [(args.input_dir, output_dir)]
    if args.also_normalized:
        passes.append(("results/real_norm", "results/real_norm"))

    try:
        for in_dir, out_dir in passes:
            print("\n" + "=" * 60)
            _run_visualization_pass(in_dir, out_dir)
        print("\n" + "=" * 60)
        print("Visualization complete!")
        print("=" * 60)
    except Exception as e:
        print(f"Error processing real data: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

