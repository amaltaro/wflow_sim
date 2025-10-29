import argparse
import json
import os
from typing import List, Dict, Any
from collections import defaultdict
from math import ceil
from pprint import pformat
import matplotlib
# Set non-interactive backend to avoid display issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path


def plot_workflow_comparison(construction_metrics: List[Dict], output_dir: str = "plots", custom_labels: List[str] = None, groups_data: List[Dict] = None):
    """Create a comprehensive comparison of workflow constructions.

    This function creates multiple visualizations to help identify trade-offs
    between different workflow constructions.
    """
    print(f"Creating comprehensive workflow construction comparison for {len(construction_metrics)} constructions")

    # Extract metrics for comparison
    num_groups = []
    event_throughputs = []
    total_cpu_times = []
    stored_data_per_event = []
    total_stored_data = []
    input_data_per_event = []
    output_data_per_event = []
    group_combinations = []

    for metrics in construction_metrics:
        num_groups.append(metrics["num_groups"])
        event_throughputs.append(metrics["event_throughput"])
        total_cpu_times.append(metrics["total_cpu_time"])
        stored_data_per_event.append(metrics["write_remote_per_event_mb"])
        total_stored_data.append(metrics["total_write_remote_mb"])
        input_data_per_event.append(metrics["read_remote_per_event_mb"])
        output_data_per_event.append(metrics["write_local_per_event_mb"])
        group_combinations.append(" + ".join(metrics["groups"]))

    # Convert lists to numpy arrays for numerical operations
    num_groups = np.array(num_groups)
    event_throughputs = np.array(event_throughputs)
    total_cpu_times = np.array(total_cpu_times)
    stored_data_per_event = np.array(stored_data_per_event)
    total_stored_data = np.array(total_stored_data)
    input_data_per_event = np.array(input_data_per_event)
    output_data_per_event = np.array(output_data_per_event)

    # Create a figure with multiple subplots - now with fixed, professional proportions
    if len(construction_metrics) <= 2:
        fig = plt.figure(figsize=(12, 16))
    else:
        fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1])  # Equal height ratios for all rows

    # Always use short labels for plots ("Const 1", "Const 2", etc.)
    # custom_labels are only used in the text output file
    construction_labels = [f"Const {i+1}" for i, _ in enumerate(construction_metrics)]

    # 1. Group Size Distribution
    ax3 = fig.add_subplot(gs[0, 0])
    group_sizes = []
    for metrics in construction_metrics:
        sizes = [len(group["tasks"]) for group in metrics["group_details"]]
        group_sizes.append(sizes)

    # Create a box plot for group sizes
    ax3.boxplot(group_sizes, tick_labels=construction_labels)
    ax3.set_xlabel("Workflow Construction")
    ax3.set_ylabel("Number of Tasks per Group")
    ax3.set_title("Group Size Distribution")
    ax3.set_xticklabels(construction_labels, rotation=45)
    ax3.grid(True)
    ax3.set_ylim(bottom=0)  # Set y-axis to start at 0

    # 2. Data Flow Analysis (Updated to use per-event metrics)
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(construction_metrics))
    width = 0.25
    ax2.bar(x - width, input_data_per_event, width, label='Remote Read')
    ax2.bar(x, output_data_per_event, width, label='Local Write')
    ax2.bar(x + width, stored_data_per_event, width, label='Remote Write')
    ax2.set_xlabel("Workflow Construction")
    ax2.set_ylabel("Data Volume per Event (MB)")
    ax2.set_title("Data Volume Analysis Per Event")
    ax2.set_xticks(x)
    ax2.set_xticklabels(construction_labels, rotation=45)
    ax2.legend()
    ax2.grid(True)

    # 3. Total Data Volume Analysis (Stacked Bar)
    ax10 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(construction_metrics))
    width = 0.6
    bottom = np.zeros(len(construction_metrics))

    # Convert MB to GB for better readability
    remote_read_gb = [m["total_read_remote_mb"] / 1024.0 for m in construction_metrics]
    local_write_gb = [m["total_write_local_mb"] / 1024.0 for m in construction_metrics]
    remote_write_gb = [m["total_write_remote_mb"] / 1024.0 for m in construction_metrics]

    # Plot each data type as a layer in the stack
    ax10.bar(x, remote_read_gb, width, label='Remote Read', bottom=bottom)
    bottom += remote_read_gb

    ax10.bar(x, local_write_gb, width, label='Local Write', bottom=bottom)
    bottom += local_write_gb

    ax10.bar(x, remote_write_gb, width, label='Remote Write', bottom=bottom)

    # Add total value labels on top of each bar
    totals_gb = [rr + lw + rw for rr, lw, rw in zip(remote_read_gb, local_write_gb, remote_write_gb)]
    for i, total in enumerate(totals_gb):
        ax10.text(i, total, f'{total:.1f}', ha='center', va='bottom')

    ax10.set_xlabel("Workflow Construction")
    ax10.set_ylabel("Total Data Volume (GB)")
    ax10.set_title("Total Workflow Data Volume Analysis")
    ax10.set_xticks(x)
    ax10.set_xticklabels(construction_labels, rotation=45)
    ax10.legend()
    ax10.grid(True)

    # 4. Performance vs Remote Write Efficiency (simplified scatter plot)
    ax1 = fig.add_subplot(gs[1, 1])

    # Create a simple scatter plot
    scatter = ax1.scatter(event_throughputs, stored_data_per_event,
                         c=num_groups, cmap='viridis', s=100, alpha=0.7)

    # Add colorbar with discrete integer values
    cbar = plt.colorbar(scatter, ax=ax1, label="Number of Groups")

    # Get unique values and set discrete ticks
    unique_groups = np.unique(num_groups)
    cbar.set_ticks(unique_groups)
    cbar.set_ticklabels([f"{int(x)}" for x in unique_groups])

    ax1.set_xlabel("Event Throughput (events/second)")
    ax1.set_ylabel("Remote Write Data per Event (MB)")
    ax1.set_title("Performance vs Remote Write Efficiency")
    ax1.grid(True, alpha=0.3)

    # set x-axis to start at 0 and add 10% padding to the right
    ax1.set_xlim(left=0, right=np.max(event_throughputs) * 1.1)
    # set y-axis to start at 0
    ax1.set_ylim(bottom=0)

    # 5. Network Transfer Analysis
    ax7 = fig.add_subplot(gs[2, 0])
    network_transfer = []
    for metrics in construction_metrics:
        # Network transfer = remote read + remote write (only remote operations)
        # Use pre-calculated field if available, otherwise calculate from per-event metrics
        transfer = metrics.get("network_transfer_per_event_mb")
        if transfer is None:
            # Fallback: calculate from read and write remote per event
            transfer = metrics.get("read_remote_per_event_mb", 0.0) + metrics.get("write_remote_per_event_mb", 0.0)
        network_transfer.append(transfer)

    ax7.bar(range(len(construction_metrics)), network_transfer)
    ax7.set_xlabel("Workflow Construction")
    ax7.set_ylabel("Network Transfer per Event (MB)")
    ax7.set_title("Network Transfer Analysis")
    ax7.set_xticks(range(len(construction_metrics)))
    ax7.set_xticklabels(construction_labels, rotation=45)
    ax7.grid(True)

    # 6. CPU Utilization Analysis
    ax4 = fig.add_subplot(gs[2, 1])
    cpu_utilization = []
    cpu_std = []  # Store standard deviations

    # Check if groups_data is available
    if groups_data is not None:
        for metrics in construction_metrics:
            # Get CPU utilization ratio for each group from the groups data
            util = []
            for group_id in metrics["groups"]:
                # Find the corresponding group in the groups data
                group_data = next((g for g in groups_data if g.get("group_id") == group_id), None)
                if group_data and "resource_metrics" in group_data:
                    util.append(group_data["resource_metrics"]["cpu"]["utilization_ratio"])
            # Calculate average and standard deviation of CPU utilization
            if util:
                avg_util = sum(util) / len(util)
                std_util = np.std(util)
                cpu_utilization.append(avg_util)
                cpu_std.append(std_util)
            else:
                cpu_utilization.append(0)
                cpu_std.append(0)
    else:
        # Use metrics data if available
        for metrics in construction_metrics:
            cpu_util = metrics.get("cpu_utilization", 0.0)
            cpu_utilization.append(cpu_util)
            cpu_std.append(0)

    # Create bar plot with error bars (only if we have std data)
    x = range(len(construction_metrics))
    if any(std > 0 for std in cpu_std):
        ax4.bar(x, cpu_utilization, yerr=cpu_std, capsize=5)
    else:
        ax4.bar(x, cpu_utilization)
    ax4.set_xlabel("Workflow Construction")
    ax4.set_ylabel("CPU Utilization Ratio")
    ax4.set_title("CPU Utilization Analysis\n(Average CPU Usage / Allocated CPU ± Std Dev)")
    ax4.set_xticks(x)
    ax4.set_xticklabels(construction_labels, rotation=45)
    ax4.set_ylim(bottom=0)  # Set Y-axis to start at 0
    ax4.grid(True)

    # 7. Memory Utilization Analysis
    ax6 = fig.add_subplot(gs[3, 0])
    memory_utilization = []
    memory_std = []  # Store standard deviations

    # Check if groups_data is available
    if groups_data is not None:
        for metrics in construction_metrics:
            # Get memory occupancy for each group from the groups data
            occupancies = []
            for group_id in metrics["groups"]:
                # Find the corresponding group in the groups data
                group_data = next((g for g in groups_data if g.get("group_id") == group_id), None)
                if group_data and "resource_metrics" in group_data:
                    occupancies.append(group_data["resource_metrics"]["memory"]["occupancy"])
            # Calculate average and standard deviation of memory occupancy
            if occupancies:
                avg_occupancy = sum(occupancies) / len(occupancies)
                std_occupancy = np.std(occupancies)
                memory_utilization.append(avg_occupancy)
                memory_std.append(std_occupancy)
            else:
                memory_utilization.append(0)
                memory_std.append(0)
    else:
        # Use metrics data if available
        for metrics in construction_metrics:
            mem_util = metrics.get("memory_occupancy", 0.0)
            memory_utilization.append(mem_util)
            memory_std.append(0)

    # Create bar plot with error bars (only if we have std data)
    x = range(len(construction_metrics))
    if any(std > 0 for std in memory_std):
        ax6.bar(x, memory_utilization, yerr=memory_std, capsize=5)
    else:
        ax6.bar(x, memory_utilization)
    ax6.set_xlabel("Workflow Construction")
    ax6.set_ylabel("Memory Utilization Ratio")
    ax6.set_title("Memory Utilization Analysis\n(Average Memory Occupancy ± Std Dev)")
    ax6.set_xticks(x)
    ax6.set_xticklabels(construction_labels, rotation=45)
    ax6.grid(True)

    # 8. Event Processing Analysis
    ax5 = fig.add_subplot(gs[3, 1])
    events_per_group = []
    for metrics in construction_metrics:
        events = [group["total_events"] for group in metrics["group_details"]]
        events_per_group.append(events)

    ax5.boxplot(events_per_group, tick_labels=construction_labels)
    ax5.set_xlabel("Workflow Construction")
    ax5.set_ylabel("Events per Group")
    ax5.set_title("Event Processing Distribution")
    ax5.set_xticklabels(construction_labels, rotation=45)
    ax5.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "workflow_comparison.png"))
    plt.close()

    # Create a detailed comparison table
    with open(os.path.join(output_dir, "workflow_comparison.txt"), "w") as f:
        f.write("Workflow Construction Comparison\n")
        f.write("==============================\n\n")

        for i, metrics in enumerate(construction_metrics, 1):
            # Use custom label if provided, otherwise use default "Construction" label
            if custom_labels and i <= len(custom_labels):
                construction_label = custom_labels[i-1]
            else:
                construction_label = f"Construction {i}"
            f.write(f"{construction_label}:\n")
            f.write(f"  Groups: {metrics['groups']}\n")
            f.write(f"  Number of Groups: {metrics['num_groups']}\n")
            f.write(f"  Event Throughput: {metrics['event_throughput']:.4f} events/second\n")
            f.write(f"  Total CPU Time: {metrics['total_cpu_time']:.2f} seconds\n")
            f.write("  Total Data Volumes for one job of each group:\n")
            f.write(f"    Remote Read Data: {metrics['total_read_remote_mb']:.2f} MB\n")
            f.write(f"    Local Write Data: {metrics['total_write_local_mb']:.2f} MB\n")
            f.write(f"    Remote Write Data: {metrics['total_write_remote_mb']:.2f} MB\n")
            f.write("  Data Flow Metrics (per event):\n")
            f.write(f"    Remote Read Data: {metrics['read_remote_per_event_mb']:.3f} MB/event\n")
            f.write(f"    Local Write Data: {metrics['write_local_per_event_mb']:.3f} MB/event\n")
            f.write(f"    Remote Write Data: {metrics['write_remote_per_event_mb']:.3f} MB/event\n")
            if i <= len(memory_utilization):
                f.write(f"  Memory Utilization: {memory_utilization[i-1]:.2f}\n")
            if i <= len(network_transfer):
                f.write(f"  Network Transfer: {network_transfer[i-1]:.2f} MB\n")
            f.write("  Workflow Performance Metrics:\n")
            f.write(f"    Total CPU Time: {metrics['total_cpu_time']:.2f} seconds\n")
            f.write(f"    Total Wallclock Time: {metrics['total_wallclock_time']:.2f} seconds\n")
            f.write(f"    Total Memory: {metrics['total_memory_mb']:,.0f} MB\n")
            f.write(f"    Total Network Transfer: {metrics['total_network_transfer_mb']:,.0f} MB\n")
            f.write("  Group Details:\n")
            for group in metrics["group_details"]:
                f.write(f"    {group['group_id']}:\n")
                f.write(f"      Tasks: {group['tasks']}\n")
                f.write(f"      Events per Task: {group['events_per_task']}\n")
                f.write(f"      CPU Time: {group['cpu_seconds']:.2f} seconds\n")
                f.write("      Data Flow (per event):\n")
                f.write(f"        Remote Read: {group['read_remote_per_event_mb']:.3f} MB/event\n")
                f.write(f"        Local Write: {group['write_local_per_event_mb']:.3f} MB/event\n")
                f.write(f"        Remote Write: {group['write_remote_per_event_mb']:.3f} MB/event\n")
            f.write("\n")


def build_construction_metrics(simulation_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Transform simulation data into the format expected by plot_workflow_comparison.

    Args:
        simulation_data_list: List of simulation data dictionaries from JSON files

    Returns:
        List of construction metrics dictionaries
    """
    construction_metrics = []

    for sim_data in simulation_data_list:
        metrics = sim_data.get('metrics', {})
        sim_result = sim_data.get('simulation_result', {})
        groups = sim_result.get('groups', [])
        jobs = sim_result.get('jobs', [])

        # Extract group IDs
        group_ids = [g['group_id'] for g in groups]

        # Build group_details
        group_details = []
        for group in groups:
            # Get taskset IDs for this group
            tasks = [ts['taskset_id'] for ts in group.get('tasksets', [])]

            # Calculate events per task (using first job of this group)
            group_jobs = [j for j in jobs if j['group_id'] == group['group_id']]
            events_per_task = group_jobs[0]['batch_size'] if group_jobs else group['input_events']

            # Calculate per-event data metrics from first job
            first_job = group_jobs[0] if group_jobs else None
            total_events = group['input_events'] * group.get('job_count', 1)

            read_remote_per_event = 0.0
            write_local_per_event = 0.0
            write_remote_per_event = 0.0

            if first_job:
                if total_events > 0:
                    read_remote_per_event = first_job.get('total_read_remote_mb', 0.0) / events_per_task
                    write_local_per_event = first_job.get('total_write_local_mb', 0.0) / events_per_task
                    write_remote_per_event = first_job.get('total_write_remote_mb', 0.0) / events_per_task

            # Calculate CPU seconds for the group
            cpu_seconds = sum(ts['time_per_event'] * events_per_task * ts.get('multicore', 1)
                            for ts in group.get('tasksets', []))

            group_details.append({
                'group_id': group['group_id'],
                'tasks': tasks,
                'events_per_task': events_per_task,
                'cpu_seconds': cpu_seconds,
                'read_remote_per_event_mb': read_remote_per_event,
                'write_local_per_event_mb': write_local_per_event,
                'write_remote_per_event_mb': write_remote_per_event,
                'total_events': total_events
            })

        # Build construction metrics
        construction_metric = {
            'groups': group_ids,
            'num_groups': len(groups),
            'event_throughput': metrics.get('event_throughput', 0.0),
            'total_cpu_time': metrics.get('total_cpu_allocated_time', 0.0),
            'write_remote_per_event_mb': metrics.get('total_write_remote_mb_per_event', 0.0),
            'total_write_remote_mb': metrics.get('total_write_remote_mb', 0.0),
            'read_remote_per_event_mb': metrics.get('total_read_remote_mb_per_event', 0.0),
            'write_local_per_event_mb': metrics.get('total_write_local_mb_per_event', 0.0),
            'total_read_remote_mb': metrics.get('total_read_remote_mb', 0.0),
            'total_write_local_mb': metrics.get('total_write_local_mb', 0.0),
            'total_wallclock_time': metrics.get('total_wall_time', 0.0),
            'total_memory_mb': metrics.get('total_memory_used_mb', 0.0),
            'total_network_transfer_mb': metrics.get('total_network_transfer_mb', 0.0),
            'network_transfer_per_event_mb': metrics.get('network_transfer_mb_per_event', 0.0),
            'cpu_utilization': metrics.get('cpu_utilization', 0.0),
            'memory_occupancy': metrics.get('memory_occupancy', 0.0),
            'group_details': group_details
        }

        construction_metrics.append(construction_metric)

    return construction_metrics


def find_simulation_files(directory_path: str) -> List[str]:
    """Find all JSON simulation files in a directory."""
    directory = Path(directory_path)
    if not directory.exists():
        raise FileNotFoundError(f"Directory '{directory_path}' not found")

    if not directory.is_dir():
        raise NotADirectoryError(f"'{directory_path}' is not a directory")

    # Find all JSON files recursively
    json_files = list(directory.glob("**/*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in directory '{directory_path}'")

    # Convert to strings and sort for consistent ordering
    return sorted([str(f) for f in json_files])


def extract_group_metrics(simulation_data: Dict[str, Any], file_name: str) -> List[Dict[str, Any]]:
    """Extract group-level metrics from simulation data."""
    groups = []

    for group in simulation_data.get('simulation_result', {}).get('groups', []):
        group_metrics = {
            'group_id': group['group_id'],
            'group_job_count': group['job_count'],
            'group_input_events': group['input_events'],
            'group_total_execution_time': group['total_execution_time'],
            'group_exact_job_count': group['exact_job_count'],
            'group_taskset_count': len(group['tasksets']),
            'group_dependencies': group.get('dependencies', []),
            'file_name': file_name,
            'composition_number': simulation_data.get('metrics', {}).get('composition_number', 0)
        }

        # Calculate aggregated metrics from tasksets
        group_size_per_event = sum(ts['size_per_event'] for ts in group['tasksets'])
        group_time_per_event = sum(ts['time_per_event'] for ts in group['tasksets'])
        group_memory = sum(ts['memory'] for ts in group['tasksets'])
        group_memory_max = max(ts['memory'] for ts in group['tasksets'])
        group_multicore = sum(ts['multicore'] for ts in group['tasksets'])
        group_multicore_max = max(ts['multicore'] for ts in group['tasksets'])

        group_metrics.update({
            'group_size_per_event': group_size_per_event,
            'group_time_per_event': group_time_per_event,
            'group_memory': group_memory,
            'group_memory_max': group_memory_max,
            'group_multicore': group_multicore,
            'group_multicore_max': group_multicore_max,
            'group_time_per_event_avg': group_time_per_event / len(group['tasksets']),
            'group_memory_avg': group_memory / len(group['tasksets']),
            'group_multicore_avg': group_multicore / len(group['tasksets']),
            'group_size_per_event_avg': group_size_per_event / len(group['tasksets'])
        })

        groups.append(group_metrics)

    return groups


def extract_job_metrics(simulation_data: Dict[str, Any], file_name: str) -> List[Dict[str, Any]]:
    """Extract job-level metrics from simulation data.

    Only extracts metrics for the first job of each group to minimize memory usage
    while still allowing analysis of group behavior. Aggregated data is available
    at the metrics level.
    """
    jobs = []
    processed_groups = set()

    for job in simulation_data.get('simulation_result', {}).get('jobs', []):
        group_id = job['group_id']

        # Only process the first job of each group
        if group_id in processed_groups:
            continue

        processed_groups.add(group_id)

        job_metrics = {
            'job_id': job['job_id'],
            'group_id': group_id,
            'batch_size': job['batch_size'],
            'wallclock_time': job['wallclock_time'],
            'start_time': job['start_time'],
            'end_time': job['end_time'],
            'status': job['status'],
            'total_cpu_used_time': job['total_cpu_used_time'],
            'total_cpu_allocated_time': job['total_cpu_allocated_time'],
            'total_write_local_mb': job['total_write_local_mb'],
            'total_write_remote_mb': job['total_write_remote_mb'],
            'total_read_local_mb': job['total_read_local_mb'],
            'total_read_remote_mb': job['total_read_remote_mb'],
            'total_network_transfer_mb': job['total_network_transfer_mb'],
            'file_name': file_name,
            'composition_number': simulation_data.get('metrics', {}).get('composition_number', 0)
        }

        # Calculate derived metrics
        cpu_utilization = (job['total_cpu_used_time'] / job['total_cpu_allocated_time']
                          if job['total_cpu_allocated_time'] > 0 else 0)

        job_metrics.update({
            'cpu_utilization': cpu_utilization,
            'throughput_eps': job['batch_size'] / job['wallclock_time'] if job['wallclock_time'] > 0 else 0,
            'cpu_efficiency': cpu_utilization,
            'data_io_ratio': (job['total_write_local_mb'] + job['total_write_remote_mb']) /
                           (job['total_read_local_mb'] + job['total_read_remote_mb'])
                           if (job['total_read_local_mb'] + job['total_read_remote_mb']) > 0 else 0
        })

        jobs.append(job_metrics)

    return jobs


def process_simulation_directory(directory_path: str) -> tuple:
    """Process all simulation files in a directory incrementally to minimize memory usage.

    This function processes each file as it loads it, extracting metrics immediately
    and keeping full simulation data for comparison plots.

    Returns:
        tuple: (all_groups, all_jobs, all_simulation_data) - aggregated metrics and
               all simulation data for comparison plots
    """
    simulation_files = find_simulation_files(directory_path)
    all_groups = []
    all_jobs = []
    all_simulation_data = []
    files_processed = 0

    print(f"Found {len(simulation_files)} JSON files in directory")

    for file_path in simulation_files:
        try:
            print(f"  Loading and processing: {Path(file_path).name}")

            # Load the file
            with open(file_path, 'r') as f:
                simulation_data = json.load(f)
            file_name = Path(file_path).name

            # Extract metrics immediately (this reduces the memory footprint)
            groups = extract_group_metrics(simulation_data, file_name)
            jobs = extract_job_metrics(simulation_data, file_name)

            # Accumulate the extracted metrics
            all_groups.extend(groups)
            all_jobs.extend(jobs)

            # Keep simulation data for comparison plots
            simulation_data['_file_name'] = file_name
            all_simulation_data.append(simulation_data)

            files_processed += 1

        except Exception as e:
            print(f"  Warning: Failed to process {Path(file_path).name}: {e}")
            continue

    if files_processed == 0:
        raise ValueError(f"No valid simulation data processed from directory '{directory_path}'")

    print(f"Successfully processed {files_processed} simulation files")
    print(f"Extracted {len(all_groups)} groups and {len(all_jobs)} jobs")
    if all_simulation_data:
        print(f"Sample workflow metrics: {pformat(all_simulation_data[0].get('metrics', {}))}")

    return all_groups, all_jobs, all_simulation_data


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Create visualizations for workflow simulation results using pandas/matplotlib/seaborn'
    )
    parser.add_argument('simulation_directory', type=str,
                       help='Path to directory containing simulation result JSON files')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Base output directory (default: output)')
    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Processing simulation data from directory: {args.simulation_directory}")
    try:
        # Process files to get metrics and full simulation data
        groups, jobs, all_simulation_data = process_simulation_directory(args.simulation_directory)

        # Create workflow comparison if we have multiple simulations
        if len(all_simulation_data) > 1:
            print(f"\nGenerating workflow comparison for {len(all_simulation_data)} workflows...")
            try:
                # Transform simulation data into construction metrics format
                construction_metrics = build_construction_metrics(all_simulation_data)

                # Generate custom labels from file names
                custom_labels = [Path(sim_data['_file_name']).stem
                               for sim_data in all_simulation_data]

                # Call the comparison function
                plot_workflow_comparison(
                    construction_metrics=construction_metrics,
                    output_dir=args.output_dir,
                    custom_labels=custom_labels,
                    groups_data=None  # Can be enhanced to extract from simulation data if needed
                )
                print(f"Workflow comparison saved to {args.output_dir}/workflow_comparison.png")
            except Exception as e:
                print(f"Warning: Could not generate workflow comparison: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\nSkipping workflow comparison (only {len(all_simulation_data)} workflow found)")

    except Exception as e:
        print(f"Error processing simulation data: {e}")
        exit(1)

