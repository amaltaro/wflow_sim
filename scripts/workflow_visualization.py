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


def plot_workflow_comparison(all_simulation_data: List[Dict],
                             sim_groups: List[Dict],
                             jobs: List[Dict],
                             output_dir: str = "plots",
                             custom_labels: List[str] = None,
                             use_aggregate_metrics: bool = False):
    """Create a comprehensive comparison of workflow constructions.

    This function creates multiple visualizations to help identify trade-offs
    between different workflow constructions.
    """
    print(f"Creating comprehensive workflow construction comparison for {len(all_simulation_data)} constructions")

    # Extract metrics for comparison directly from simulation data
    num_groups = []
    event_throughputs = []
    total_cpu_times = []
    write_remote_pevt = []
    total_write_remote = []
    read_remote_pevt = []
    write_local_pevt = []
    read_local_pevt = []
    group_combinations = []
    construction_metrics = []  # Build this for text output

    for i, sim_data in enumerate(all_simulation_data):
        # sim_data now contains metrics directly (not nested under 'metrics')
        file_name = sim_data.get('_file_name', f'simulation_{i}')

        # Get groups for this simulation from the groups parameter
        sim_groups = [g for g in groups if g.get('file_name') == file_name]

        # Extract basic metrics
        num_groups.append(len(sim_groups))
        event_throughputs.append(sim_data.get('event_throughput', 0.0))
        total_cpu_times.append(sim_data.get('total_cpu_allocated_time', 0.0))
        write_remote_pevt.append(sim_data.get('total_write_remote_mb_per_event', 0.0))
        total_write_remote.append(sim_data.get('total_write_remote_mb', 0.0))
        read_remote_pevt.append(sim_data.get('total_read_remote_mb_per_event', 0.0))
        write_local_pevt.append(sim_data.get('total_write_local_mb_per_event', 0.0))
        read_local_pevt.append(sim_data.get('total_read_local_mb_per_event', 0.0))

        # Build group combinations
        group_ids = [g['group_id'] for g in sim_groups]
        group_combinations.append(" + ".join(group_ids))

        # Build construction metrics for text output
        construction_metric = {
            'groups': group_ids,
            'num_groups': len(sim_groups),
            'event_throughput': sim_data.get('event_throughput', 0.0),
            'total_cpu_time': sim_data.get('total_cpu_allocated_time', 0.0),
            'write_remote_per_event_mb': sim_data.get('total_write_remote_mb_per_event', 0.0),
            'total_write_remote_mb': sim_data.get('total_write_remote_mb', 0.0),
            'read_remote_per_event_mb': sim_data.get('total_read_remote_mb_per_event', 0.0),
            'write_local_per_event_mb': sim_data.get('total_write_local_mb_per_event', 0.0),
            'read_local_per_event_mb': sim_data.get('total_read_local_mb_per_event', 0.0),
            'total_read_remote_mb': sim_data.get('total_read_remote_mb', 0.0),
            'total_write_local_mb': sim_data.get('total_write_local_mb', 0.0),
            'total_wallclock_time': sim_data.get('total_wall_time', 0.0),
            'total_memory_mb': sim_data.get('total_memory_used_mb', 0.0),
            'total_network_transfer_mb': sim_data.get('total_network_transfer_mb', 0.0),
            'network_transfer_per_event_mb': sim_data.get('network_transfer_mb_per_event', 0.0),
            'cpu_utilization': sim_data.get('cpu_utilization', 0.0),
            'memory_occupancy': sim_data.get('memory_occupancy', 0.0),
            'group_details': []  # Will be populated below
        }

        # Build group_details for text output
        for group in sim_groups:
            # Get taskset IDs for this group (from the extracted group metrics)
            tasks = group.get('group_taskset_count', 0)  # This is the count, not the actual IDs

            # Calculate events per task (using first job of this group)
            group_jobs = [j for j in jobs if j['group_id'] == group['group_id'] and j.get('file_name') == file_name]
            events_per_task = group_jobs[0]['batch_size'] if group_jobs else group.get('group_input_events', 0)

            # Calculate per-event data metrics from first job
            first_job = group_jobs[0] if group_jobs else None
            total_events = group.get('group_input_events', 0) * group.get('group_job_count', 1)

            read_local_per_event = 0.0
            read_remote_per_event = 0.0
            write_local_per_event = 0.0
            write_remote_per_event = 0.0

            if first_job and total_events > 0:
                read_local_per_event = first_job.get('total_read_local_mb', 0.0) / events_per_task
                read_remote_per_event = first_job.get('total_read_remote_mb', 0.0) / events_per_task
                write_local_per_event = first_job.get('total_write_local_mb', 0.0) / events_per_task
                write_remote_per_event = first_job.get('total_write_remote_mb', 0.0) / events_per_task

            # Calculate CPU seconds for the group (using extracted metrics)
            cpu_seconds = group.get('group_time_per_event', 0.0) * events_per_task

            construction_metric['group_details'].append({
                'group_id': group['group_id'],
                'tasks': [f"taskset_{i+1}" for i in range(tasks)],  # Generate task IDs
                'events_per_task': events_per_task,
                'cpu_seconds': cpu_seconds,
                'read_local_per_event_mb': read_local_per_event,
                'read_remote_per_event_mb': read_remote_per_event,
                'write_local_per_event_mb': write_local_per_event,
                'write_remote_per_event_mb': write_remote_per_event,
                'total_events': total_events
            })

        construction_metrics.append(construction_metric)

    # Convert lists to numpy arrays for numerical operations
    num_groups = np.array(num_groups)
    event_throughputs = np.array(event_throughputs)
    total_cpu_times = np.array(total_cpu_times)
    write_remote_pevt = np.array(write_remote_pevt)
    total_write_remote = np.array(total_write_remote)
    read_remote_pevt = np.array(read_remote_pevt)
    write_local_pevt = np.array(write_local_pevt)
    read_local_pevt = np.array(read_local_pevt)

    # Create a figure with multiple subplots - now with fixed, professional proportions
    if len(construction_metrics) <= 2:
        fig = plt.figure(figsize=(12, 16))
    else:
        fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1])  # Equal height ratios for all rows

    # Always use short labels for plots ("Const 1", "Const 2", etc.)
    # custom_labels are only used in the text output file
    construction_labels = [f"Const {i+1}" for i, _ in enumerate(all_simulation_data)]

    # Define consistent colors for each metric type
    colors = {
        'Local Read': '#1f77b4',    # Blue
        'Remote Read': '#ff7f0e',   # Orange
        'Local Write': '#2ca02c',   # Green
        'Remote Write': '#d62728'   # Red
    }

    # 1. Data Volume Analysis Per Event (with Local Read)
    ax3 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(all_simulation_data))
    width = 0.2
    ax3.bar(x - 1.5*width, read_local_pevt, width, label='Local Read', color=colors['Local Read'])
    ax3.bar(x - 0.5*width, read_remote_pevt, width, label='Remote Read', color=colors['Remote Read'])
    ax3.bar(x + 0.5*width, write_local_pevt, width, label='Local Write', color=colors['Local Write'])
    ax3.bar(x + 1.5*width, write_remote_pevt, width, label='Remote Write', color=colors['Remote Write'])
    ax3.set_xlabel("Workflow Construction")
    ax3.set_ylabel("Data Volume per Event (MB)")
    ax3.set_title("Data Volume Analysis Per Event")
    ax3.set_xticks(x)
    ax3.set_xticklabels(construction_labels, rotation=45)
    ax3.legend()
    ax3.grid(True)

    # 2. Data Flow Analysis (Updated to use per-event metrics)
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(all_simulation_data))
    width = 0.25
    ax2.bar(x - width, read_remote_pevt, width, label='Remote Read', color=colors['Remote Read'])
    ax2.bar(x, write_local_pevt, width, label='Local Write', color=colors['Local Write'])
    ax2.bar(x + width, write_remote_pevt, width, label='Remote Write', color=colors['Remote Write'])
    ax2.set_xlabel("Workflow Construction")
    ax2.set_ylabel("Data Volume per Event (MB)")
    ax2.set_title("Data Volume Analysis Per Event")
    ax2.set_xticks(x)
    ax2.set_xticklabels(construction_labels, rotation=45)
    ax2.legend()
    ax2.grid(True)

    # 3. Total Data Volume Analysis (Stacked Bar)
    ax10 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(all_simulation_data))
    width = 0.6
    bottom = np.zeros(len(all_simulation_data))

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
    scatter = ax1.scatter(event_throughputs, write_remote_pevt,
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
    max_throughput = np.max(event_throughputs)
    if max_throughput > 0:
        ax1.set_xlim(left=0, right=max_throughput * 1.1)
    else:
        # If all throughputs are 0, set a small range to avoid the warning
        ax1.set_xlim(left=0, right=1.0)
    # set y-axis to start at 0
    ax1.set_ylim(bottom=0)

    # 5. Network Transfer Analysis
    ax7 = fig.add_subplot(gs[2, 0])
    network_transfer = []
    for sim_data in all_simulation_data:
        # sim_data now contains metrics directly
        # Network transfer = remote read + remote write (only remote operations)
        # Use pre-calculated field if available, otherwise calculate from per-event metrics
        transfer = sim_data.get("network_transfer_mb_per_event")
        if transfer is None:
            # Fallback: calculate from read and write remote per event
            transfer = sim_data.get("total_read_remote_mb_per_event", 0.0) + sim_data.get("total_write_remote_mb_per_event", 0.0)
        network_transfer.append(transfer)

    ax7.bar(range(len(all_simulation_data)), network_transfer)
    ax7.set_xlabel("Workflow Construction")
    ax7.set_ylabel("Network Transfer per Event (MB)")
    ax7.set_title("Network Transfer Analysis")
    ax7.set_xticks(range(len(all_simulation_data)))
    ax7.set_xticklabels(construction_labels, rotation=45)
    ax7.grid(True)

    # 6. CPU Utilization Analysis
    ax4 = fig.add_subplot(gs[2, 1])
    cpu_utilization = []
    cpu_std = []  # Store standard deviations

    # Use metrics data from simulation
    for sim_data in all_simulation_data:
        # sim_data now contains metrics directly
        cpu_util = sim_data.get("cpu_utilization", 0.0)
        cpu_utilization.append(cpu_util)
        cpu_std.append(0)  # No std dev available from aggregated metrics

    # Create bar plot with error bars (only if we have std data)
    x = range(len(all_simulation_data))
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

    # Use metrics data from simulation
    for sim_data in all_simulation_data:
        # sim_data now contains metrics directly
        mem_util = sim_data.get("memory_occupancy", 0.0)
        memory_utilization.append(mem_util)
        memory_std.append(0)  # No std dev available from aggregated metrics

    # Create bar plot with error bars (only if we have std data)
    x = range(len(all_simulation_data))
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
    for i, sim_data in enumerate(all_simulation_data):
        file_name = sim_data.get('_file_name', f'simulation_{i}')
        # Get groups for this simulation from the groups parameter
        sim_groups = [g for g in groups if g.get('file_name') == file_name]
        events = [g.get('group_input_events', 0) * g.get('group_job_count', 1) for g in sim_groups]
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
            f.write(f"    Local Read Data: {metrics['read_local_per_event_mb']:.3f} MB/event\n")
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
                f.write(f"        Local Read: {group['read_local_per_event_mb']:.3f} MB/event\n")
                f.write(f"        Remote Read: {group['read_remote_per_event_mb']:.3f} MB/event\n")
                f.write(f"        Local Write: {group['write_local_per_event_mb']:.3f} MB/event\n")
                f.write(f"        Remote Write: {group['write_remote_per_event_mb']:.3f} MB/event\n")
            f.write("\n")


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
            workflow_metrics = {'_file_name': file_name}
            workflow_metrics.update(simulation_data.get('metrics', {}))
            all_simulation_data.append(workflow_metrics)

            files_processed += 1

        except Exception as e:
            print(f"  Warning: Failed to process {Path(file_path).name}: {e}")
            continue

    if files_processed == 0:
        raise ValueError(f"No valid simulation data processed from directory '{directory_path}'")

    print(f"Successfully processed {files_processed} simulation files")
    print(f"Extracted {len(all_groups)} groups and {len(all_jobs)} jobs")
    if all_simulation_data:
        print(f"Sample workflow metrics: {pformat(all_simulation_data[0])}")
        print(f"Sample group metrics: {pformat(all_groups[0])}")
        print(f"Sample job metrics: {pformat(all_jobs[0])}")

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
                # Call the comparison function
                plot_workflow_comparison(
                    all_simulation_data=all_simulation_data,
                    sim_groups=groups,
                    jobs=jobs,
                    output_dir=args.output_dir
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

