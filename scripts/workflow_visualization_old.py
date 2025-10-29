#!/usr/bin/env python3
"""
Workflow Simulation Visualization Script

This script creates comprehensive visualizations for workflow simulation results
using pandas, matplotlib, and seaborn. It processes all JSON files in a directory
and aggregates the data for comparative analysis.

Usage:
    python workflow_visualization.py <simulation_directory> [--output-dir OUTPUT_DIR]
    
Example:
    python workflow_visualization.py results/others/5tasks_fullsim/ --plots all
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import warnings

import matplotlib
# Set non-interactive backend to avoid display issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_simulation_data(file_path: str) -> Dict[str, Any]:
    """Load simulation result data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


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
            'job_count': group['job_count'],
            'input_events': group['input_events'],
            'total_execution_time': group['total_execution_time'],
            'taskset_count': len(group['tasksets']),
            'dependencies': group.get('dependencies', []),
            'file_name': file_name,
            'composition_number': simulation_data.get('metrics', {}).get('composition_number', 0)
        }
        
        # Calculate aggregated metrics from tasksets
        total_time_per_event = sum(ts['time_per_event'] for ts in group['tasksets'])
        total_memory = sum(ts['memory'] for ts in group['tasksets'])
        total_multicore = sum(ts['multicore'] for ts in group['tasksets'])
        total_size_per_event = sum(ts['size_per_event'] for ts in group['tasksets'])
        
        group_metrics.update({
            'total_time_per_event': total_time_per_event,
            'total_memory': total_memory,
            'total_multicore': total_multicore,
            'total_size_per_event': total_size_per_event,
            'avg_time_per_event': total_time_per_event / len(group['tasksets']),
            'avg_memory': total_memory / len(group['tasksets']),
            'avg_multicore': total_multicore / len(group['tasksets']),
            'avg_size_per_event': total_size_per_event / len(group['tasksets'])
        })
        
        groups.append(group_metrics)
    
    return groups


def extract_job_metrics(simulation_data: Dict[str, Any], file_name: str) -> List[Dict[str, Any]]:
    """Extract job-level metrics from simulation data."""
    jobs = []
    
    for job in simulation_data.get('simulation_result', {}).get('jobs', []):
        job_metrics = {
            'job_id': job['job_id'],
            'group_id': job['group_id'],
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
    and only keeping the extracted metrics in memory, not the full JSON data.
    
    Returns:
        tuple: (all_groups, all_jobs, first_simulation_data) - aggregated metrics and 
               first simulation data for summary plots
    """
    simulation_files = find_simulation_files(directory_path)
    all_groups = []
    all_jobs = []
    first_simulation_data = None
    files_processed = 0
    
    print(f"Found {len(simulation_files)} JSON files in directory")
    
    for file_path in simulation_files:
        try:
            print(f"  Loading and processing: {Path(file_path).name}")
            
            # Load the file
            simulation_data = load_simulation_data(file_path)
            file_name = Path(file_path).name
            
            # Extract metrics immediately (this reduces the memory footprint)
            groups = extract_group_metrics(simulation_data, file_name)
            jobs = extract_job_metrics(simulation_data, file_name)
            
            # Accumulate the extracted metrics
            all_groups.extend(groups)
            all_jobs.extend(jobs)
            
            # Keep the first simulation data for summary plots (only need one)
            if first_simulation_data is None:
                first_simulation_data = simulation_data
                first_simulation_data['_file_name'] = file_name
            
            files_processed += 1
            
            # Clear the full simulation data from memory (only keep extracted metrics)
            del simulation_data
            
        except Exception as e:
            print(f"  Warning: Failed to process {Path(file_path).name}: {e}")
            continue
    
    if files_processed == 0:
        raise ValueError(f"No valid simulation data processed from directory '{directory_path}'")
    
    print(f"Successfully processed {files_processed} simulation files")
    print(f"Extracted {len(all_groups)} groups and {len(all_jobs)} jobs")
    
    return all_groups, all_jobs, first_simulation_data


def plot_resource_utilization(groups: List[Dict[str, Any]], jobs: List[Dict[str, Any]], 
                             output_dir: str = "plots") -> None:
    """Plot resource utilization metrics for groups and jobs."""
    print(f"Plotting resource utilization for {len(groups)} groups and {len(jobs)} jobs")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrames for easier plotting
    groups_df = pd.DataFrame(groups)
    jobs_df = pd.DataFrame(jobs)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Resource Utilization Analysis", fontsize=16, fontweight='bold')
    
    # Plot 1: CPU Utilization vs Group Size
    if not groups_df.empty:
        sns.scatterplot(data=groups_df, x='taskset_count', y='total_multicore', 
                       ax=axes[0, 0], s=100, alpha=0.7)
        axes[0, 0].set_title("Total CPU Cores vs Group Size")
        axes[0, 0].set_xlabel("Number of Tasksets in Group")
        axes[0, 0].set_ylabel("Total CPU Cores")
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Memory Usage vs Group Size
    if not groups_df.empty:
        sns.scatterplot(data=groups_df, x='taskset_count', y='total_memory', 
                       ax=axes[0, 1], s=100, alpha=0.7)
        axes[0, 1].set_title("Total Memory vs Group Size")
        axes[0, 1].set_xlabel("Number of Tasksets in Group")
        axes[0, 1].set_ylabel("Total Memory (MB)")
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Job CPU Utilization Distribution
    if not jobs_df.empty:
        sns.histplot(data=jobs_df, x='cpu_utilization', bins=20, ax=axes[1, 0])
        axes[1, 0].set_title("Job CPU Utilization Distribution")
        axes[1, 0].set_xlabel("CPU Utilization Ratio")
        axes[1, 0].set_ylabel("Number of Jobs")
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: CPU Utilization vs Memory Usage (jobs)
    if not jobs_df.empty:
        sns.scatterplot(data=jobs_df, x='total_cpu_used_time', y='total_write_local_mb', 
                       ax=axes[1, 1], s=50, alpha=0.6)
        axes[1, 1].set_title("CPU Usage vs Local Write Data")
        axes[1, 1].set_xlabel("Total CPU Used Time")
        axes[1, 1].set_ylabel("Local Write Data (MB)")
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "resource_utilization.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_throughput_analysis(groups: List[Dict[str, Any]], jobs: List[Dict[str, Any]], 
                            output_dir: str = "plots") -> None:
    """Plot throughput and I/O metrics for groups and jobs."""
    print(f"Plotting throughput analysis for {len(groups)} groups and {len(jobs)} jobs")
    
    # Convert to DataFrames
    groups_df = pd.DataFrame(groups)
    jobs_df = pd.DataFrame(jobs)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Throughput and I/O Analysis", fontsize=16, fontweight='bold')
    
    # Plot 1: Group Throughput vs Group Size
    if not groups_df.empty:
        # Calculate throughput as events per second
        groups_df['throughput_eps'] = groups_df['input_events'] / groups_df['total_execution_time']
        sns.scatterplot(data=groups_df, x='taskset_count', y='throughput_eps', 
                       ax=axes[0, 0], s=100, alpha=0.7)
        axes[0, 0].set_title("Group Throughput vs Group Size")
        axes[0, 0].set_xlabel("Number of Tasksets in Group")
        axes[0, 0].set_ylabel("Events per Second")
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Job Throughput Distribution
    if not jobs_df.empty:
        sns.histplot(data=jobs_df, x='throughput_eps', bins=20, ax=axes[0, 1])
        axes[0, 1].set_title("Job Throughput Distribution")
        axes[0, 1].set_xlabel("Events per Second")
        axes[0, 1].set_ylabel("Number of Jobs")
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Data I/O Patterns
    if not jobs_df.empty:
        axes[1, 0].scatter(jobs_df['total_read_local_mb'], jobs_df['total_write_local_mb'], 
                          alpha=0.6, s=30, label='Local I/O')
        axes[1, 0].scatter(jobs_df['total_read_remote_mb'], jobs_df['total_write_remote_mb'], 
                          alpha=0.6, s=30, label='Remote I/O')
        axes[1, 0].set_title("Data I/O Patterns")
        axes[1, 0].set_xlabel("Read Data (MB)")
        axes[1, 0].set_ylabel("Write Data (MB)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Throughput vs Data Size
    if not jobs_df.empty:
        jobs_df['total_data_mb'] = (jobs_df['total_read_local_mb'] + jobs_df['total_read_remote_mb'] + 
                                   jobs_df['total_write_local_mb'] + jobs_df['total_write_remote_mb'])
        sns.scatterplot(data=jobs_df, x='total_data_mb', y='throughput_eps', 
                       ax=axes[1, 1], s=50, alpha=0.6)
        axes[1, 1].set_title("Throughput vs Total Data Size")
        axes[1, 1].set_xlabel("Total Data Size (MB)")
        axes[1, 1].set_ylabel("Events per Second")
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "throughput_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_job_scaling_analysis(groups: List[Dict[str, Any]], jobs: List[Dict[str, Any]], 
                             output_dir: str = "plots") -> None:
    """Plot job scaling analysis."""
    print(f"Plotting job scaling analysis for {len(groups)} groups and {len(jobs)} jobs")
    
    # Convert to DataFrames
    groups_df = pd.DataFrame(groups)
    jobs_df = pd.DataFrame(jobs)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Job Scaling Analysis", fontsize=16, fontweight='bold')
    
    # Plot 1: Job Count vs Group Size
    if not groups_df.empty:
        sns.scatterplot(data=groups_df, x='taskset_count', y='job_count', 
                       ax=axes[0, 0], s=100, alpha=0.7)
        axes[0, 0].set_title("Job Count vs Group Size")
        axes[0, 0].set_xlabel("Number of Tasksets in Group")
        axes[0, 0].set_ylabel("Number of Jobs")
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Batch Size Distribution
    if not jobs_df.empty:
        sns.histplot(data=jobs_df, x='batch_size', bins=20, ax=axes[0, 1])
        axes[0, 1].set_title("Job Batch Size Distribution")
        axes[0, 1].set_xlabel("Batch Size (Events)")
        axes[0, 1].set_ylabel("Number of Jobs")
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Wallclock Time vs Batch Size
    if not jobs_df.empty:
        sns.scatterplot(data=jobs_df, x='batch_size', y='wallclock_time', 
                       ax=axes[1, 0], s=50, alpha=0.6)
        axes[1, 0].set_title("Wallclock Time vs Batch Size")
        axes[1, 0].set_xlabel("Batch Size (Events)")
        axes[1, 0].set_ylabel("Wallclock Time (seconds)")
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Job Efficiency (CPU Utilization vs Throughput)
    if not jobs_df.empty:
        sns.scatterplot(data=jobs_df, x='cpu_utilization', y='throughput_eps', 
                       ax=axes[1, 1], s=50, alpha=0.6)
        axes[1, 1].set_title("Job Efficiency: CPU Utilization vs Throughput")
        axes[1, 1].set_xlabel("CPU Utilization Ratio")
        axes[1, 1].set_ylabel("Events per Second")
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "job_scaling_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_time_analysis(groups: List[Dict[str, Any]], jobs: List[Dict[str, Any]], 
                      output_dir: str = "plots") -> None:
    """Plot time-based analysis."""
    print(f"Plotting time analysis for {len(groups)} groups and {len(jobs)} jobs")
    
    # Convert to DataFrames
    groups_df = pd.DataFrame(groups)
    jobs_df = pd.DataFrame(jobs)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Time Analysis", fontsize=16, fontweight='bold')
    
    # Plot 1: Execution Time vs Group Size
    if not groups_df.empty:
        sns.scatterplot(data=groups_df, x='taskset_count', y='total_execution_time', 
                       ax=axes[0, 0], s=100, alpha=0.7)
        axes[0, 0].set_title("Total Execution Time vs Group Size")
        axes[0, 0].set_xlabel("Number of Tasksets in Group")
        axes[0, 0].set_ylabel("Total Execution Time (seconds)")
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Wallclock Time Distribution
    if not jobs_df.empty:
        sns.histplot(data=jobs_df, x='wallclock_time', bins=20, ax=axes[0, 1])
        axes[0, 1].set_title("Job Wallclock Time Distribution")
        axes[0, 1].set_xlabel("Wallclock Time (seconds)")
        axes[0, 1].set_ylabel("Number of Jobs")
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Time per Event vs Group Size
    if not groups_df.empty:
        groups_df['time_per_event'] = groups_df['total_execution_time'] / groups_df['input_events']
        sns.scatterplot(data=groups_df, x='taskset_count', y='time_per_event', 
                       ax=axes[1, 0], s=100, alpha=0.7)
        axes[1, 0].set_title("Time per Event vs Group Size")
        axes[1, 0].set_xlabel("Number of Tasksets in Group")
        axes[1, 0].set_ylabel("Time per Event (seconds)")
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Job Timeline (if we have start/end times)
    if not jobs_df.empty and 'start_time' in jobs_df.columns:
        # Sample a subset of jobs for timeline visualization
        sample_jobs = jobs_df.sample(min(50, len(jobs_df)))
        for idx, job in sample_jobs.iterrows():
            axes[1, 1].barh(job['job_id'], job['wallclock_time'], 
                           left=job['start_time'], height=0.8, alpha=0.7)
        axes[1, 1].set_title("Job Timeline (Sample)")
        axes[1, 1].set_xlabel("Time (seconds)")
        axes[1, 1].set_ylabel("Job ID")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Fallback: CPU vs Wallclock time
        if not jobs_df.empty:
            sns.scatterplot(data=jobs_df, x='wallclock_time', y='total_cpu_used_time', 
                           ax=axes[1, 1], s=50, alpha=0.6)
            axes[1, 1].set_title("CPU Time vs Wallclock Time")
            axes[1, 1].set_xlabel("Wallclock Time (seconds)")
            axes[1, 1].set_ylabel("CPU Time (seconds)")
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_workflow_summary(simulation_data: Dict[str, Any], output_dir: str = "plots") -> None:
    """Plot workflow-level summary metrics."""
    print("Plotting workflow summary metrics")
    
    metrics = simulation_data.get('metrics', {})
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Workflow Summary Metrics", fontsize=16, fontweight='bold')
    
    # Plot 1: Resource Utilization Overview
    cpu_util = metrics.get('cpu_utilization', 0)
    memory_occ = metrics.get('memory_occupancy', 0)
    
    categories = ['CPU Utilization', 'Memory Occupancy']
    values = [cpu_util, memory_occ]
    colors = ['skyblue', 'lightcoral']
    
    bars = axes[0, 0].bar(categories, values, color=colors, alpha=0.7)
    axes[0, 0].set_title("Resource Utilization Overview")
    axes[0, 0].set_ylabel("Utilization Ratio")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 2: Data Transfer Overview
    read_local = metrics.get('total_read_local_mb_per_event', 0)
    write_local = metrics.get('total_write_local_mb_per_event', 0)
    write_remote = metrics.get('total_write_remote_mb_per_event', 0)
    
    data_categories = ['Read Local', 'Write Local', 'Write Remote']
    data_values = [read_local, write_local, write_remote]
    
    axes[0, 1].bar(data_categories, data_values, color=['lightgreen', 'orange', 'purple'], alpha=0.7)
    axes[0, 1].set_title("Data Transfer per Event")
    axes[0, 1].set_ylabel("Data Size (MB)")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Performance Metrics
    wall_time_per_event = metrics.get('wall_time_per_event', 0)
    cpu_time_per_event = metrics.get('cpu_time_per_event', 0)
    event_throughput = metrics.get('event_throughput', 0)
    
    perf_categories = ['Wall Time/Event', 'CPU Time/Event', 'Event Throughput']
    perf_values = [wall_time_per_event, cpu_time_per_event, event_throughput]
    
    axes[1, 0].bar(perf_categories, perf_values, color=['red', 'blue', 'green'], alpha=0.7)
    axes[1, 0].set_title("Performance Metrics")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Resource Requirements
    cpu_cores_per_event = metrics.get('cpu_cores_per_event', 0)
    memory_mb_per_event = metrics.get('memory_mb_per_event', 0)
    
    resource_categories = ['CPU Cores/Event', 'Memory MB/Event']
    resource_values = [cpu_cores_per_event, memory_mb_per_event]
    
    axes[1, 1].bar(resource_categories, resource_values, color=['cyan', 'magenta'], alpha=0.7)
    axes[1, 1].set_title("Resource Requirements per Event")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "workflow_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_table(groups: List[Dict[str, Any]], jobs: List[Dict[str, Any]], 
                        simulation_data: Dict[str, Any], output_dir: str = "plots") -> None:
    """Create a summary table of key metrics."""
    print("Creating summary table")
    
    metrics = simulation_data.get('metrics', {})
    
    # Create summary data
    summary_data = {
        'Metric': [
            'Total Events',
            'Total Tasksets',
            'Total Groups',
            'Total Jobs',
            'Total Wall Time (hours)',
            'Total Turnaround Time (hours)',
            'Event Throughput (events/sec)',
            'CPU Utilization',
            'Memory Occupancy',
            'Success Rate',
            'Total CPU Used Time (hours)',
            'Total CPU Allocated Time (hours)',
            'Total Write Local (GB)',
            'Total Write Remote (GB)',
            'Total Read Local (GB)',
            'Total Network Transfer (GB)'
        ],
        'Value': [
            f"{metrics.get('total_events', 0):,}",
            metrics.get('total_tasksets', 0),
            metrics.get('total_groups', 0),
            metrics.get('total_jobs', 0),
            f"{metrics.get('total_wall_time', 0) / 3600:.2f}",
            f"{metrics.get('total_turnaround_time', 0) / 3600:.2f}",
            f"{metrics.get('event_throughput', 0):.6f}",
            f"{metrics.get('cpu_utilization', 0):.3f}",
            f"{metrics.get('memory_occupancy', 0):.3f}",
            f"{metrics.get('success_rate', 0):.3f}",
            f"{metrics.get('total_cpu_used_time', 0) / 3600:.2f}",
            f"{metrics.get('total_cpu_allocated_time', 0) / 3600:.2f}",
            f"{metrics.get('total_write_local_mb', 0) / 1024:.2f}",
            f"{metrics.get('total_write_remote_mb', 0) / 1024:.2f}",
            f"{metrics.get('total_read_local_mb', 0) / 1024:.2f}",
            f"{metrics.get('total_network_transfer_mb', 0) / 1024:.2f}"
        ]
    }
    
    # Create DataFrame and save as CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, "workflow_summary.csv"), index=False)
    
    # Also create a visual table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_df.values,
                    colLabels=summary_df.columns,
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title("Workflow Simulation Summary", fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(output_dir, "workflow_summary_table.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function to run the visualization script."""
    parser = argparse.ArgumentParser(
        description='Create visualizations for workflow simulation results using pandas/matplotlib/seaborn'
    )
    parser.add_argument('simulation_directory', type=str, 
                       help='Path to directory containing simulation result JSON files')
    parser.add_argument('--output-dir', type=str, default='visualizations', 
                       help='Output directory for plots (default: visualizations)')
    parser.add_argument('--plots', nargs='+', 
                       choices=['resource', 'throughput', 'scaling', 'time', 'summary', 'all'],
                       default=['all'],
                       help='Which plots to generate (default: all)')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.simulation_directory):
        print(f"Error: Directory '{args.simulation_directory}' not found")
        return 1
    
    if not os.path.isdir(args.simulation_directory):
        print(f"Error: '{args.simulation_directory}' is not a directory")
        return 1
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Processing simulation data from directory: {args.simulation_directory}")
    try:
        # Process files incrementally to minimize memory usage
        groups, jobs, first_simulation_data = process_simulation_directory(args.simulation_directory)
    except Exception as e:
        print(f"Error processing simulation data: {e}")
        return 1
    
    # Generate requested plots
    plots_to_generate = args.plots if 'all' not in args.plots else ['resource', 'throughput', 'scaling', 'time', 'summary']
    
    if 'resource' in plots_to_generate:
        plot_resource_utilization(groups, jobs, args.output_dir)
    
    if 'throughput' in plots_to_generate:
        plot_throughput_analysis(groups, jobs, args.output_dir)
    
    if 'scaling' in plots_to_generate:
        plot_job_scaling_analysis(groups, jobs, args.output_dir)
    
    if 'time' in plots_to_generate:
        plot_time_analysis(groups, jobs, args.output_dir)
    
    if 'summary' in plots_to_generate:
        # For summary, use the first simulation data as representative
        if first_simulation_data is not None:
            plot_workflow_summary(first_simulation_data, args.output_dir)
            create_summary_table(groups, jobs, first_simulation_data, args.output_dir)
        else:
            print("Warning: No simulation data available for summary plots")
    
    print(f"\nVisualization complete! Plots saved to: {args.output_dir}")
    print("Generated plots:")
    for plot_type in plots_to_generate:
        if plot_type == 'summary':
            print(f"  - workflow_summary.png")
            print(f"  - workflow_summary_table.png")
            print(f"  - workflow_summary.csv")
        else:
            print(f"  - {plot_type}_analysis.png")
    
    return 0


if __name__ == "__main__":
    exit(main())
