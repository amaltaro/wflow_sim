import argparse
import json
import os
from typing import List, Dict, Any
from collections import defaultdict
from math import ceil

import matplotlib
# Set non-interactive backend to avoid display issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path









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
            with open(file_path, 'r') as f:
                simulation_data = json.load(f)
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
        # Process files incrementally to minimize memory usage
        groups, jobs, first_simulation_data = process_simulation_directory(args.simulation_directory)
    except Exception as e:
        print(f"Error processing simulation data: {e}")
        exit(1)
    