#!/usr/bin/env python3
"""
Get Time and Size per Event

This script extracts time and size per event metrics from condor producer documents.
It focuses solely on calculating:
- Time per event: ChirpCMSSW_cmsRunXXX_Elapsed / ChirpCMSSW_cmsRunXXX_Events
- Size per event: ChirpCMSSW_cmsRunXXX_WriteBytes / ChirpCMSSW_cmsRunXXX_Events (in KB)

Where XXX is the last CMSSW run number in the workflow.

Note: For cmsRun1, ChirpCMSSW_cmsRun2_Events is used instead of
ChirpCMSSW_cmsRun1_Events (because cmsRun1_Events doesn't consider filter efficiency).

Only Production and Processing jobs are included. Jobs that have been internally
restarted are warned about and skipped.
"""

import json
import sys
import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict


def load_elasticsearch_data(json_filepath: str) -> List[Dict[str, Any]]:
    """
    Load and extract hits from Elasticsearch response JSON.

    Args:
        json_filepath: Path to JSON file containing Elasticsearch results

    Returns:
        List of hit documents
    """
    with open(json_filepath, 'r') as f:
        data = json.load(f)

    # Extract jobs from Elasticsearch response structure
    hits = []
    if 'responses' in data:
        for response in data['responses']:
            if 'hits' in response and 'hits' in response['hits']:
                hits.extend(response['hits']['hits'])
    elif 'hits' in data and 'hits' in data['hits']:
        hits = data['hits']['hits']
    else:
        raise ValueError("Unexpected JSON structure. Expected Elasticsearch response format.")

    return hits


def calculate_time_and_size_per_event(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Calculate time and size per event for all CMSSW runs in a single job.

    Loops through all CMSSW runs and calculates time and size per event for each.
    Uses the number of events from the last step for all calculations.

    Args:
        data: Job data dictionary from Elasticsearch document

    Returns:
        List of dictionaries, one for each CMSSW run, with 'time_per_event_sec' and
        'size_per_event_kb'. Returns empty list if calculation not possible.
    """
    # Get number of CMSSW runs
    num_cmssw_steps = data.get('ChirpCMSSWRuns', 0)
    if num_cmssw_steps == 0:
        return []

    # Determine the last run number
    last_run = num_cmssw_steps

    # Get events count: use cmsRun2_Events for cmsRun1 case, otherwise use the last run's Events
    if last_run == 1:
        # For single-run jobs, use cmsRun2_Events if available
        events_field = 'ChirpCMSSW_cmsRun2_Events'
        events = data.get(events_field, None)
        if events is None or events <= 0:
            # If cmsRun2_Events not available, we can't calculate accurately
            return []
    else:
        # For multi-run jobs, use the last run's Events for all calculations
        events_field = f'ChirpCMSSW_cmsRun{last_run}_Events'
        events = data.get(events_field, 0)
        if events <= 0:
            return []

    results = []

    # Loop through all CMSSW runs
    for run_num in range(1, num_cmssw_steps + 1):
        # Get elapsed time for this run
        elapsed_field = f'ChirpCMSSW_cmsRun{run_num}_Elapsed'
        elapsed_sec = data.get(elapsed_field, 0.0)
        if elapsed_sec <= 0:
            continue

        # Get write bytes for this run
        write_bytes_field = f'ChirpCMSSW_cmsRun{run_num}_WriteBytes'
        write_bytes = data.get(write_bytes_field, 0)
        if write_bytes < 0:
            continue

        # Calculate time per event (seconds) using events from last step
        time_per_event_sec = elapsed_sec / events

        # Calculate size per event (KB) using events from last step
        size_per_event_kb = (write_bytes / 1024.0) / events

        results.append({
            'time_per_event_sec': time_per_event_sec,
            'size_per_event_kb': size_per_event_kb,
            'run_number': run_num,
            'events_used': events,
            'elapsed_sec': elapsed_sec,
            'write_bytes': write_bytes,
        })

    return results


def extract_time_size_metrics(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract time and size per event metrics from condor producer documents.

    Args:
        hits: List of Elasticsearch hit documents

    Returns:
        List of dictionaries containing job metrics
    """
    results = []

    for hit in hits:
        source = hit.get('_source', {})
        metadata = source.get('metadata', {})
        producer = metadata.get('producer', 'unknown')

        # Filter: only process condor documents
        if producer != 'condor':
            continue

        data = source.get('data', {})

        # Filter: only include Production and Processing jobs
        job_type = data.get('CMS_JobType', 'Unknown')
        if job_type not in ['Production', 'Processing']:
            continue

        # Filter: skip jobs that did not exit successfully
        exit_code = data.get('ExitCode', None)
        exit_status = data.get('ExitStatus', None)
        if exit_code != 0 or exit_status != 0:
            wmagent_job_id = data.get('WMAgent_JobID', 'Unknown')
            print(
                f"WARNING: Job did not exit successfully - skipping: WMAgent_JobID={wmagent_job_id}, "
                f"ExitCode={exit_code}, ExitStatus={exit_status}",
                file=sys.stderr
            )
            continue

        # Filter: for the moment, disregard jobs that have been internally restarted
        num_shadow_starts = data.get('NumShadowStarts', 0)
        num_restarts = data.get('NumRestarts', 0)
        num_job_starts = data.get('NumJobStarts', 0)
        if num_shadow_starts > 1 or num_restarts > 0 or num_job_starts > 1:
            wmagent_job_id = data.get('WMAgent_JobID', 'Unknown')
            print(
                f"WARNING: Job internally restarted - skipping: WMAgent_JobID={wmagent_job_id}, "
                f"NumShadowStarts={num_shadow_starts}, NumRestarts={num_restarts}, "
                f"NumJobStarts={num_job_starts}",
                file=sys.stderr
            )
            continue

        # Calculate time and size per event for all runs
        run_metrics_list = calculate_time_and_size_per_event(data)
        if not run_metrics_list:
            continue

        # Add job identification to each run's metrics
        job_id = data.get('WMAgent_JobID', 'Unknown')
        task_type = data.get('WMAgent_TaskType', 'Unknown')
        for metrics in run_metrics_list:
            metrics['job_id'] = job_id
            metrics['task_type'] = task_type
            metrics['job_type'] = job_type

        results.extend(run_metrics_list)

    return results


def print_metrics(results: List[Dict[str, Any]]) -> None:
    """
    Print time and size per event metrics to stdout, grouped by cmsRun number.

    Args:
        results: List of job metrics dictionaries
    """
    if not results:
        print("No valid jobs found for time/size per event calculation.")
        return

    print("="*80)
    print("TIME AND SIZE PER EVENT METRICS")
    print("="*80)

    # Group results by run number
    results_by_run = defaultdict(list)
    for result in results:
        run_num = result['run_number']
        results_by_run[run_num].append(result)

    # Count unique jobs (not runs)
    unique_jobs = len(set(r['job_id'] for r in results))
    print(f"\nTotal Jobs Processed: {unique_jobs:,}")
    print(f"Total Run Measurements: {len(results):,}")

    # Calculate overall workflow averages (accumulated across all cmsRuns per job)
    # Group results by job_id to sum time and size per event across all runs
    results_by_job = defaultdict(list)
    for result in results:
        job_id = result['job_id']
        results_by_job[job_id].append(result)

    # Calculate total time and size per event for each job (sum across all runs)
    job_total_times = []
    job_total_sizes = []
    for job_id, job_results in results_by_job.items():
        total_time = sum(r['time_per_event_sec'] for r in job_results)
        total_size = sum(r['size_per_event_kb'] for r in job_results)
        job_total_times.append(total_time)
        job_total_sizes.append(total_size)

    # Print overall workflow statistics
    print(f"\n{'='*80}")
    print("Overall Workflow Statistics (Accumulated across all cmsRuns)")
    print(f"{'='*80}")
    print(f"\nTotal Time per Event (seconds) - Sum across all cmsRuns:")
    print(f"  Count:  {len(job_total_times):,}")
    print(f"  Mean:   {statistics.mean(job_total_times):.6f} sec/event")
    print(f"  Median: {statistics.median(job_total_times):.6f} sec/event")
    if len(job_total_times) > 1:
        print(f"  Stdev:  {statistics.stdev(job_total_times):.6f} sec/event")
    print(f"  Min:    {min(job_total_times):.6f} sec/event")
    print(f"  Max:    {max(job_total_times):.6f} sec/event")

    print(f"\nTotal Size per Event (KB) - Sum across all cmsRuns:")
    print(f"  Count:  {len(job_total_sizes):,}")
    print(f"  Mean:   {statistics.mean(job_total_sizes):.6f} KB/event")
    print(f"  Median: {statistics.median(job_total_sizes):.6f} KB/event")
    if len(job_total_sizes) > 1:
        print(f"  Stdev:  {statistics.stdev(job_total_sizes):.6f} KB/event")
    print(f"  Min:    {min(job_total_sizes):.6f} KB/event")
    print(f"  Max:    {max(job_total_sizes):.6f} KB/event")

    # Calculate and print statistics for each cmsRun
    for run_num in sorted(results_by_run.keys()):
        run_results = results_by_run[run_num]
        time_values = [r['time_per_event_sec'] for r in run_results]
        size_values = [r['size_per_event_kb'] for r in run_results]

        print(f"\n{'='*80}")
        print(f"cmsRun{run_num} Statistics (n={len(run_results):,} measurements)")
        print(f"{'='*80}")

        # Time per event statistics
        print(f"\nTime per Event (seconds):")
        print(f"  Count:  {len(time_values):,}")
        print(f"  Mean:   {statistics.mean(time_values):.6f} sec/event")
        print(f"  Median: {statistics.median(time_values):.6f} sec/event")
        if len(time_values) > 1:
            print(f"  Stdev:  {statistics.stdev(time_values):.6f} sec/event")
        print(f"  Min:    {min(time_values):.6f} sec/event")
        print(f"  Max:    {max(time_values):.6f} sec/event")

        # Size per event statistics
        print(f"\nSize per Event (KB):")
        print(f"  Count:  {len(size_values):,}")
        print(f"  Mean:   {statistics.mean(size_values):.6f} KB/event")
        print(f"  Median: {statistics.median(size_values):.6f} KB/event")
        if len(size_values) > 1:
            print(f"  Stdev:  {statistics.stdev(size_values):.6f} KB/event")
        print(f"  Min:    {min(size_values):.6f} KB/event")
        print(f"  Max:    {max(size_values):.6f} KB/event")

    print("\n" + "="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Extract time and size per event from condor producer documents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze condor documents
  python get_time_per_event.py data/const001.json
        """
    )
    parser.add_argument('json_file', help='Path to JSON file containing Elasticsearch results')

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.json_file}...")
    try:
        hits = load_elasticsearch_data(args.json_file)
        print(f"Found {len(hits)} documents")
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract metrics
    results = extract_time_size_metrics(hits)

    # Print metrics
    print_metrics(results)


if __name__ == '__main__':
    main()
