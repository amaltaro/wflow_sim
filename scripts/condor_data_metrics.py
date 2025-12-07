#!/usr/bin/env python3
"""
Condor Data Metrics

This script extracts high-level statistics from Elasticsearch job data,
focusing specifically on condor producer documents.

Metrics provided:
- Total number of documents in the file
- Number of condor documents (filtered to Production and Processing jobs only)
- Number of jobs per job type (CMS_JobType)
- Number of jobs per task type (WMAgent_TaskType)
- Total wallclock time: Sum of WallClockHr (hours) or CommittedTime (seconds)
- Workflow turnaround time: Maximum CompletionDate minus minimum JobStartDate
  across all jobs
- Total CPU time: Sum of CpuTimeHr (hours) or ChirpCMSSWTotalCPU (seconds)
- Total CPU used time: Same as total CPU time (CPU time actually used)
- Total CPU allocated time: Sum of CoreHr (CPU hours allocated)
- Total read (local and remote): Calculated from ChirpCMSSWReadBytes and
  ChirpCMSSW_cmsRun1_ReadBytes
- Total write (local and remote): Calculated from ChirpCMSSWWriteBytes
- CPU utilization: Ratio of used CPU time to allocated CPU time
- Memory utilization: Ratio of used memory (MemoryUsage) to allocated memory OriginalMemory
- Event throughput: Total events divided by total wallclock time (events per second)
- Wallclock time per event: Total wallclock time divided by total events
- CPU time per event: Total CPU time divided by total events

Note: Only jobs with CMS_JobType of 'Production' or 'Processing' are included.
"""

import json
import sys
import argparse
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


def _normalize_timestamp(timestamp: Any) -> Optional[float]:
    """
    Normalize timestamp to seconds (handle both seconds and milliseconds).

    Args:
        timestamp: Unix timestamp (may be in seconds or milliseconds)

    Returns:
        Timestamp in seconds, or None if invalid
    """
    if timestamp is None:
        return None
    try:
        ts = float(timestamp)
        # If timestamp is very large (> year 2100 in seconds), assume milliseconds
        if ts > 4102444800:  # Year 2100 in seconds
            return ts / 1000.0
        return ts
    except (ValueError, TypeError):
        return None


def _extract_job_metrics(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract metrics from a single condor job document.

    Args:
        data: Job data dictionary from Elasticsearch document

    Returns:
        Dictionary with extracted metrics, or None if job should be skipped
    """
    metrics = {}

    # Wallclock time: WallClockHr (hours) or CommittedTime (seconds)
    wallclock_hr = data.get('WallClockHr', 0.0)
    committed_time = data.get('CommittedTime', 0.0)
    if wallclock_hr > 0:
        metrics['wallclock_time_sec'] = wallclock_hr * 3600.0
    elif committed_time > 0:
        metrics['wallclock_time_sec'] = committed_time
    else:
        metrics['wallclock_time_sec'] = 0.0

    # CPU time used: ChirpCMSSWTotalCPU (seconds) or CpuTimeHr (hours)
    chirp_cpu = data.get('ChirpCMSSWTotalCPU', 0.0)  # seconds
    cpu_time_hr = data.get('CpuTimeHr', 0.0)  # hours
    if chirp_cpu > 0:
        metrics['cpu_time_used_sec'] = chirp_cpu
    elif cpu_time_hr > 0:
        metrics['cpu_time_used_sec'] = cpu_time_hr * 3600.0
    else:
        metrics['cpu_time_used_sec'] = 0.0

    # CPU allocated time: CoreHr (CPU hours)
    core_hr = data.get('CoreHr', 0.0)  # CPU hours
    metrics['cpu_time_allocated_sec'] = core_hr * 3600.0

    # Memory used: Peak memory used by the job is given by MemoryUsage
    metrics['memory_used_mb'] = data.get('MemoryUsage', 0.0)
    # Memory allocated: OriginalMemory
    metrics['memory_allocated_mb'] = data.get('OriginalMemory', 0.0)
    # avoid memory overcommitment in these measurements
    if metrics['memory_used_mb'] > metrics['memory_allocated_mb']:
        metrics['memory_used_mb'] = metrics['memory_allocated_mb']

    # Read bytes: Calculate local and remote reads
    read_total_bytes = data.get('ChirpCMSSWReadBytes', 0)
    desired_cms_dataset = data.get('DESIRED_CMSDataset')
    cmsrun1_read_bytes = data.get('ChirpCMSSW_cmsRun1_ReadBytes', 0)

    # Local read is always total - cmsRun1
    metrics['read_local_mb'] = max(0.0, (read_total_bytes - cmsrun1_read_bytes) / (1024.0 * 1024.0))

    # Remote read is cmsRun1 only if DESIRED_CMSDataset is not None
    if desired_cms_dataset is not None:
        metrics['read_remote_mb'] = cmsrun1_read_bytes / (1024.0 * 1024.0)
    else:
        metrics['read_remote_mb'] = 0.0

    # Write bytes: Calculate local and remote writes
    write_total_bytes = data.get('ChirpCMSSWWriteBytes', 0)
    # For now, all writes are considered local (write_remote_mb not accurately calculated)
    metrics['write_local_mb'] = write_total_bytes / (1024.0 * 1024.0)
    metrics['write_remote_mb'] = 0.0

    # Events processed: Use output events from last taskset if available
    num_cmssw_steps = data.get('ChirpCMSSWRuns', 0)
    if num_cmssw_steps > 1:
        last_step_field = f'ChirpCMSSW_cmsRun{num_cmssw_steps}_Events'
        last_step_events = data.get(last_step_field, None)
        if last_step_events is not None:
            metrics['events_processed'] = last_step_events
        else:
            metrics['events_processed'] = data.get('ChirpCMSSWEvents', 0)
    else:
        metrics['events_processed'] = data.get('ChirpCMSSWEvents', 0)

    # Job timing for workflow turnaround time calculation
    metrics['job_start_date'] = _normalize_timestamp(data.get('JobStartDate'))
    metrics['completion_date'] = _normalize_timestamp(data.get('CompletionDate'))

    return metrics


def extract_condor_stats(hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract statistics from condor producer documents.

    Args:
        hits: List of Elasticsearch hit documents

    Returns:
        Dictionary containing statistics
    """
    total_docs = len(hits)
    condor_docs = 0
    job_type_counts = defaultdict(int)
    task_type_counts = defaultdict(int)

    # Metrics accumulators
    total_wallclock_time_sec = 0.0
    total_cpu_time_used_sec = 0.0
    total_cpu_time_allocated_sec = 0.0
    total_memory_used_mb = 0.0
    total_memory_allocated_mb = 0.0
    total_read_local_mb = 0.0
    total_read_remote_mb = 0.0
    total_write_local_mb = 0.0
    total_write_remote_mb = 0.0
    total_events = 0

    # For workflow turnaround time
    job_start_dates = []
    completion_dates = []

    for hit in hits:
        source = hit.get('_source', {})
        metadata = source.get('metadata', {})
        producer = metadata.get('producer', 'unknown')

        # Only process condor documents
        if producer != 'condor':
            continue

        data = source.get('data', {})

        # Check for internally restarted jobs
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

        # Extract job type (CMS_JobType)
        job_type = data.get('CMS_JobType', 'Unknown')

        # Filter: only include Production and Processing jobs
        if job_type not in ['Production', 'Processing']:
            continue

        condor_docs += 1
        job_type_counts[job_type] += 1

        # Extract task type (WMAgent_TaskType)
        task_type = data.get('WMAgent_TaskType', 'Unknown')
        task_type_counts[task_type] += 1

        # Extract metrics
        job_metrics = _extract_job_metrics(data)
        if job_metrics is None:
            continue

        # Accumulate metrics
        total_wallclock_time_sec += job_metrics['wallclock_time_sec']
        total_cpu_time_used_sec += job_metrics['cpu_time_used_sec']
        total_cpu_time_allocated_sec += job_metrics['cpu_time_allocated_sec']
        total_memory_used_mb += job_metrics['memory_used_mb']
        total_memory_allocated_mb += job_metrics['memory_allocated_mb']
        total_read_local_mb += job_metrics['read_local_mb']
        total_read_remote_mb += job_metrics['read_remote_mb']
        total_write_local_mb += job_metrics['write_local_mb']
        total_write_remote_mb += job_metrics['write_remote_mb']
        total_events += job_metrics['events_processed']

        # Collect timestamps for workflow turnaround time
        if job_metrics['job_start_date'] is not None:
            job_start_dates.append(job_metrics['job_start_date'])
        if job_metrics['completion_date'] is not None:
            completion_dates.append(job_metrics['completion_date'])

    # Calculate workflow turnaround time
    workflow_turnaround_time_sec = None
    if job_start_dates and completion_dates:
        min_start = min(job_start_dates)
        max_completion = max(completion_dates)
        workflow_turnaround_time_sec = max_completion - min_start

    # Calculate derived metrics
    cpu_utilization = None
    if total_cpu_time_allocated_sec > 0:
        cpu_utilization = total_cpu_time_used_sec / total_cpu_time_allocated_sec

    memory_utilization = None
    if total_memory_allocated_mb > 0:
        memory_utilization = total_memory_used_mb / total_memory_allocated_mb

    event_throughput = None
    if total_wallclock_time_sec > 0:
        event_throughput = total_events / total_wallclock_time_sec

    wallclock_time_per_event = None
    if total_events > 0:
        wallclock_time_per_event = total_wallclock_time_sec / total_events

    cpu_time_per_event = None
    if total_events > 0:
        cpu_time_per_event = total_cpu_time_used_sec / total_events

    return {
        'total_docs': total_docs,
        'condor_docs': condor_docs,
        'job_type_counts': dict(job_type_counts),
        'task_type_counts': dict(task_type_counts),
        'total_wallclock_time_sec': total_wallclock_time_sec,
        'workflow_turnaround_time_sec': workflow_turnaround_time_sec,
        'total_cpu_time_used_sec': total_cpu_time_used_sec,
        'total_cpu_time_allocated_sec': total_cpu_time_allocated_sec,
        'total_read_local_mb': total_read_local_mb,
        'total_read_remote_mb': total_read_remote_mb,
        'total_write_local_mb': total_write_local_mb,
        'total_write_remote_mb': total_write_remote_mb,
        'total_memory_used_mb': total_memory_used_mb,
        'total_memory_allocated_mb': total_memory_allocated_mb,
        'total_events': total_events,
        'cpu_utilization': cpu_utilization,
        'memory_utilization': memory_utilization,
        'event_throughput': event_throughput,
        'wallclock_time_per_event': wallclock_time_per_event,
        'cpu_time_per_event': cpu_time_per_event,
    }


def print_stats(stats: Dict[str, Any]) -> None:
    """
    Print statistics in a formatted way.

    Args:
        stats: Dictionary containing statistics
    """
    print("\n" + "="*80)
    print("CONDOR DATA METRICS")
    print("="*80)

    # Document counts
    print(f"\nDocument Counts:")
    print(f"  Total Documents: {stats['total_docs']:,}")
    print(f"  Condor Documents (Production/Processing only): {stats['condor_docs']:,}")

    # Jobs per job type
    job_type_counts = stats['job_type_counts']
    if job_type_counts:
        print(f"\nJobs per Job Type (CMS_JobType):")
        for job_type, count in sorted(job_type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {job_type}: {count:,} jobs")
    else:
        print(f"\nJobs per Job Type (CMS_JobType): No condor documents found")

    # Jobs per task type
    task_type_counts = stats['task_type_counts']
    if task_type_counts:
        print(f"\nJobs per Task Type (WMAgent_TaskType):")
        for task_type, count in sorted(task_type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {task_type}: {count:,} jobs")
    else:
        print(f"\nJobs per Task Type (WMAgent_TaskType): No condor documents found")

    # Time metrics
    print(f"\nTime Metrics:")
    total_wallclock_hr = stats['total_wallclock_time_sec'] / 3600.0
    print(f"  Total Wallclock Time: {stats['total_wallclock_time_sec']:,.2f} sec ({total_wallclock_hr:,.2f} hours)")

    if stats['workflow_turnaround_time_sec'] is not None:
        workflow_turnaround_hr = stats['workflow_turnaround_time_sec'] / 3600.0
        print(f"  Workflow Turnaround Time: {stats['workflow_turnaround_time_sec']:,.2f} sec ({workflow_turnaround_hr:,.2f} hours)")
        print(f"    (Maximum job completion time - Minimum job start time)")
    else:
        print(f"  Workflow Turnaround Time: N/A (insufficient timing data)")

    # CPU metrics
    print(f"\nCPU Metrics:")
    total_cpu_used_hr = stats['total_cpu_time_used_sec'] / 3600.0
    total_cpu_allocated_hr = stats['total_cpu_time_allocated_sec'] / 3600.0
    print(f"  Total CPU Time: {stats['total_cpu_time_used_sec']:,.2f} sec ({total_cpu_used_hr:,.2f} hours)")
    print(f"  Total CPU Used Time: {stats['total_cpu_time_used_sec']:,.2f} sec ({total_cpu_used_hr:,.2f} hours)")
    print(f"  Total CPU Allocated Time: {stats['total_cpu_time_allocated_sec']:,.2f} sec ({total_cpu_allocated_hr:,.2f} hours)")

    if stats['cpu_utilization'] is not None:
        print(f"  CPU Utilization: {stats['cpu_utilization']:.4f} ({stats['cpu_utilization']*100:.2f}%)")
        print(f"    (Ratio of used CPU time to allocated CPU time)")
    else:
        print(f"  CPU Utilization: N/A (no allocated CPU time data)")

    # Memory metrics
    print(f"\nMemory Metrics:")
    total_memory_used_gb = stats['total_memory_used_mb'] / 1024.0
    total_memory_allocated_gb = stats['total_memory_allocated_mb'] / 1024.0
    print(f"  Total Memory (Peak Used): {stats['total_memory_used_mb']:,.2f} MB ({total_memory_used_gb:,.2f} GB)")
    print(f"  Total Memory (Allocated): {stats['total_memory_allocated_mb']:,.2f} MB ({total_memory_allocated_gb:,.2f} GB)")

    if stats['memory_utilization'] is not None:
        print(f"  Memory Utilization: {stats['memory_utilization']:.4f} ({stats['memory_utilization']*100:.2f}%)")
        print(f"    (Ratio of **peak** used memory to allocated memory)")
    else:
        print(f"  Memory Utilization: N/A (no allocated memory data)")

    # I/O metrics
    print(f"\nI/O Metrics:")
    total_read_mb = stats['total_read_local_mb'] + stats['total_read_remote_mb']
    total_write_mb = stats['total_write_local_mb'] + stats['total_write_remote_mb']
    total_read_gb = total_read_mb / 1024.0
    total_write_gb = total_write_mb / 1024.0
    print(f"  Total Read (Local): {stats['total_read_local_mb']:,.2f} MB ({stats['total_read_local_mb']/1024.0:,.2f} GB)")
    print(f"  Total Read (Remote): {stats['total_read_remote_mb']:,.2f} MB ({stats['total_read_remote_mb']/1024.0:,.2f} GB)")
    print(f"  Total Read (Local + Remote): {total_read_mb:,.2f} MB ({total_read_gb:,.2f} GB)")
    print(f"  Total Write (Local): {stats['total_write_local_mb']:,.2f} MB ({stats['total_write_local_mb']/1024.0:,.2f} GB)")
    print(f"  Total Write (Remote): {stats['total_write_remote_mb']:,.2f} MB ({stats['total_write_remote_mb']/1024.0:,.2f} GB)")
    print(f"  Total Write (Local + Remote): {total_write_mb:,.2f} MB ({total_write_gb:,.2f} GB)")

    # Event metrics
    print(f"\nEvent Metrics:")
    print(f"  Total Events Processed: {stats['total_events']:,}")

    if stats['event_throughput'] is not None:
        print(f"  Event Throughput: {stats['event_throughput']:.6f} events/sec")
        print(f"    (Total events / Total wallclock time)")
    else:
        print(f"  Event Throughput: N/A (no wallclock time data)")

    if stats['wallclock_time_per_event'] is not None:
        print(f"  Wallclock Time per Event: {stats['wallclock_time_per_event']:.6f} sec/event")
        print(f"    (Total wallclock time / Total events)")
    else:
        print(f"  Wallclock Time per Event: N/A (no events processed)")

    if stats['cpu_time_per_event'] is not None:
        print(f"  CPU Time per Event: {stats['cpu_time_per_event']:.6f} sec/event")
        print(f"    (Total CPU time / Total events)")
    else:
        print(f"  CPU Time per Event: N/A (no events processed)")

    print("\n" + "="*80)


def save_stats_to_json(stats: Dict[str, Any], output_file: str, document_name: str) -> None:
    """
    Save statistics to a JSON file.

    Args:
        stats: Dictionary containing statistics
        output_file: Path to output JSON file
        document_name: Name of the input document
    """
    # Create a clean dictionary for JSON output
    output_data = {
        'document_name': document_name,
        'document_counts': {
            'total_docs': stats['total_docs'],
            'condor_docs': stats['condor_docs'],
            'job_type_counts': stats['job_type_counts'],
            'task_type_counts': stats['task_type_counts'],
        },
        'time_metrics': {
            'total_wallclock_time_sec': stats['total_wallclock_time_sec'],
            'total_wallclock_time_hours': stats['total_wallclock_time_sec'] / 3600.0,
            'workflow_turnaround_time_sec': stats['workflow_turnaround_time_sec'],
            'workflow_turnaround_time_hours': (
                stats['workflow_turnaround_time_sec'] / 3600.0
                if stats['workflow_turnaround_time_sec'] is not None
                else None
            ),
        },
        'cpu_metrics': {
            'total_cpu_time_sec': stats['total_cpu_time_used_sec'],
            'total_cpu_time_hours': stats['total_cpu_time_used_sec'] / 3600.0,
            'total_cpu_used_time_sec': stats['total_cpu_time_used_sec'],
            'total_cpu_used_time_hours': stats['total_cpu_time_used_sec'] / 3600.0,
            'total_cpu_allocated_time_sec': stats['total_cpu_time_allocated_sec'],
            'total_cpu_allocated_time_hours': stats['total_cpu_time_allocated_sec'] / 3600.0,
            'cpu_utilization': stats['cpu_utilization'],
            'cpu_utilization_percent': (
                stats['cpu_utilization'] * 100.0
                if stats['cpu_utilization'] is not None
                else None
            ),
        },
        'memory_metrics': {
            'total_memory_used_mb': stats['total_memory_used_mb'],
            'total_memory_used_gb': stats['total_memory_used_mb'] / 1024.0,
            'total_memory_allocated_mb': stats['total_memory_allocated_mb'],
            'total_memory_allocated_gb': stats['total_memory_allocated_mb'] / 1024.0,
            'memory_utilization': stats['memory_utilization'],
            'memory_utilization_percent': (
                stats['memory_utilization'] * 100.0
                if stats['memory_utilization'] is not None
                else None
            ),
        },
        'io_metrics': {
            'total_read_local_mb': stats['total_read_local_mb'],
            'total_read_local_gb': stats['total_read_local_mb'] / 1024.0,
            'total_read_remote_mb': stats['total_read_remote_mb'],
            'total_read_remote_gb': stats['total_read_remote_mb'] / 1024.0,
            'total_read_mb': stats['total_read_local_mb'] + stats['total_read_remote_mb'],
            'total_read_gb': (stats['total_read_local_mb'] + stats['total_read_remote_mb']) / 1024.0,
            'total_write_local_mb': stats['total_write_local_mb'],
            'total_write_local_gb': stats['total_write_local_mb'] / 1024.0,
            'total_write_remote_mb': stats['total_write_remote_mb'],
            'total_write_remote_gb': stats['total_write_remote_mb'] / 1024.0,
            'total_write_mb': stats['total_write_local_mb'] + stats['total_write_remote_mb'],
            'total_write_gb': (stats['total_write_local_mb'] + stats['total_write_remote_mb']) / 1024.0,
        },
        'event_metrics': {
            'total_events': stats['total_events'],
            'event_throughput_events_per_sec': stats['event_throughput'],
            'wallclock_time_per_event_sec': stats['wallclock_time_per_event'],
            'cpu_time_per_event_sec': stats['cpu_time_per_event'],
        },
    }

    # Write to JSON file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nMetrics saved to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Extract high-level statistics from condor producer documents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze condor documents
  python condor_data_metrics.py data/const001.json

  # Analyze and save metrics to JSON file
  python condor_data_metrics.py data/const001.json --output metrics.json
        """
    )
    parser.add_argument('json_file', help='Path to JSON file containing Elasticsearch results')
    parser.add_argument(
        '--output', '-o',
        dest='output_file',
        help='Path to output JSON file for calculated metrics'
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.json_file}...")
    try:
        hits = load_elasticsearch_data(args.json_file)
        print(f"Found {len(hits)} documents")
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract statistics
    stats = extract_condor_stats(hits)

    # Print statistics
    print_stats(stats)

    # Save to JSON file if specified
    if args.output_file:
        try:
            # Extract document name from input file path
            document_name = Path(args.json_file).name
            save_stats_to_json(stats, args.output_file, document_name)
        except Exception as e:
            print(f"Error saving metrics to JSON: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == '__main__':
    main()

