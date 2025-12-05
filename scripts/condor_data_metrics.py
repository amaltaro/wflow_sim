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

Note: Only jobs with CMS_JobType of 'Production' or 'Processing' are included.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any
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

    return {
        'total_docs': total_docs,
        'condor_docs': condor_docs,
        'job_type_counts': dict(job_type_counts),
        'task_type_counts': dict(task_type_counts),
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

    print("\n" + "="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Extract high-level statistics from condor producer documents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze condor documents
  python condor_data_metrics.py data/const001.json
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

    # Extract statistics
    stats = extract_condor_stats(hits)

    # Print statistics
    print_stats(stats)


if __name__ == '__main__':
    main()

