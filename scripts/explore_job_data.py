#!/usr/bin/env python3
"""
Job Data Explorer

This script explores grid job information extracted from Elasticsearch,
extracting metrics similar to those simulated in the workflow simulator.

The script handles two document types:
1. condor: Condor job accounting data
2. wmarchive: WMAgent archive data

Metrics extracted:
- Number of events processed
- Number of jobs per task
- Total job turnaround time (and payload time without overhead)
- Total job CPU time
- Data volume transferred over network
- Data volume written to local disk
- Data volume written to remote storage
- Number of cores used
- Amount of memory allocated
"""

import json
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class JobMetrics:
    """Extracted metrics for a single job."""
    job_id: str
    producer: str
    task_name: Optional[str] = None  # WMAgent_TaskType
    job_type: Optional[str] = None  # CMS_JobType
    
    # Events
    events_processed: int = 0
    
    # Time metrics
    turnaround_time_sec: float = 0.0  # Total job turnaround time
    payload_time_sec: float = 0.0  # Payload execution time (without overhead)
    time_per_event_sec: float = 0.0  # Seconds per event processed
    
    # CPU metrics
    cpu_time_sec: float = 0.0  # Total CPU time used
    
    # Network transfer
    network_transfer_bytes: int = 0  # Bytes sent + received
    
    # Disk I/O
    write_total_bytes: int = 0  # Total bytes written (local + remote)
    read_total_bytes: int = 0  # Total bytes read (local + remote)
    disk_usage_kb: int = 0  # Local disk usage in kilobytes
    
    # CMSSW steps
    num_cmssw_steps: int = 0  # Number of CMSSW steps executed
    
    # Throughput
    event_rate: float = 0.0  # Events per second
    
    # Resource allocation
    cores_requested: int = 0  # OriginalCpus (cores requested)
    memory_requested_mb: float = 0.0  # OriginalMemory (memory requested in MB)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class JobDataExplorer:
    """Explorer for grid job data from Elasticsearch."""
    
    def __init__(self, json_filepath: str, producer_filter: Optional[str] = None):
        """
        Initialize the explorer.
        
        Args:
            json_filepath: Path to JSON file containing Elasticsearch results
            producer_filter: Optional filter for producer type ('condor', 'wmarchive', or None for both)
        """
        self.json_filepath = Path(json_filepath)
        self.producer_filter = producer_filter
        self.jobs: List[JobMetrics] = []
        self.condor_jobs: List[JobMetrics] = []
        self.wmarchive_jobs: List[JobMetrics] = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.WARNING,
            format='%(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_data(self) -> None:
        """Load and parse the JSON file."""
        print(f"Loading data from {self.json_filepath}...")
        with open(self.json_filepath, 'r') as f:
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
        
        print(f"Found {len(hits)} job documents")
        
        # Process each job document
        for hit in hits:
            job_metrics = self._extract_job_metrics(hit)
            if job_metrics:
                # Apply producer filter if specified
                if self.producer_filter and job_metrics.producer != self.producer_filter:
                    continue
                
                self.jobs.append(job_metrics)
                if job_metrics.producer == 'condor':
                    self.condor_jobs.append(job_metrics)
                elif job_metrics.producer == 'wmarchive':
                    self.wmarchive_jobs.append(job_metrics)
        
        producer_info = f" (filtered to {self.producer_filter})" if self.producer_filter else ""
        print(f"Successfully extracted metrics from {len(self.jobs)} jobs{producer_info}")
        if not self.producer_filter or self.producer_filter == 'condor':
            print(f"  - Condor jobs: {len(self.condor_jobs)}")
        if not self.producer_filter or self.producer_filter == 'wmarchive':
            print(f"  - WMA archive jobs: {len(self.wmarchive_jobs)}")
    
    def _extract_job_metrics(self, hit: Dict[str, Any]) -> Optional[JobMetrics]:
        """
        Extract metrics from a single job document.
        
        Args:
            hit: Elasticsearch hit document
            
        Returns:
            JobMetrics object or None if extraction fails
        """
        source = hit.get('_source', {})
        data = source.get('data', {})
        metadata = source.get('metadata', {})
        
        producer = metadata.get('producer', 'unknown')
        job_id = hit.get('_id', 'unknown')
        
        metrics = JobMetrics(job_id=job_id, producer=producer)
        metrics.metadata = {
            'index': hit.get('_index', 'unknown'),
            'cluster_id': data.get('ClusterId'),
        }
        
        # Extract job type for filtering
        if producer == 'condor':
            metrics.job_type = data.get('CMS_JobType')
        elif producer == 'wmarchive':
            metrics.job_type = data.get('meta_data', {}).get('jobtype')
        
        # Filter: only include Production and Processing jobs
        if metrics.job_type not in ['Production', 'Processing']:
            return None
        
        if producer == 'condor':
            self._extract_condor_metrics(data, metrics)
        elif producer == 'wmarchive':
            self._extract_wmarchive_metrics(data, metrics)
        else:
            print(f"Warning: Unknown producer '{producer}' for job {job_id}")
            return None
        
        return metrics
    
    def _extract_condor_metrics(self, data: Dict[str, Any], metrics: JobMetrics) -> None:
        """
        Extract metrics from condor producer document.
        
        Field hierarchy:
        - Events: data.ChirpCMSSWEvents (total events processed)
        - Turnaround time: data.CommittedTime (seconds) - total job time
        - Payload time: data.ChirpCMSSWElapsed (seconds) - CMSSW execution time
        - Time per event: data.TimePerEvent (seconds per event)
        - CPU time: data.ChirpCMSSWTotalCPU (seconds) or data.CpuTimeHr (hours)
        - Network: data.BytesSent + data.BytesRecvd
        - Total writes: data.ChirpCMSSWWriteBytes (total written, local + remote)
        - Total reads: data.ChirpCMSSWReadBytes (total read, local + remote)
        - Local disk usage: data.DiskUsage (kilobytes)
        - CMSSW steps: data.ChirpCMSSWRuns (number of CMSSW steps)
        - Event rate: data.EventRate (events per second)
        - Cores requested: data.OriginalCpus
        - Memory requested: data.OriginalMemory (MB)
        - Task name: data.WMAgent_TaskType
        """
        # Events processed
        metrics.events_processed = data.get('ChirpCMSSWEvents', 0)
        
        # Turnaround time (total job time including overhead)
        metrics.turnaround_time_sec = data.get('CommittedTime', 0.0)
        
        # Payload time (CMSSW execution time, without Condor overhead)
        metrics.payload_time_sec = data.get('ChirpCMSSWElapsed', 0.0)
        
        # Time per event
        metrics.time_per_event_sec = data.get('TimePerEvent', 0.0)
        
        # CPU time - prefer ChirpCMSSWTotalCPU (more accurate), fallback to CpuTimeHr
        chirp_cpu = data.get('ChirpCMSSWTotalCPU', 0.0)  # seconds
        cpu_time_hr = data.get('CpuTimeHr', 0.0)  # hours
        if chirp_cpu > 0:
            metrics.cpu_time_sec = chirp_cpu
        elif cpu_time_hr > 0:
            metrics.cpu_time_sec = cpu_time_hr * 3600.0
        
        # Network transfer
        bytes_sent = data.get('BytesSent', 0)
        bytes_recvd = data.get('BytesRecvd', 0)
        metrics.network_transfer_bytes = bytes_sent + bytes_recvd
        
        # Disk I/O - total reads and writes (includes both local and remote)
        metrics.write_total_bytes = data.get('ChirpCMSSWWriteBytes', 0)
        metrics.read_total_bytes = data.get('ChirpCMSSWReadBytes', 0)
        
        # Local disk usage (in kilobytes)
        metrics.disk_usage_kb = data.get('DiskUsage', 0)
        
        # CMSSW steps
        metrics.num_cmssw_steps = data.get('ChirpCMSSWRuns', 0)
        
        # Event rate (events per second)
        metrics.event_rate = data.get('EventRate', 0.0)
        
        # Resource allocation - use requested values
        metrics.cores_requested = data.get('OriginalCpus', 0)
        metrics.memory_requested_mb = data.get('OriginalMemory', 0.0)
        
        # Task identification - use WMAgent_TaskType
        metrics.task_name = data.get('WMAgent_TaskType', 'Unknown')
        
        # Check for job failures
        exit_status = data.get('ExitStatus', 0)
        exit_code = data.get('ExitCode', 0)
        
        if exit_status != 0 or exit_code != 0:
            wmagent_task_type = data.get('WMAgent_TaskType', 'Unknown')
            wmagent_job_id = data.get('WMAgent_JobID', 'Unknown')
            self.logger.warning(
                f"Job failed: WMAgent_TaskType={wmagent_task_type}, "
                f"WMAgent_JobID={wmagent_job_id}, "
                f"ExitStatus={exit_status}, ExitCode={exit_code}"
            )
    
    def _extract_wmarchive_metrics(self, data: Dict[str, Any], metrics: JobMetrics) -> None:
        """
        Extract metrics from wmarchive producer document.
        
        Field hierarchy:
        - Events: sum of data.steps[].input[].events (input events processed)
        - Turnaround time: data.WMTiming.WMTotalWallClockTime (seconds)
        - Payload time: sum of data.steps[].WMCMSSWSubprocess.wallClockTime
        - CPU time: sum of data.steps[].WMCMSSWSubprocess.userTime + sysTime
        - Network: Not directly available in wmarchive
        - Disk writes: sum of data.steps[].output[].size (bytes)
        - Disk reads: Not directly available (input file sizes not in standard format)
        - Cores: Not directly available in wmarchive
        - Memory: data.steps[].performance.cmssw.ApplicationMemory.PeakValueRss (MB)
        """
        steps = data.get('steps', [])
        
        # Events processed - sum input events from all steps
        total_events = 0
        for step in steps:
            inputs = step.get('input', [])
            for inp in inputs:
                if isinstance(inp, dict):
                    total_events += inp.get('events', 0)
        metrics.events_processed = total_events
        
        # Turnaround time
        wm_timing = data.get('WMTiming', {})
        metrics.turnaround_time_sec = wm_timing.get('WMTotalWallClockTime', 0.0)
        
        # Payload time - sum of CMSSW wall clock time from all steps
        payload_time = 0.0
        cpu_time = 0.0
        max_memory = 0.0
        
        for step in steps:
            wm_cmssw = step.get('WMCMSSWSubprocess', {})
            if wm_cmssw:
                payload_time += wm_cmssw.get('wallClockTime', 0.0)
                cpu_time += wm_cmssw.get('userTime', 0.0) + wm_cmssw.get('sysTime', 0.0)
            
            # Memory - peak RSS across all steps
            performance = step.get('performance', {})
            cmssw_perf = performance.get('cmssw', {})
            app_memory = cmssw_perf.get('ApplicationMemory', {})
            peak_rss = app_memory.get('PeakValueRss', 0.0)
            if peak_rss > max_memory:
                max_memory = peak_rss
        
        metrics.payload_time_sec = payload_time
        metrics.cpu_time_sec = cpu_time
        
        # Calculate time per event if we have events
        if metrics.events_processed > 0 and payload_time > 0:
            metrics.time_per_event_sec = payload_time / metrics.events_processed
        
        # Disk writes - sum output file sizes
        total_write_bytes = 0
        for step in steps:
            outputs = step.get('output', [])
            for out in outputs:
                if isinstance(out, dict):
                    total_write_bytes += out.get('size', 0)
        metrics.write_total_bytes = total_write_bytes
        
        # Number of CMSSW steps
        metrics.num_cmssw_steps = len([s for s in steps if s.get('WMCMSSWSubprocess')])
        
        # Calculate event rate if we have events and turnaround time
        if metrics.events_processed > 0 and metrics.turnaround_time_sec > 0:
            metrics.event_rate = metrics.events_processed / metrics.turnaround_time_sec
        
        # Memory requested (use peak RSS as requested memory)
        metrics.memory_requested_mb = max_memory
        
        # Network, cores, and local disk usage not directly available in wmarchive
        # Reads not directly available (input file sizes not in standard format)
        
        # Task identification - wmarchive doesn't have WMAgent_TaskType
        # Use jobtype as fallback
        meta_data = data.get('meta_data', {})
        metrics.task_name = meta_data.get('jobtype', 'Unknown')
        
        # TODO: Add failure check for wmarchive jobs
        # Need to investigate which fields indicate job failure in wmarchive documents
        # Possible fields to check: meta_data.jobstate, steps[].errors, etc.
    
    def print_summary(self) -> None:
        """Print summary statistics."""
        if not self.jobs:
            print("No jobs to summarize")
            return
        
        print("\n" + "="*80)
        print("JOB METRICS SUMMARY")
        if self.producer_filter:
            print(f"Producer Filter: {self.producer_filter.upper()}")
        print("="*80)
        
        # Overall statistics
        print(f"\nTotal Jobs: {len(self.jobs)}")
        if not self.producer_filter:
            print(f"  - Condor: {len(self.condor_jobs)}")
            print(f"  - WMA Archive: {len(self.wmarchive_jobs)}")
        elif self.producer_filter == 'condor':
            print(f"  - Condor: {len(self.condor_jobs)}")
        elif self.producer_filter == 'wmarchive':
            print(f"  - WMA Archive: {len(self.wmarchive_jobs)}")
        
        # Jobs per job type (CMS_JobType)
        job_type_counts = defaultdict(int)
        for job in self.jobs:
            job_type = job.job_type or 'Unknown'
            job_type_counts[job_type] += 1
        
        print(f"\nJobs per Job Type (CMS_JobType):")
        for job_type, count in sorted(job_type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {job_type}: {count} jobs")
        
        # Jobs per task type (WMAgent_TaskType)
        task_type_counts = defaultdict(int)
        for job in self.jobs:
            task_name = job.task_name or 'Unknown'
            task_type_counts[task_name] += 1
        
        print(f"\nJobs per Task Type (WMAgent_TaskType):")
        for task_type, count in sorted(task_type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {task_type}: {count} jobs")
        
        # Events
        total_events = sum(j.events_processed for j in self.jobs)
        print(f"\nTotal Events Processed: {total_events:,}")
        if self.jobs:
            avg_events = total_events / len(self.jobs)
            print(f"Average Events per Job: {avg_events:,.1f}")
        
        # Time metrics
        total_turnaround = sum(j.turnaround_time_sec for j in self.jobs)
        total_payload = sum(j.payload_time_sec for j in self.jobs)
        print(f"\nTime Metrics:")
        print(f"  Total Turnaround Time: {total_turnaround:,.1f} seconds ({total_turnaround/3600:.2f} hours)")
        print(f"  Total Payload Time: {total_payload:,.1f} seconds ({total_payload/3600:.2f} hours)")
        if total_turnaround > 0:
            overhead_pct = ((total_turnaround - total_payload) / total_turnaround) * 100
            print(f"  Overhead: {total_turnaround - total_payload:,.1f} seconds ({overhead_pct:.1f}%)")
        
        # CPU time
        total_cpu = sum(j.cpu_time_sec for j in self.jobs)
        print(f"\nCPU Metrics:")
        print(f"  Total CPU Time: {total_cpu:,.1f} seconds ({total_cpu/3600:.2f} hours)")
        if total_payload > 0:
            cpu_efficiency = (total_cpu / total_payload) * 100
            print(f"  CPU Efficiency: {cpu_efficiency:.1f}% (CPU time / Payload time)")
        
        # Network transfer
        total_network = sum(j.network_transfer_bytes for j in self.jobs)
        print(f"\nNetwork Transfer:")
        print(f"  Total Network Transfer: {total_network:,} bytes ({total_network/(1024**3):.2f} GB)")
        if not self.producer_filter:
            if self.condor_jobs:
                condor_network = sum(j.network_transfer_bytes for j in self.condor_jobs)
                print(f"    - Condor: {condor_network:,} bytes ({condor_network/(1024**3):.2f} GB)")
            if self.wmarchive_jobs:
                wma_network = sum(j.network_transfer_bytes for j in self.wmarchive_jobs)
                print(f"    - WMA Archive: {wma_network:,} bytes ({wma_network/(1024**3):.2f} GB)")
                if wma_network == 0:
                    print(f"      (Note: Network metrics not available in wmarchive documents)")
        elif self.producer_filter == 'wmarchive':
            print(f"    (Note: Network metrics not available in wmarchive documents)")
        
        # Disk I/O
        total_write_total = sum(j.write_total_bytes for j in self.jobs)
        total_read_total = sum(j.read_total_bytes for j in self.jobs)
        total_disk_usage = sum(j.disk_usage_kb for j in self.jobs)
        
        print(f"\nDisk I/O:")
        print(f"  Total Write (all): {total_write_total:,} bytes ({total_write_total/(1024**3):.2f} GB)")
        print(f"  Total Read (all): {total_read_total:,} bytes ({total_read_total/(1024**3):.2f} GB)")
        print(f"  Total Local Disk Usage: {total_disk_usage:,} KB ({total_disk_usage/(1024**2):.2f} GB)")
        if self.producer_filter != 'wmarchive' and self.condor_jobs:
            print(f"    (Note: ChirpCMSSWWriteBytes and ChirpCMSSWReadBytes include both local and remote)")
        
        # CMSSW steps
        total_steps = sum(j.num_cmssw_steps for j in self.jobs)
        if total_steps > 0:
            print(f"\nCMSSW Steps:")
            print(f"  Total CMSSW Steps: {total_steps:,}")
            if self.jobs:
                avg_steps = total_steps / len(self.jobs)
                print(f"  Average Steps per Job: {avg_steps:.1f}")
        
        # Event rate
        total_event_rate = sum(j.event_rate for j in self.jobs)
        if total_event_rate > 0:
            print(f"\nEvent Throughput:")
            if self.jobs:
                avg_event_rate = total_event_rate / len(self.jobs)
                print(f"  Average Event Rate: {avg_event_rate:.4f} events/second")
        
        # Time per event
        total_time_per_event = sum(j.time_per_event_sec for j in self.jobs if j.time_per_event_sec > 0)
        if total_time_per_event > 0:
            jobs_with_tpe = sum(1 for j in self.jobs if j.time_per_event_sec > 0)
            if jobs_with_tpe > 0:
                avg_time_per_event = total_time_per_event / jobs_with_tpe
                print(f"  Average Time per Event: {avg_time_per_event:.4f} seconds/event")
        
        # Resource allocation
        total_cores = sum(j.cores_requested for j in self.jobs)
        total_memory = sum(j.memory_requested_mb for j in self.jobs)
        
        print(f"\nResource Allocation (Requested):")
        print(f"  Total Cores Requested: {total_cores:,}")
        if self.jobs:
            avg_cores = total_cores / len(self.jobs)
            print(f"  Average Cores per Job: {avg_cores:.1f}")
        print(f"  Total Memory Requested: {total_memory:,.1f} MB ({total_memory/1024:.2f} GB)")
        if self.jobs:
            avg_memory = total_memory / len(self.jobs)
            print(f"  Average Memory per Job: {avg_memory:.1f} MB")
        
        print("\n" + "="*80)
    
    def print_field_hierarchy(self) -> None:
        """Print field hierarchy documentation for each metric."""
        print("\n" + "="*80)
        print("FIELD HIERARCHY DOCUMENTATION")
        if self.producer_filter:
            print(f"Producer Filter: {self.producer_filter.upper()}")
        print("="*80)
        
        if self.producer_filter == 'wmarchive':
            # Skip condor documentation if filtering to wmarchive
            pass
        else:
            print("\n## Condor Producer Fields")
        print("\n- **Events Processed**: `data.ChirpCMSSWEvents`")
        print("- **Turnaround Time**: `data.CommittedTime` (seconds)")
        print("- **Payload Time**: `data.ChirpCMSSWElapsed` (seconds, CMSSW execution)")
        print("- **Time per Event**: `data.TimePerEvent` (seconds per event)")
        print("- **CPU Time**: `data.ChirpCMSSWTotalCPU` (seconds) or `data.CpuTimeHr` (hours)")
        print("- **Network Transfer**: `data.BytesSent + data.BytesRecvd`")
        print("- **Total Writes**: `data.ChirpCMSSWWriteBytes` (includes both local and remote)")
        print("- **Total Reads**: `data.ChirpCMSSWReadBytes` (includes both local and remote)")
        print("- **Local Disk Usage**: `data.DiskUsage` (kilobytes)")
        print("- **CMSSW Steps**: `data.ChirpCMSSWRuns` (number of CMSSW steps executed)")
        print("- **Event Rate**: `data.EventRate` (events per second)")
        print("- **Cores Requested**: `data.OriginalCpus` (alternative: `GLIDEIN_Cpus` or `JobCpus`)")
        print("- **Memory Requested**: `data.OriginalMemory` (MB, alternative: `GLIDEIN_Memory` or `MemoryProvisioned`)")
        print("- **Task Name**: `data.WMAgent_TaskType`")
        print("- **Job Type**: `data.CMS_JobType` (e.g., Production, Processing, Merge)")
        
        if self.producer_filter == 'condor':
            # Skip wmarchive documentation if filtering to condor
            pass
        else:
            print("\n## WMA Archive Producer Fields")
        print("\n- **Events Processed**: `sum(data.steps[].input[].events)`")
        print("- **Turnaround Time**: `data.WMTiming.WMTotalWallClockTime` (seconds)")
        print("- **Payload Time**: `sum(data.steps[].WMCMSSWSubprocess.wallClockTime)`")
        print("- **CPU Time**: `sum(data.steps[].WMCMSSWSubprocess.userTime + sysTime)`")
        print("- **Network Transfer**: Not available")
        print("- **Remote Writes**: `sum(data.steps[].output[].size)` (bytes)")
        print("- **Remote Reads**: Not directly available")
        print("- **Cores**: Not available")
        print("- **Memory**: `max(data.steps[].performance.cmssw.ApplicationMemory.PeakValueRss)` (MB)")
        print("- **Task Name**: `data.meta_data.jobtype`")
        
        print("\n## Notes")
        print("\n- **Job Filtering**: Only Production and Processing jobs are included")
        print("- **Disk I/O**: ChirpCMSSWWriteBytes and ChirpCMSSWReadBytes include both local and remote I/O")
        print("- **Local Disk**: DiskUsage provides local disk usage in kilobytes")
        print("- **Event Rate**: May or may not include job overhead (field documentation unclear)")
        print("- **Time per Event**: Field may need verification for accuracy")
        print("- **Network metrics**: Not available in wmarchive documents")
        print("- **Core count**: Not available in wmarchive documents")
        print("- **Condor data**: Provides more comprehensive metrics")
        print("- **WMA archive data**: Provides detailed step-by-step breakdown")
        
        print("\n" + "="*80)
    
    def export_to_json(self, output_file: str) -> None:
        """Export extracted metrics to JSON file."""
        output_data = {
            'summary': {
                'total_jobs': len(self.jobs),
                'condor_jobs': len(self.condor_jobs),
                'wmarchive_jobs': len(self.wmarchive_jobs),
            },
            'jobs': [
                {
                    'job_id': j.job_id,
                    'producer': j.producer,
                    'task_name': j.task_name,
                    'job_type': j.job_type,
                    'events_processed': j.events_processed,
                    'turnaround_time_sec': j.turnaround_time_sec,
                    'payload_time_sec': j.payload_time_sec,
                    'time_per_event_sec': j.time_per_event_sec,
                    'cpu_time_sec': j.cpu_time_sec,
                    'network_transfer_bytes': j.network_transfer_bytes,
                    'write_total_bytes': j.write_total_bytes,
                    'read_total_bytes': j.read_total_bytes,
                    'disk_usage_kb': j.disk_usage_kb,
                    'num_cmssw_steps': j.num_cmssw_steps,
                    'event_rate': j.event_rate,
                    'cores_requested': j.cores_requested,
                    'memory_requested_mb': j.memory_requested_mb,
                    'metadata': j.metadata,
                }
                for j in self.jobs
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nExported metrics to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Explore grid job information extracted from Elasticsearch',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all producers
  python explore_job_data.py data/const001.json

  # Analyze only condor jobs
  python explore_job_data.py data/const001.json --producer condor

  # Analyze only wmarchive jobs
  python explore_job_data.py data/const001.json --producer wmarchive

  # Export to JSON file
  python explore_job_data.py data/const001.json --producer condor results/job_metrics.json
        """
    )
    parser.add_argument('json_file', help='Path to JSON file containing Elasticsearch results')
    parser.add_argument('output_file', nargs='?', help='Optional output JSON file for exported metrics')
    parser.add_argument(
        '--producer',
        choices=['condor', 'wmarchive'],
        help='Filter to specific producer type (default: analyze both)'
    )
    
    args = parser.parse_args()
    
    explorer = JobDataExplorer(args.json_file, producer_filter=args.producer)
    explorer.load_data()
    explorer.print_summary()
    explorer.print_field_hierarchy()
    
    if args.output_file:
        explorer.export_to_json(args.output_file)


if __name__ == '__main__':
    main()

