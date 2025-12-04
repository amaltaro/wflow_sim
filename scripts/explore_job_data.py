#!/usr/bin/env python3
"""
Job Data Explorer

This script explores grid job information extracted from Elasticsearch,
extracting metrics similar to those simulated in the workflow simulator.

The script handles two document types:
1. condor: Condor job accounting data
2. wmarchive: WMArchive data

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
    turnaround_time_sec: float = 0.0  # Total job turnaround time (CommittedTime)
    payload_time_sec: float = 0.0  # Payload execution time (without overhead, ChirpCMSSWElapsed)
    time_per_event_sec: float = 0.0  # Seconds per event processed

    # Job timing (for workflow turnaround time calculation)
    job_start_date: Optional[float] = None  # JobStartDate (Unix timestamp)
    completion_date: Optional[float] = None  # CompletionDate (Unix timestamp)

    # CPU metrics
    cpu_time_sec: float = 0.0  # Total CPU time used

    # Network transfer
    network_transfer_bytes: int = 0  # Placeholder - not used (sandbox/log files disregarded)

    # Disk I/O
    write_local_mb: float = 0.0  # Local write megabytes
    write_remote_mb: float = 0.0  # Remote write megabytes
    read_local_mb: float = 0.0  # Local read megabytes
    read_remote_mb: float = 0.0  # Remote read megabytes
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

    @staticmethod
    def _normalize_timestamp(timestamp: Optional[float]) -> Optional[float]:
        """
        Normalize timestamp to seconds.

        Detects if timestamp is in milliseconds and converts to seconds.
        Uses heuristic: if timestamp > 1e12, assume it's in milliseconds.
        (1e12 seconds = year 31688, which is unreasonable)
        (1e12 milliseconds = year 2001, which is reasonable)

        Args:
            timestamp: Timestamp value (may be in seconds or milliseconds)

        Returns:
            Timestamp in seconds, or None if input is None
        """
        if timestamp is None:
            return None
        # If timestamp is very large (> 1e12), assume it's in milliseconds
        if timestamp > 1e12:
            return timestamp / 1000.0
        return timestamp

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

        Field hierarchy (based on convert_to_json.py from cms-htcondor-es):
        - Events: data.ChirpCMSSW_cmsRun{N}_Events (output events from last taskset, where N = ChirpCMSSWRuns)
                  Falls back to data.ChirpCMSSWEvents if last step field not available
                  This accounts for filter efficiency < 1.0 (fewer events out than in)
        - Turnaround time: data.CommittedTime (seconds) - total job time including overhead
        - Payload time: data.ChirpCMSSWElapsed (seconds) - CMSSW execution time without overhead
        - Time per event: data.TimePerEvent (CPU seconds per event)
                          Calculated as: 1.0 / EventRate, where EventRate = ChirpCMSSWEvents / (CoreHr * 3600.0)
                          This is based on CPU time (CoreHr = CPU hours)
        - Job start date: data.JobStartDate (Unix timestamp, may be in seconds or milliseconds)
        - Completion date: data.CompletionDate (Unix timestamp, may be in seconds or milliseconds)
        - CPU time: data.ChirpCMSSWTotalCPU (seconds) or data.CpuTimeHr (hours)
                    CoreHr is also available (CPU hours, used for EventRate calculation)
        - Network transfer (physics data): read_remote_mb + write_remote_mb
          (Set to 0.0 for now since write_remote_mb is not accurately calculated)
          Note: BytesSent + BytesRecvd (sandbox/log files) are disregarded
        - Local writes (MB): ChirpCMSSWWriteBytes / (1024 * 1024)
        - Remote writes (MB): 0.0 (not all written data is staged to shared storage)
        - Local reads (MB): Always (ChirpCMSSWReadBytes - ChirpCMSSW_cmsRun1_ReadBytes) / (1024 * 1024).
                            Note that ChirpCMSSW_cmsRun1_ReadBytes might report a value even if no data is read.
        - Remote reads (MB): If DESIRED_CMSDataset is not None (string),
                            then ChirpCMSSW_cmsRun1_ReadBytes / (1024 * 1024). Otherwise 0
        - Local disk usage: data.DiskUsage (kilobytes)
        - CMSSW steps: data.ChirpCMSSWRuns (number of CMSSW steps)
        - Event rate: data.EventRate (events per second, based on CPU time)
                      Calculated as: ChirpCMSSWEvents / (CoreHr * 3600.0)
                      Note: There's also CMSSWEventRate = ChirpCMSSWEvents / (ChirpCMSSWElapsed * RequestCpus)
                            which is based on wall clock time, not CPU time
        - Cores requested: data.OriginalCpus (or RequestCpus)
        - Memory requested: data.OriginalMemory (MB)
        - Task name: data.WMAgent_TaskType
        """
        # Turnaround time (total job time including overhead)
        metrics.turnaround_time_sec = data.get('CommittedTime', 0.0)

        # Payload time (CMSSW execution time, without Condor overhead)
        metrics.payload_time_sec = data.get('ChirpCMSSWElapsed', 0.0)

        # Time per event (CPU seconds per event)
        # According to convert_to_json.py:
        # TimePerEvent = 1.0 / EventRate if EventRate > 0
        # where EventRate = ChirpCMSSWEvents / (CoreHr * 3600.0)
        # This is based on CPU time, not wall clock time
        # Note: There's also CMSSWTimePerEvent = 1.0 / CMSSWEventRate
        #       which is based on wall clock time (payload time), but we use TimePerEvent here
        metrics.time_per_event_sec = data.get('TimePerEvent', 0.0)

        # CPU time - prefer ChirpCMSSWTotalCPU (more accurate), fallback to CpuTimeHr
        # According to convert_to_json.py, CoreHr is also available (CPU hours)
        # and is used for EventRate calculation: EventRate = ChirpCMSSWEvents / (CoreHr * 3600.0)
        chirp_cpu = data.get('ChirpCMSSWTotalCPU', 0.0)  # seconds
        cpu_time_hr = data.get('CpuTimeHr', 0.0)  # hours
        if chirp_cpu > 0:
            metrics.cpu_time_sec = chirp_cpu
        elif cpu_time_hr > 0:
            metrics.cpu_time_sec = cpu_time_hr * 3600.0

        # Network transfer
        # Note: BytesSent + BytesRecvd are for input sandbox and output log files,
        # not physics data. These are disregarded.
        # Physics data network transfer = read_remote_mb + write_remote_mb
        # For now, network_transfer_mb is set to 0.0 since write_remote_mb is not accurately calculated
        bytes_sent = data.get('BytesSent', 0)
        bytes_recvd = data.get('BytesRecvd', 0)
        # Disregard sandbox/log files - not tracking these
        metrics.network_transfer_bytes = 0  # Placeholder - not used (sandbox/log files disregarded)

        # Disk I/O - write bytes calculation with local/remote distinction
        # For now: write_remote_mb = 0.0 (not all written data is staged to shared storage)
        # write_local_mb = total writes converted to MB
        write_total_bytes = data.get('ChirpCMSSWWriteBytes', 0)
        metrics.write_local_mb = write_total_bytes / (1024.0 * 1024.0)
        metrics.write_remote_mb = 0.0

        # Read bytes calculation with local/remote distinction
        # Local read is ALWAYS ChirpCMSSWReadBytes - ChirpCMSSW_cmsRun1_ReadBytes (regardless of DESIRED_CMSDataset)
        # Remote read is ChirpCMSSW_cmsRun1_ReadBytes ONLY if DESIRED_CMSDataset is not None, otherwise 0
        read_total_bytes = data.get('ChirpCMSSWReadBytes', 0)
        desired_cms_dataset = data.get('DESIRED_CMSDataset')
        cmsrun1_read_bytes = data.get('ChirpCMSSW_cmsRun1_ReadBytes', 0)  # Default to 0 if not present

        # Local read is always total - cmsRun1
        metrics.read_local_mb = max(0.0, (read_total_bytes - cmsrun1_read_bytes) / (1024.0 * 1024.0))

        # Remote read is cmsRun1 only if DESIRED_CMSDataset is not None
        if desired_cms_dataset is not None:
            metrics.read_remote_mb = cmsrun1_read_bytes / (1024.0 * 1024.0)
        else:
            metrics.read_remote_mb = 0.0

        # Local disk usage (in kilobytes)
        metrics.disk_usage_kb = data.get('DiskUsage', 0)

        # CMSSW steps
        metrics.num_cmssw_steps = data.get('ChirpCMSSWRuns', 0)

        # Events processed - use output events from last taskset if available
        # This accounts for filter efficiency < 1.0 (fewer events out than in)
        # NOTE that for a single step job, both metrics will contain the same value (input events)
        if metrics.num_cmssw_steps > 1:
            # Try to get output events from the last cmsRun step
            last_step_field = f'ChirpCMSSW_cmsRun{metrics.num_cmssw_steps}_Events'
            last_step_events = data.get(last_step_field, None)
            if last_step_events is not None:
                metrics.events_processed = last_step_events
            else:
                # Fallback to total events processed if last step field not available
                metrics.events_processed = data.get('ChirpCMSSWEvents', 0)
        else:
            # No CMSSW steps, use total events
            metrics.events_processed = data.get('ChirpCMSSWEvents', 0)

        # Event rate (events per second)
        # According to convert_to_json.py:
        # EventRate = ChirpCMSSWEvents / (CoreHr * 3600.0) if CoreHr > 0
        # This is based on CPU time (CoreHr = CPU hours), not wall clock time
        # Note: There's also CMSSWEventRate = ChirpCMSSWEvents / (ChirpCMSSWElapsed * RequestCpus)
        #       which is based on wall clock time (payload time), but we use EventRate here
        metrics.event_rate = data.get('EventRate', 0.0)

        # Resource allocation - use requested values
        metrics.cores_requested = data.get('OriginalCpus', 0)
        metrics.memory_requested_mb = data.get('OriginalMemory', 0.0)

        # Task identification - use WMAgent_TaskType
        metrics.task_name = data.get('WMAgent_TaskType', 'Unknown')

        # Job timing (for workflow turnaround time calculation)
        # Normalize timestamps to seconds (handle both seconds and milliseconds)
        metrics.job_start_date = self._normalize_timestamp(data.get('JobStartDate'))
        metrics.completion_date = self._normalize_timestamp(data.get('CompletionDate'))

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
        # Convert to MB (wmarchive doesn't distinguish local vs remote writes)
        metrics.write_local_mb = total_write_bytes / (1024.0 * 1024.0)
        metrics.write_remote_mb = 0.0

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

    def _calculate_summary_metrics(self) -> Dict[str, Any]:
        """
        Calculate aggregated summary metrics from all jobs.

        Returns:
            Dictionary containing all aggregated metrics
        """
        if not self.jobs:
            return {}

        # Basic counts
        total_events = sum(j.events_processed for j in self.jobs)

        # Time metrics
        total_wall_time_overhead = sum(j.turnaround_time_sec for j in self.jobs)
        total_wall_time_non_overhead = sum(j.payload_time_sec for j in self.jobs)

        # Workflow turnaround time (from first job start to last job completion)
        job_start_dates = [j.job_start_date for j in self.jobs if j.job_start_date is not None]
        completion_dates = [j.completion_date for j in self.jobs if j.completion_date is not None]
        total_turnaround_time = None
        if job_start_dates and completion_dates:
            first_start = min(job_start_dates)
            last_completion = max(completion_dates)
            total_turnaround_time = last_completion - first_start

        # CPU metrics
        total_cpu_used_time = sum(j.cpu_time_sec for j in self.jobs)
        # CPU allocated time: sum of (turnaround_time * cores_requested) for each job
        total_cpu_allocated_time = sum(
            j.turnaround_time_sec * j.cores_requested for j in self.jobs
        )

        # Network transfer
        # Physics data network transfer = read_remote_mb + write_remote_mb
        # Note: BytesSent + BytesRecvd represent sandbox/log files and are disregarded
        # For now, set to 0.0 since write_remote_mb is not accurately calculated
        total_network_transfer_mb = 0.0  # Should be total_read_remote_mb + total_write_remote_mb when accurate

        # Disk I/O - only calculate local and remote (totals can be derived from sum)
        total_write_local_mb = sum(j.write_local_mb for j in self.jobs)
        total_write_remote_mb = sum(j.write_remote_mb for j in self.jobs)
        total_read_local_mb = sum(j.read_local_mb for j in self.jobs)
        total_read_remote_mb = sum(j.read_remote_mb for j in self.jobs)
        total_disk_usage_kb = sum(j.disk_usage_kb for j in self.jobs)

        # Resource allocation
        total_cpu_cores_used = sum(j.cores_requested for j in self.jobs)
        total_memory_used_mb = sum(j.memory_requested_mb for j in self.jobs)

        # Per-event metrics
        cpu_cores_per_event = (
            total_cpu_cores_used / total_events if total_events > 0 else 0.0
        )
        memory_mb_per_event = (
            total_memory_used_mb / total_events if total_events > 0 else 0.0
        )

        # Event throughput and time per event metrics
        event_throughput = (
            total_events / total_cpu_allocated_time if total_cpu_allocated_time > 0 else 0.0
        )
        wall_time_per_event = (
            total_wall_time_non_overhead / total_events if total_events > 0 else 0.0
        )
        cpu_time_per_event = (
            total_cpu_allocated_time / total_events if total_events > 0 else 0.0
        )

        return {
            'total_jobs': len(self.jobs),
            'condor_jobs': len(self.condor_jobs),
            'wmarchive_jobs': len(self.wmarchive_jobs),
            'total_events': total_events,
            'total_wall_time_overhead': total_wall_time_overhead,
            'total_wall_time_non_overhead': total_wall_time_non_overhead,
            'total_turnaround_time': total_turnaround_time,
            'total_cpu_used_time': total_cpu_used_time,
            'total_cpu_allocated_time': total_cpu_allocated_time,
            'total_network_transfer_mb': total_network_transfer_mb,
            'total_write_local_mb': total_write_local_mb,
            'total_write_remote_mb': total_write_remote_mb,
            'total_read_local_mb': total_read_local_mb,
            'total_read_remote_mb': total_read_remote_mb,
            'total_disk_usage_kb': total_disk_usage_kb,
            'total_cpu_cores_used': total_cpu_cores_used,
            'total_memory_used_mb': total_memory_used_mb,
            'cpu_cores_per_event': cpu_cores_per_event,
            'memory_mb_per_event': memory_mb_per_event,
            'event_throughput': event_throughput,
            'wall_time_per_event': wall_time_per_event,
            'cpu_time_per_event': cpu_time_per_event,
        }

    def print_summary(self, metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Print summary statistics.

        Args:
            metrics: Optional pre-calculated summary metrics. If None, will be calculated.
        """
        if not self.jobs:
            print("No jobs to summarize")
            return

        # Use provided metrics or calculate if not provided
        if metrics is None:
            metrics = self._calculate_summary_metrics()

        print("\n" + "="*80)
        print("JOB METRICS SUMMARY")
        if self.producer_filter:
            print(f"Producer Filter: {self.producer_filter.upper()}")
        print("="*80)

        # Overall statistics
        print(f"\nTotal Jobs: {metrics['total_jobs']}")
        if not self.producer_filter:
            print(f"  - Condor: {metrics['condor_jobs']}")
            print(f"  - WMA Archive: {metrics['wmarchive_jobs']}")
        elif self.producer_filter == 'condor':
            print(f"  - Condor: {metrics['condor_jobs']}")
        elif self.producer_filter == 'wmarchive':
            print(f"  - WMA Archive: {metrics['wmarchive_jobs']}")

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
        total_events = metrics['total_events']
        print(f"\nTotal Events Processed: {total_events:,}")
        if metrics['total_jobs'] > 0:
            avg_events = total_events / metrics['total_jobs']
            print(f"Average Events per Job: {avg_events:,.1f}")

        # Time metrics
        total_wall_time_overhead = metrics['total_wall_time_overhead']
        total_wall_time_non_overhead = metrics['total_wall_time_non_overhead']
        total_turnaround_time = metrics['total_turnaround_time']

        print(f"\nTime Metrics:")
        print(f"  Total Wall Time (with overhead): {total_wall_time_overhead:,.1f} seconds ({total_wall_time_overhead/3600:.2f} hours)")
        print(f"    (Sum of all job CommittedTime)")
        print(f"  Total Wall Time (without overhead): {total_wall_time_non_overhead:,.1f} seconds ({total_wall_time_non_overhead/3600:.2f} hours)")
        print(f"    (Sum of all job ChirpCMSSWElapsed)")
        if total_wall_time_overhead > 0:
            overhead_pct = ((total_wall_time_overhead - total_wall_time_non_overhead) / total_wall_time_overhead) * 100
            print(f"  Overhead: {total_wall_time_overhead - total_wall_time_non_overhead:,.1f} seconds ({overhead_pct:.1f}%)")
        if total_turnaround_time is not None:
            print(f"  Total Turnaround Time (workflow): {total_turnaround_time:,.1f} seconds ({total_turnaround_time/3600:.2f} hours)")
            print(f"    (From first job start to last job completion)")
        else:
            print(f"  Total Turnaround Time (workflow): Not available (missing JobStartDate or CompletionDate)")

        # CPU metrics
        total_cpu_used_time = metrics['total_cpu_used_time']
        total_cpu_allocated_time = metrics['total_cpu_allocated_time']
        print(f"\nCPU Metrics:")
        print(f"  Total CPU Used Time: {total_cpu_used_time:,.1f} seconds ({total_cpu_used_time/3600:.2f} hours)")
        print(f"  Total CPU Allocated Time: {total_cpu_allocated_time:,.1f} seconds ({total_cpu_allocated_time/3600:.2f} hours)")
        if total_cpu_allocated_time > 0:
            cpu_efficiency = (total_cpu_used_time / total_cpu_allocated_time) * 100
            print(f"  CPU Efficiency: {cpu_efficiency:.1f}% (CPU used time / CPU allocated time)")

        # Network transfer
        total_network_transfer_mb = metrics['total_network_transfer_mb']
        print(f"\nNetwork Transfer (Physics Data):")
        print(f"  Total Network Transfer: {total_network_transfer_mb:.2f} MB ({total_network_transfer_mb/(1024**2):.2f} GB)")
        if total_network_transfer_mb == 0.0:
            print(f"    (Set to 0.0 for now. Should be read_remote_mb + write_remote_mb when write_remote_mb is accurately calculated)")
        else:
            print(f"    (read_remote_mb + write_remote_mb)")
        print(f"    Note: BytesSent + BytesRecvd (sandbox/log files) are disregarded")

        # Disk I/O
        total_write_local_mb = metrics['total_write_local_mb']
        total_write_remote_mb = metrics['total_write_remote_mb']
        total_read_local_mb = metrics['total_read_local_mb']
        total_read_remote_mb = metrics['total_read_remote_mb']
        total_disk_usage_kb = metrics['total_disk_usage_kb']

        # Calculate totals from local + remote
        total_write_mb = total_write_local_mb + total_write_remote_mb
        total_read_mb = total_read_local_mb + total_read_remote_mb

        print(f"\nDisk I/O:")
        print(f"  Write:")
        print(f"    - Local Write: {total_write_local_mb:.2f} MB ({total_write_local_mb/(1024**2):.2f} GB)")
        print(f"    - Remote Write: {total_write_remote_mb:.2f} MB ({total_write_remote_mb/(1024**2):.2f} GB)")
        print(f"    - Total Write: {total_write_mb:.2f} MB ({total_write_mb/(1024**2):.2f} GB)")
        print(f"  Read:")
        print(f"    - Local Read: {total_read_local_mb:.2f} MB ({total_read_local_mb/(1024**2):.2f} GB)")
        print(f"    - Remote Read: {total_read_remote_mb:.2f} MB ({total_read_remote_mb/(1024**2):.2f} GB)")
        print(f"    - Total Read: {total_read_mb:.2f} MB ({total_read_mb/(1024**2):.2f} GB)")
        print(f"  Total Local Disk Usage: {total_disk_usage_kb:,} KB ({total_disk_usage_kb/(1024**2):.2f} GB)")
        if self.producer_filter != 'wmarchive' and self.condor_jobs:
            print(f"    (Note: write_remote_mb is set to 0.0 for now - not all written data is staged to shared storage)")

        # CMSSW steps
        total_steps = sum(j.num_cmssw_steps for j in self.jobs)
        if total_steps > 0:
            print(f"\nCMSSW Steps:")
            print(f"  Total CMSSW Steps: {total_steps:,}")
            if metrics['total_jobs'] > 0:
                avg_steps = total_steps / metrics['total_jobs']
                print(f"  Average Steps per Job: {avg_steps:.1f}")

        # Event throughput and time per event metrics
        event_throughput = metrics['event_throughput']
        wall_time_per_event = metrics['wall_time_per_event']
        cpu_time_per_event = metrics['cpu_time_per_event']

        print(f"\nEvent Throughput:")
        print(f"  Event Throughput: {event_throughput:.6f} events/second")
        print(f"  Wall Time per Event: {wall_time_per_event:.6f} seconds/event")
        print(f"  CPU Time per Event: {cpu_time_per_event:.6f} seconds/event")

        # Resource allocation
        total_cpu_cores_used = metrics['total_cpu_cores_used']
        total_memory_used_mb = metrics['total_memory_used_mb']
        cpu_cores_per_event = metrics['cpu_cores_per_event']
        memory_mb_per_event = metrics['memory_mb_per_event']

        print(f"\nResource Allocation (Requested):")
        print(f"  Total Cores Requested: {total_cpu_cores_used:,}")
        if metrics['total_jobs'] > 0:
            avg_cores = total_cpu_cores_used / metrics['total_jobs']
            print(f"  Average Cores per Job: {avg_cores:.1f}")
        print(f"  Total Memory Requested: {total_memory_used_mb:,.1f} MB ({total_memory_used_mb/1024:.2f} GB)")
        if metrics['total_jobs'] > 0:
            avg_memory = total_memory_used_mb / metrics['total_jobs']
            print(f"  Average Memory per Job: {avg_memory:.1f} MB")
        if total_events > 0:
            print(f"  CPU Cores per Event: {cpu_cores_per_event:.6f} cores/event")
            print(f"  Memory per Event: {memory_mb_per_event:.6f} MB/event")

        print("\n" + "="*80)

    def print_field_hierarchy(self) -> None:
        """Print field hierarchy documentation for each metric."""
        print("\n" + "="*80)
        print("FIELD HIERARCHY DOCUMENTATION")
        if self.producer_filter:
            print(f"Producer Filter: {self.producer_filter.upper()}")
        print("="*80)

        if self.producer_filter is None or self.producer_filter == 'condor':
            print("\n## Condor Producer Fields")
            print("\n- **Events Processed**: `data.ChirpCMSSW_cmsRun{N}_Events` (output events from last taskset, where N = ChirpCMSSWRuns)")
            print("    Falls back to `data.ChirpCMSSWEvents` if last step field not available")
            print("    Note: Uses last step output events to account for filter efficiency < 1.0")
            print("- **Turnaround Time (job)**: `data.CommittedTime` (seconds, total job time including overhead)")
            print("- **Payload Time (job)**: `data.ChirpCMSSWElapsed` (seconds, CMSSW execution without overhead)")
            print("- **Time per Event**: `data.TimePerEvent` (CPU seconds per event)")
            print("    Calculated as: `1.0 / EventRate`, where `EventRate = ChirpCMSSWEvents / (CoreHr * 3600.0)`")
            print("    This is based on CPU time (CoreHr = CPU hours), not wall clock time")
            print("    Note: There's also `CMSSWTimePerEvent = 1.0 / CMSSWEventRate`")
            print("          which is based on wall clock time (payload time)")
            print("- **Job Start Date**: `data.JobStartDate` (Unix timestamp, auto-detected as seconds or milliseconds)")
            print("- **Completion Date**: `data.CompletionDate` (Unix timestamp, auto-detected as seconds or milliseconds)")
            print("\n  Workflow-level time metrics:")
            print("  - **Total Wall Time (with overhead)**: Sum of all `CommittedTime`")
            print("  - **Total Wall Time (without overhead)**: Sum of all `ChirpCMSSWElapsed`")
            print("  - **Total Turnaround Time (workflow)**: `max(CompletionDate) - min(JobStartDate)`")
            print("- **CPU Time**: `data.ChirpCMSSWTotalCPU` (seconds) or `data.CpuTimeHr` (hours)")
            print("    `CoreHr` is also available (CPU hours) and is used for EventRate calculation")
            print("- **Network Transfer (Physics Data)**: `read_remote_mb + write_remote_mb`")
            print("    Set to 0.0 for now (should be read_remote_mb + write_remote_mb when write_remote_mb is accurately calculated)")
            print("    Note: `BytesSent + BytesRecvd` (sandbox/log files) are disregarded")
            print("- **Local Writes (MB)**: `ChirpCMSSWWriteBytes / (1024 * 1024)`")
            print("- **Remote Writes (MB)**: `0.0` (not all written data is staged to shared storage)")
            print("- **Local Reads (MB)**: Always `(ChirpCMSSWReadBytes - ChirpCMSSW_cmsRun1_ReadBytes) / (1024 * 1024)`")
            print("- **Remote Reads (MB)**: If `DESIRED_CMSDataset` is not None (string),")
            print("    then `ChirpCMSSW_cmsRun1_ReadBytes / (1024 * 1024)`")
            print("    Otherwise 0")
            print("- **Local Disk Usage**: `data.DiskUsage` (kilobytes)")
            print("- **CMSSW Steps**: `data.ChirpCMSSWRuns` (number of CMSSW steps executed)")
            print("- **Event Rate**: `data.EventRate` (events per second, based on CPU time)")
            print("    Calculated as: `ChirpCMSSWEvents / (CoreHr * 3600.0)` if CoreHr > 0")
            print("    Note: There's also `CMSSWEventRate = ChirpCMSSWEvents / (ChirpCMSSWElapsed * RequestCpus)`")
            print("          which is based on wall clock time (payload time), not CPU time")
            print("- **Cores Requested**: `data.OriginalCpus` (alternative: `RequestCpus`, `GLIDEIN_Cpus`, or `JobCpus`)")
            print("- **Memory Requested**: `data.OriginalMemory` (MB, alternative: `GLIDEIN_Memory` or `MemoryProvisioned`)")
            print("- **Task Name**: `data.WMAgent_TaskType`")
            print("- **Job Type**: `data.CMS_JobType` (e.g., Production, Processing, Merge)")

        if self.producer_filter is None or self.producer_filter == 'wmarchive':
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
        print("- **Read I/O Local vs Remote**:")
        print("  - Local read (MB) = `(ChirpCMSSWReadBytes - ChirpCMSSW_cmsRun1_ReadBytes) / (1024 * 1024)`")
        print("    Always calculated this way (regardless of DESIRED_CMSDataset)")
        print("  - Remote read (MB) = `ChirpCMSSW_cmsRun1_ReadBytes / (1024 * 1024)`")
        print("    ONLY if `DESIRED_CMSDataset` is not None (string), otherwise 0")
        print("  - If `ChirpCMSSW_cmsRun1_ReadBytes` doesn't exist, defaults to 0")
        print("- **Write I/O Local vs Remote**:")
        print("  - Local write (MB) = `ChirpCMSSWWriteBytes / (1024 * 1024)`")
        print("  - Remote write (MB) = `0.0` (not all written data is staged to shared storage)")
        print("- **Network Transfer**:")
        print("  - Physics data: `read_remote_mb + write_remote_mb` (set to 0.0 for now)")
        print("  - Sandbox/log files: `BytesSent + BytesRecvd` are disregarded")
        print("- **Write I/O**: ChirpCMSSWWriteBytes includes both local and remote writes (distinction not available)")
        print("- **Local Disk**: DiskUsage provides local disk usage in kilobytes")
        print("- **Event Rate vs CMSSWEventRate**:")
        print("  - `EventRate` is based on CPU time: `ChirpCMSSWEvents / (CoreHr * 3600.0)`")
        print("  - `CMSSWEventRate` is based on wall clock time: `ChirpCMSSWEvents / (ChirpCMSSWElapsed * RequestCpus)`")
        print("  - This script uses `EventRate` (CPU-based) for consistency with CPU time metrics")
        print("- **Time per Event vs CMSSWTimePerEvent**:")
        print("  - `TimePerEvent` is the inverse of `EventRate` (CPU-based)")
        print("  - `CMSSWTimePerEvent` is the inverse of `CMSSWEventRate` (wall clock-based)")
        print("  - This script uses `TimePerEvent` (CPU-based) for consistency with CPU time metrics")
        print("- **Network metrics**: Not available in wmarchive documents")
        print("- **Core count**: Not available in wmarchive documents")
        print("- **Condor data**: Provides more comprehensive metrics")
        print("- **WMA archive data**: Provides detailed step-by-step breakdown")
        print("\n## Reference")
        print("\nMetric definitions are based on `convert_to_json.py` from cms-htcondor-es:")
        print("https://github.com/dmwm/cms-htcondor-es/blob/master/src/htcondor_es/convert_to_json.py")

        print("\n" + "="*80)

    def export_to_json(self, output_file: str, metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Export extracted metrics to JSON file.

        Args:
            output_file: Path to output JSON file
            metrics: Optional pre-calculated summary metrics. If None, will be calculated.
        """
        # Use provided metrics or calculate if not provided
        if metrics is None:
            metrics = self._calculate_summary_metrics()

        # Handle empty jobs case
        if not metrics:
            print("No jobs to export")
            return

        output_data = {
            'summary': {
                'total_jobs': metrics['total_jobs'],
                'condor_jobs': metrics['condor_jobs'],
                'wmarchive_jobs': metrics['wmarchive_jobs'],
                'total_events': metrics['total_events'],
                'total_wall_time_overhead': metrics['total_wall_time_overhead'],
                'total_wall_time_non_overhead': metrics['total_wall_time_non_overhead'],
                'total_turnaround_time': metrics['total_turnaround_time'],
                'total_cpu_used_time': metrics['total_cpu_used_time'],
                'total_cpu_allocated_time': metrics['total_cpu_allocated_time'],
                'total_write_local_mb': metrics['total_write_local_mb'],
                'total_write_remote_mb': metrics['total_write_remote_mb'],
                'total_read_local_mb': metrics['total_read_local_mb'],
                'total_read_remote_mb': metrics['total_read_remote_mb'],
                'total_network_transfer_mb': metrics['total_network_transfer_mb'],
                'total_cpu_cores_used': metrics['total_cpu_cores_used'],
                'total_memory_used_mb': metrics['total_memory_used_mb'],
                'cpu_cores_per_event': metrics['cpu_cores_per_event'],
                'memory_mb_per_event': metrics['memory_mb_per_event'],
                'event_throughput': metrics['event_throughput'],
                'wall_time_per_event': metrics['wall_time_per_event'],
                'cpu_time_per_event': metrics['cpu_time_per_event'],
                'total_disk_usage_kb': metrics['total_disk_usage_kb'],
                'note_local_remote_io': (
                    'Read local vs remote distinction (in MB) is calculated when ChirpCMSSW_cmsRun1_ReadBytes is available. '
                    'Local read (MB) = (ChirpCMSSWReadBytes - ChirpCMSSW_cmsRun1_ReadBytes) / (1024 * 1024) always. '
                    'Remote read (MB) = ChirpCMSSW_cmsRun1_ReadBytes / (1024 * 1024) ONLY if DESIRED_CMSDataset is not None (string), otherwise 0. '
                    'If ChirpCMSSW_cmsRun1_ReadBytes doesn\'t exist, defaults to 0. '
                    'Write local (MB) = ChirpCMSSWWriteBytes / (1024 * 1024). '
                    'Write remote (MB) = 0.0 for now (not all written data is staged to shared storage). '
                    'Network transfer (MB) = read_remote_mb + write_remote_mb (physics data only). '
                    'Set to 0.0 for now since write_remote_mb is not accurately calculated. '
                    'BytesSent + BytesRecvd (sandbox/log files) are disregarded.'
                ),
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
                    'job_start_date': j.job_start_date,
                    'completion_date': j.completion_date,
                    'cpu_time_sec': j.cpu_time_sec,
                    'network_transfer_bytes': j.network_transfer_bytes,
                    'write_local_mb': j.write_local_mb,
                    'write_remote_mb': j.write_remote_mb,
                    'read_local_mb': j.read_local_mb,
                    'read_remote_mb': j.read_remote_mb,
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

  # Print field hierarchy documentation
  python explore_job_data.py data/const001.json --verbose
        """
    )
    parser.add_argument('json_file', help='Path to JSON file containing Elasticsearch results')
    parser.add_argument('output_file', nargs='?', help='Optional output JSON file for exported metrics')
    parser.add_argument(
        '--producer',
        choices=['condor', 'wmarchive'],
        help='Filter to specific producer type (default: analyze both)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print field hierarchy documentation'
    )

    args = parser.parse_args()

    explorer = JobDataExplorer(args.json_file, producer_filter=args.producer)
    explorer.load_data()

    # Calculate summary metrics once and reuse for both print and export
    summary_metrics = explorer._calculate_summary_metrics()
    explorer.print_summary(metrics=summary_metrics)

    if args.verbose:
        explorer.print_field_hierarchy()

    if args.output_file:
        explorer.export_to_json(args.output_file, metrics=summary_metrics)


if __name__ == '__main__':
    main()

