"""
Workflow Simulator

This module provides a comprehensive workflow simulation engine that executes
workflow compositions following DAG processing rules with group-based job scheduling.
"""

import json
import logging
import math
import argparse
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from collections import defaultdict, deque



@dataclass
class ResourceConfig:
    """Resource configuration for workflow execution."""
    target_wallclock_time: float = 43200.0  # 12 hours in seconds
    max_job_slots: int = -1  # -1 means infinite
    cpu_per_slot: int = 1
    memory_per_slot: int = 1000  # MB


@dataclass
class TasksetInfo:
    """Information about a single taskset."""
    taskset_id: str
    group_name: str
    input_taskset: Optional[str]
    time_per_event: float
    memory: int
    multicore: int
    size_per_event: int
    group_input_events: int
    scram_arch: List[str]
    requires_gpu: str
    keep_output: bool


@dataclass
class GroupInfo:
    """Information about a workflow group."""
    group_id: str
    tasksets: List[TasksetInfo]
    input_events: int
    job_count: int
    exact_job_count: float  # Fractional job count for precise calculations
    total_execution_time: float
    dependencies: List[str]


@dataclass
class JobInfo:
    """Information about a single job."""
    job_id: str
    group_id: str
    batch_size: int
    wallclock_time: float
    start_time: float
    end_time: float
    status: str  # 'pending', 'running', 'completed', 'failed'


@dataclass
class EventBuffer:
    """Event buffer to track available events for each group."""
    group_id: str
    available_events: int
    processed_events: int
    total_events: int


@dataclass
class SimulationResult:
    """Result of workflow simulation."""
    workflow_id: str
    composition_number: int
    total_events: int
    total_groups: int
    total_jobs: int
    total_wall_time: float  # Sum of all job wallclock times
    total_turnaround_time: float  # Time from workflow start to completion
    groups: List[GroupInfo]
    jobs: List[JobInfo]
    execution_log: List[Dict[str, Any]]
    success: bool
    error_message: Optional[str] = None


class WorkflowSimulator:
    """
    Workflow simulation engine with DAG execution and group-based job scheduling.

    This class simulates workflow execution following the specified rules:
    - Tasksets within groups execute sequentially
    - Independent groups can execute in parallel
    - Job count per group based on RequestNumEvents/GroupInputEvents
    - Respects target wallclock time constraints
    """

    def __init__(self, resource_config: Optional[ResourceConfig] = None):
        """
        Initialize the workflow simulator.

        Args:
            resource_config: Resource configuration for simulation
        """
        self.resource_config = resource_config or ResourceConfig()
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s:%(name)s:%(levelname)s: %(message)s'
        )

    def simulate_workflow(self, workflow_filepath: Union[str, Path]) -> SimulationResult:
        """
        Simulate workflow execution based on the provided workflow file.

        Args:
            workflow_filepath: Path to JSON file containing workflow definition

        Returns:
            SimulationResult object with execution details and metrics
        """
        self.logger.info("Starting workflow simulation")

        ### Load and setup workflow simulation
        try:
            with open(workflow_filepath, 'r') as f:
                workflow_data = json.load(f)

            self.logger.info(f"Loaded workflow data from: {workflow_filepath}")
            self.logger.info(f"Using resources: {self.resource_config}")
            # Parse workflow data
            # Use file path as workflow_id if not present in JSON
            workflow_id = workflow_data.get('workflow_id', str(workflow_filepath))
            composition_number = workflow_data.get('CompositionNumber', 0)
            request_num_events = workflow_data.get('RequestNumEvents', 0)

            # Extract tasksets and build groups
            tasksets = self._extract_tasksets(workflow_data)
            groups = self._build_groups(tasksets, request_num_events)

            # Build dependency graph
            dependency_graph = self._build_dependency_graph(tasksets)
            self.logger.info(f"Dependency graph: {dependency_graph}")

            # Calculate job counts for each group
            self._calculate_job_counts(groups, request_num_events)
        except Exception as e:
            self.logger.exception(f"Setup workflow simulation failed: {str(e)}")
            raise RuntimeError(f"Setup workflow simulation failed: {str(e)}")

        # Simulate workflow execution
        try:
            execution_result = self._simulate_execution(groups, dependency_graph, request_num_events)

            # Create simulation result
            result = SimulationResult(
                workflow_id=workflow_id,
                composition_number=composition_number,
                total_events=request_num_events,
                total_groups=len(groups),
                total_jobs=sum(group.job_count for group in groups),
                total_wall_time=execution_result['total_wall_time'],
                total_turnaround_time=execution_result['total_turnaround_time'],
                groups=groups,
                jobs=execution_result['jobs'],
                execution_log=execution_result['execution_log'],
                success=True
            )

            self.logger.info(f"Workflow simulation completed successfully. "
                           f"Total jobs: {result.total_jobs}, "
                           f"Wall time: {result.total_wall_time:.2f}s")

            return result
        except Exception as e:
            self.logger.error(f"Workflow simulation failed: {str(e)}")
            return SimulationResult(
                workflow_id='unknown',
                composition_number=0,
                total_events=0,
                total_groups=0,
                total_jobs=0,
                total_wall_time=0.0,
                total_turnaround_time=0.0,
                groups=[],
                jobs=[],
                execution_log=[],
                success=False,
                error_message=str(e)
            )

    def _extract_tasksets(self, workflow_data: Dict[str, Any]) -> List[TasksetInfo]:
        """Extract taskset information from workflow data."""
        tasksets = []

        for key, value in workflow_data.items():
            if key.startswith('Taskset') and isinstance(value, dict):
                taskset = TasksetInfo(
                    taskset_id=key,
                    group_name=value.get('GroupName', ''),
                    input_taskset=value.get('InputTaskset'),
                    time_per_event=value.get('TimePerEvent', 0.0),
                    memory=value.get('Memory', 0),
                    multicore=value.get('Multicore', 1),
                    size_per_event=value.get('SizePerEvent', 0),
                    group_input_events=value.get('GroupInputEvents', 0),
                    scram_arch=value.get('ScramArch', []),
                    requires_gpu=value.get('RequiresGPU', 'forbidden'),
                    keep_output=value.get('KeepOutput', False)
                )
                tasksets.append(taskset)

        return tasksets

    def _build_groups(self, tasksets: List[TasksetInfo],
                     request_num_events: int) -> List[GroupInfo]:
        """Build group information from tasksets."""
        group_dict = defaultdict(list)

        # Group tasksets by GroupName
        for taskset in tasksets:
            group_dict[taskset.group_name].append(taskset)

        groups = []
        for group_id, group_tasksets in group_dict.items():
            # Sort tasksets by dependency order within group
            sorted_tasksets = self._sort_tasksets_by_dependency(group_tasksets)

            # Calculate total execution time for the group
            total_time = sum(ts.time_per_event * ts.group_input_events
                           for ts in sorted_tasksets)

            group = GroupInfo(
                group_id=group_id,
                tasksets=sorted_tasksets,
                input_events=group_tasksets[0].group_input_events if group_tasksets else 0,
                job_count=0,  # Will be calculated later
                exact_job_count=0.0,  # Will be calculated later
                total_execution_time=total_time,
                dependencies=self._get_group_dependencies(sorted_tasksets)
            )
            groups.append(group)

        return groups

    def _sort_tasksets_by_dependency(self, tasksets: List[TasksetInfo]) -> List[TasksetInfo]:
        """Sort tasksets within a group by dependency order."""
        # Create a mapping of taskset_id to taskset
        taskset_map = {ts.taskset_id: ts for ts in tasksets}

        # Build dependency graph for this group
        dependencies = {}
        for ts in tasksets:
            if ts.input_taskset and ts.input_taskset in taskset_map:
                dependencies[ts.taskset_id] = ts.input_taskset

        # Topological sort
        sorted_tasksets = []
        visited = set()
        temp_visited = set()

        def visit(taskset_id: str) -> None:
            if taskset_id in temp_visited:
                raise ValueError(f"Circular dependency detected: {taskset_id}")
            if taskset_id in visited:
                return

            temp_visited.add(taskset_id)
            if taskset_id in dependencies:
                visit(dependencies[taskset_id])
            temp_visited.remove(taskset_id)
            visited.add(taskset_id)
            sorted_tasksets.append(taskset_map[taskset_id])

        for ts in tasksets:
            if ts.taskset_id not in visited:
                visit(ts.taskset_id)

        return sorted_tasksets

    def _get_group_dependencies(self, tasksets: List[TasksetInfo]) -> List[str]:
        """Get external dependencies for a group."""
        dependencies = set()
        for ts in tasksets:
            if ts.input_taskset:
                # Check if input_taskset is from a different group
                # This is a simplified check - in practice, you'd need group mapping
                dependencies.add(ts.input_taskset)
        return list(dependencies)

    def _build_dependency_graph(self, tasksets: List[TasksetInfo]) -> Dict[str, List[str]]:
        """Build dependency graph for parallel execution."""
        # Group tasksets by group name
        group_tasksets = defaultdict(list)
        for ts in tasksets:
            group_tasksets[ts.group_name].append(ts)

        # Build group-level dependencies
        group_deps = defaultdict(list)
        for group_name, group_ts in group_tasksets.items():
            for ts in group_ts:
                if ts.input_taskset:
                    # Find which group the input taskset belongs to
                    for other_group, other_ts in group_tasksets.items():
                        if other_group != group_name:
                            if any(ots.taskset_id == ts.input_taskset for ots in other_ts):
                                group_deps[group_name].append(other_group)
                                break

        return dict(group_deps)

    def _calculate_job_counts(self, groups: List[GroupInfo],
                             request_num_events: int) -> None:
        """Calculate job count for each group based on event scaling."""
        for group in groups:
            if group.input_events > 0:
                # Calculate exact number of jobs needed (including fractional)
                exact_job_count = request_num_events / group.input_events
                group.job_count = math.ceil(exact_job_count)
                group.exact_job_count = exact_job_count  # Store fractional value
            else:
                group.job_count = 1
                group.exact_job_count = 1.0

            self.logger.info(f"Group {group.group_id}: {group.job_count} jobs "
                           f"({request_num_events} events / {group.input_events} per job, "
                           f"exact: {group.exact_job_count:.3f})")

    def _simulate_execution(self, groups: List[GroupInfo],
                          dependency_graph: Dict[str, List[str]],
                          request_num_events: int) -> Dict[str, Any]:
        """Simulate the execution of groups with dependency resolution and job slot constraints."""
        execution_log = []
        current_time = 0.0

        # Initialize event buffers for each group
        event_buffers = self._initialize_event_buffers(groups, request_num_events)

        # Track running jobs and available slots
        running_jobs = []
        all_created_jobs = []  # Track all jobs created during simulation
        available_slots = self.resource_config.max_job_slots if self.resource_config.max_job_slots > 0 else float('inf')
        batch_number = 0  # Track batch numbers across the entire simulation

        # Track group completion status
        completed_groups = set()

        # Initialize execution queue with groups that have no dependencies
        execution_queue = deque()
        for group in groups:
            if not dependency_graph.get(group.group_id, []):
                execution_queue.append(group.group_id)

        self.logger.info(f"Starting simulation with {available_slots} job slots available")

        while execution_queue or running_jobs:
            # Process completed jobs and update event buffers
            self._process_completed_jobs(running_jobs, event_buffers, current_time, execution_log)

            # Check if all events have been processed
            total_processed = sum(buffer.get('processed_events', 0) for buffer in event_buffers.values())
            if total_processed >= request_num_events:
                self.logger.info(f"All events processed: {total_processed}/{request_num_events}")
                break

            # Create new jobs based on available events and slots
            available_slots_for_new_jobs = available_slots - len(running_jobs)
            if available_slots_for_new_jobs > 0 and execution_queue:
                batch_number += 1
                self.logger.info(f"=== BATCH {batch_number} ===")
                self.logger.debug(f"Before job creation: {len(execution_queue)} groups in queue, {available_slots_for_new_jobs} slots available")
                new_jobs = self._create_jobs_for_ready_groups(
                    groups, event_buffers, execution_queue, available_slots_for_new_jobs, request_num_events, running_jobs, batch_number
                )
                self.logger.debug(f"After job creation: {len(execution_queue)} groups in queue, {len(new_jobs)} new jobs created")
            else:
                new_jobs = []
                if available_slots_for_new_jobs <= 0:
                    self.logger.info(f"No job creation: {len(running_jobs)} jobs running, {available_slots} max slots")
                elif not execution_queue:
                    self.logger.info(f"No job creation: no groups in execution queue")

            # Start new jobs
            for job in new_jobs:
                job.start_time = current_time
                job.status = 'running'
                running_jobs.append(job)
                all_created_jobs.append(job)  # Track all created jobs

                log_entry = {
                    'timestamp': current_time,
                    'event': 'job_started',
                    'job_id': job.job_id,
                    'group_id': job.group_id,
                    'batch_size': job.batch_size,
                    'wallclock_time': job.wallclock_time
                }
                execution_log.append(log_entry)

                self.logger.debug(f"Started job {job.job_id}: {job.batch_size} events, {job.wallclock_time:.2f}s")

            # If no jobs were created and no jobs are running, break
            if not new_jobs and not running_jobs:
                break

            # Advance time to next job completion
            if running_jobs:
                next_completion_time = min(job.start_time + job.wallclock_time for job in running_jobs)
                current_time = next_completion_time
            else:
                break

        # Calculate total wall time as sum of all job wallclock times
        total_wall_time = sum(job.wallclock_time for job in all_created_jobs)

        return {
            'total_wall_time': total_wall_time,
            'total_turnaround_time': current_time,
            'jobs': all_created_jobs,
            'execution_log': execution_log
        }

    def _calculate_batch_size(self, group: GroupInfo) -> int:
        """Calculate batch size to meet wallclock time constraints."""
        if not group.tasksets:
            return group.input_events

        # Calculate time per event for the group
        total_time_per_event = sum(ts.time_per_event for ts in group.tasksets)

        if total_time_per_event <= 0:
            return group.input_events

        # Calculate how many events can fit in target wallclock time
        max_events_per_job = int(self.resource_config.target_wallclock_time / total_time_per_event)

        # Don't exceed the group's input events
        return min(max_events_per_job, group.input_events)

    def _calculate_job_wallclock_time(self, group: GroupInfo, batch_size: int) -> float:
        """
        Calculate actual wallclock time for a job.
        Args:
            group: GroupInfo object
            batch_size: int, number of events to process in the job
        Returns:
            float: Actual wallclock time for a job in seconds
        """
        if not group.tasksets:
            return 0.0

        total_time_per_event = sum(ts.time_per_event for ts in group.tasksets)
        return total_time_per_event * batch_size

    def _execute_group_jobs(self, group_jobs: List[JobInfo],
                           start_time: float,
                           execution_log: List[Dict[str, Any]]) -> float:
        """Execute all jobs in a group in parallel and return total execution time."""
        if not group_jobs:
            return 0.0

        # Jobs in a group execute in parallel (all start at the same time)
        max_job_time = 0.0

        for job in group_jobs:
            job.start_time = start_time
            job.end_time = start_time + job.wallclock_time
            job.status = 'completed'

            # Log job execution
            log_entry = {
                'timestamp': start_time,
                'event': 'job_started',
                'job_id': job.job_id,
                'group_id': job.group_id,
                'batch_size': job.batch_size,
                'wallclock_time': job.wallclock_time,
                'start_time': job.start_time,
                'end_time': job.end_time
            }
            execution_log.append(log_entry)

            self.logger.debug(f"Job {job.job_id}: {job.batch_size} events, {job.wallclock_time:.2f}s wallclock")

            max_job_time = max(max_job_time, job.wallclock_time)

        return max_job_time

    def _initialize_event_buffers(self, groups: List[GroupInfo], request_num_events: int) -> Dict[str, Dict[str, Any]]:
        """Initialize event buffers for each group."""
        event_buffers = {}

        for group in groups:
            # Check if this group has any input tasksets (tasksets with no input_taskset)
            has_input_tasksets = any(not ts.input_taskset for ts in group.tasksets)

            if has_input_tasksets:
                # This group has input tasksets - starts with all events
                available_events = request_num_events
            else:
                # This group only has dependent tasksets - starts with no events
                available_events = 0

            event_buffers[group.group_id] = {
                'group_id': group.group_id,
                'available_events': available_events,
                'processed_events': 0,
                'total_events': request_num_events,
                'jobs': [],
                'job_counter': 0  # Track total jobs created for this group
            }

            self.logger.debug(f"Initialized buffer for {group.group_id}: {available_events} events available")

        return event_buffers

    def _process_completed_jobs(self, running_jobs: List[JobInfo],
                               event_buffers: Dict[str, Dict[str, Any]],
                               current_time: float,
                               execution_log: List[Dict[str, Any]]) -> None:
        """Process completed jobs and update event buffers."""
        completed_jobs = []

        for job in running_jobs:
            if current_time >= job.start_time + job.wallclock_time:
                # Job completed
                job.end_time = current_time
                job.status = 'completed'
                completed_jobs.append(job)

                # Update event buffer
                if job.group_id in event_buffers:
                    buffer = event_buffers[job.group_id]
                    buffer['processed_events'] += job.batch_size
                    buffer['jobs'].append(job)

                    # Log job completion
                    log_entry = {
                        'timestamp': current_time,
                        'event': 'job_completed',
                        'job_id': job.job_id,
                        'group_id': job.group_id,
                        'batch_size': job.batch_size,
                        'wallclock_time': job.wallclock_time,
                        'processed_events': buffer['processed_events']
                    }
                    execution_log.append(log_entry)

                    self.logger.debug(f"Completed job {job.job_id}: processed {job.batch_size} events")

        # Remove completed jobs from running list
        for job in completed_jobs:
            running_jobs.remove(job)

    def _create_jobs_for_ready_groups(self, groups: List[GroupInfo],
                                     event_buffers: Dict[str, Dict[str, Any]],
                                     execution_queue: deque,
                                     available_slots: int,
                                     request_num_events: int,
                                     running_jobs: List[JobInfo],
                                     batch_number: int) -> List[JobInfo]:
        """Create jobs for ready groups based on available events and slots."""
        new_jobs = []
        slots_used = 0

        self.logger.debug(f"Creating jobs: {len(execution_queue)} groups in queue, {available_slots} slots available")
        self.logger.info(f"Job creation: {len(execution_queue)} groups in queue, {available_slots} slots available, {len(running_jobs)} jobs running")

        # Process groups in queue order - create jobs in batches
        temp_queue = deque()
        completed_groups_in_batch = set()  # Track groups that completed in this batch
        batch_count = 0

        while execution_queue and slots_used < available_slots:
            group_id = execution_queue.popleft()
            group = next((g for g in groups if g.group_id == group_id), None)

            if not group:
                continue

            buffer = event_buffers.get(group_id, {})
            available_events = buffer.get('available_events', 0)

            self.logger.debug(f"Group {group_id}: {available_events} events available, needs {group.input_events}")

            if available_events > 0:
                # Can create a job for this group
                batch_size = self._calculate_batch_size(group)

                # For the last job, only process remaining events
                # If available events is less than or equal to batch size, this is the last job
                if available_events <= batch_size:
                    # This is the last job - process all remaining events
                    actual_batch_size = available_events
                    self.logger.debug(f"Group {group_id}: Last job processing {actual_batch_size} remaining events")
                else:
                    # Regular job - use full batch size
                    actual_batch_size = batch_size

                # Don't create a job if it would process more events than needed
                if actual_batch_size > available_events:
                    actual_batch_size = available_events

                # Don't create a job if it would process more events than the total requested
                # Count both processed events AND events in pending/running jobs
                total_processed = sum(buffer.get('processed_events', 0) for buffer in event_buffers.values())
                total_pending = sum(job.batch_size for job in new_jobs)  # Jobs created in this batch
                total_running = sum(job.batch_size for job in running_jobs if job.group_id == group_id)  # Running jobs for this group
                total_accounted = total_processed + total_pending + total_running

                if total_accounted + actual_batch_size > request_num_events:
                    actual_batch_size = max(0, request_num_events - total_accounted)

                self.logger.debug(f"Group {group_id}: batch_size={batch_size}, actual_batch_size={actual_batch_size}")

                if actual_batch_size > 0:
                    # Use the job counter from the event buffer
                    buffer['job_counter'] += 1
                    job_id = f"{group_id}_job_{buffer['job_counter']}"
                    job_wallclock = self._calculate_job_wallclock_time(group, actual_batch_size)

                    job = JobInfo(
                        job_id=job_id,
                        group_id=group_id,
                        batch_size=actual_batch_size,
                        wallclock_time=job_wallclock,
                        start_time=0.0,  # Will be set when job starts
                        end_time=0.0,
                        status='pending'
                    )

                    new_jobs.append(job)
                    slots_used += 1
                    batch_count += 1

                    # Update available events
                    buffer['available_events'] -= actual_batch_size

                    self.logger.info(f"Created job {job_id}: {actual_batch_size} events, {job_wallclock:.2f}s")

                    # Check if we need to create more jobs for this group
                    # Count total events that will be processed (completed + pending + running)
                    total_processed = sum(buf.get('processed_events', 0) for buf in event_buffers.values())
                    total_pending = sum(job.batch_size for job in new_jobs)
                    total_running = sum(job.batch_size for job in running_jobs if job.group_id == group_id)
                    total_accounted = total_processed + total_pending + total_running

                    self.logger.debug(f"Group {group_id}: available_events={buffer['available_events']}, input_events={group.input_events}, total_accounted={total_accounted}, request_num_events={request_num_events}")

                    if buffer['available_events'] > 0 and total_accounted < request_num_events:
                        # Still have events to process, put group back at front of queue
                        execution_queue.appendleft(group_id)
                        self.logger.debug(f"Group {group_id} put back in queue: {buffer['available_events']} events available")
                    else:
                        # This group is done, add dependent groups to queue
                        self.logger.debug(f"Group {group_id} completed: {buffer['available_events']} events available, {total_accounted} accounted")
                        completed_groups_in_batch.add(group_id)
                        self._add_dependent_groups_to_queue(group_id, groups, event_buffers, execution_queue)
                else:
                    # Not enough events, put back in queue
                    temp_queue.append(group_id)
            else:
                # Not enough events, put back in queue
                temp_queue.append(group_id)

        # Put back groups that couldn't be processed (but not completed groups)
        for group_id in reversed(temp_queue):
            if group_id not in completed_groups_in_batch:
                execution_queue.appendleft(group_id)

                if len(new_jobs) > 0:
                    self.logger.info(f"Batch {batch_number} summary: {len(new_jobs)} jobs created")

        return new_jobs

    def _add_dependent_groups_to_queue(self, completed_group_id: str,
                                      groups: List[GroupInfo],
                                      event_buffers: Dict[str, Dict[str, Any]],
                                      execution_queue: deque) -> None:
        """Add dependent groups to execution queue when a group completes."""
        for group in groups:
            # Check if this group depends on the completed group
            depends_on_completed = any(
                ts.input_taskset and
                any(ts2.taskset_id == ts.input_taskset for ts2 in
                    next((g.tasksets for g in groups if g.group_id == completed_group_id), []))
                for ts in group.tasksets
            )

            if depends_on_completed:
                # This group depends on the completed group
                # Add processed events from completed group to this group's available events
                completed_buffer = event_buffers.get(completed_group_id, {})
                processed_events = completed_buffer.get('processed_events', 0)

                if processed_events > 0:
                    current_buffer = event_buffers.get(group.group_id, {})
                    current_buffer['available_events'] += processed_events

                    # Add to execution queue if not already there
                    if group.group_id not in execution_queue:
                        execution_queue.append(group.group_id)

                    self.logger.debug(f"Added {processed_events} events to {group.group_id} from {completed_group_id}")


    def print_simulation_summary(self, result: SimulationResult) -> None:
        """Print a summary of the simulation results."""
        print("\n" + "="*60)
        print("WORKFLOW SIMULATION SUMMARY")
        print("="*60)

        print(f"Workflow ID: {result.workflow_id}")
        print(f"Composition Number: {result.composition_number}")
        print(f"Total Events: {result.total_events:,}")
        print(f"Total Groups: {result.total_groups}")
        print(f"Total Jobs: {result.total_jobs}")
        print(f"Total Wall Time: {result.total_wall_time:.2f} seconds ({result.total_wall_time/3600:.2f} hours)")
        print(f"Total Turnaround Time: {result.total_turnaround_time:.2f} seconds ({result.total_turnaround_time/3600:.2f} hours)")
        print(f"Success: {result.success}")

        if result.error_message:
            print(f"Error: {result.error_message}")

        print("\n" + "-"*40)
        print("GROUP DETAILS")
        print("-"*40)

        for group in result.groups:
            print(f"\nGroup: {group.group_id}")
            print(f"  Jobs: {group.job_count}")
            print(f"  Input Events: {group.input_events:,}")
            print(f"  Total Execution Time: {group.total_execution_time:.2f}s")
            print(f"  Tasksets: {len(group.tasksets)}")

            for taskset in group.tasksets:
                print(f"    {taskset.taskset_id}: {taskset.time_per_event}s/event, "
                      f"{taskset.memory}MB, {taskset.multicore} cores")

        print("\n" + "-"*40)
        print("JOB EXECUTION LOG (last 10 jobs)")
        print("-"*40)

        for log_entry in result.execution_log[-10:]:  # Show last 10 jobs
            print(f"Time {log_entry['timestamp']:.2f}s: {log_entry['event']} - "
                  f"{log_entry['job_id']} ({log_entry['batch_size']} events, "
                  f"{log_entry['wallclock_time']:.2f}s)")

    def write_simulation_result(self, result: SimulationResult,
                               filepath: Union[str, Path]) -> None:
        """Write simulation result to a JSON file."""
        result_dict = asdict(result)

        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)

        self.logger.info(f"Simulation result written to {filepath}")


def load_workflow_from_file(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load workflow data from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def _get_output_path(input_path: str) -> str:
    """
    Generate output path based on input path structure.

    Args:
        input_path: Path to input workflow file

    Returns:
        Output path in results/ directory with same structure (excluding templates/ prefix)
    """
    input_path_obj = Path(input_path)

    # Remove 'templates/' prefix if present
    if input_path_obj.parts[0] == 'templates':
        relative_path = input_path_obj.relative_to('templates')
    else:
        relative_path = input_path_obj

    # Create output path: results/ + relative path
    output_path = Path("results") / relative_path

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return str(output_path)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Workflow Simulator - Workflow execution simulation engine'
    )
    parser.add_argument(
        '--target-wallclock-time',
        type=int,
        default=43200,
        help='Target wallclock time in seconds (default: 43200 = 12 hours)'
    )
    parser.add_argument(
        '--max-job-slots',
        type=int,
        default=-1,
        help='Maximum number of job slots (-1 for infinite, default: -1)'
    )
    parser.add_argument(
        '--input-workflow-path',
        type=str,
        default='templates/3tasks_composition_001.json',
        help='Path to input workflow JSON file (default: templates/3tasks_composition_001.json)'
    )
    return parser.parse_args()


def main():
    """Main function with command line argument support."""
    args = parse_arguments()

    # Configure resources from command line arguments
    resource_config = ResourceConfig(
        target_wallclock_time=args.target_wallclock_time,
        max_job_slots=args.max_job_slots
    )

    # Create simulator and run simulation
    simulator = WorkflowSimulator(resource_config)
    result = simulator.simulate_workflow(args.input_workflow_path)

    # Print results
    simulator.print_simulation_summary(result)

    # Write results to file with same structure as input
    output_path = _get_output_path(args.input_workflow_path)
    simulator.write_simulation_result(result, output_path)


if __name__ == "__main__":
    main()
