#!/usr/bin/env python3
"""
Normalize Real Workflow Metrics to 1M Events

This script scales whole-workflow total metrics from real data to 1,000,000 events
for fair comparison with simulated data. Event-normalized metrics and ratios are
preserved unchanged.

Usage:
    python scripts/normalize_real_metrics.py input_file.json output_file.json
    python scripts/normalize_real_metrics.py results/real/summary_const001_1M.json results/real/summary_const001_1M_normalized.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any


# Metrics that should be scaled (whole-workflow totals)
METRICS_TO_SCALE = {
    'time_metrics': [
        'total_wallclock_time_with_overhead_sec',
        'total_wallclock_time_with_overhead_hours',
        'total_wallclock_time_non_overhead_sec',
        'total_wallclock_time_non_overhead_hours',
        'total_overhead_sec',
        'total_overhead_hours',
    ],
    'cpu_metrics': [
        'total_cpu_time_sec',
        'total_cpu_time_hours',
        'total_cpu_used_time_sec',
        'total_cpu_used_time_hours',
        'total_cpu_allocated_time_sec',
        'total_cpu_allocated_time_hours',
        'total_cpu_cores_used',
    ],
    'memory_metrics': [
        'total_memory_used_mb',
        'total_memory_used_gb',
        'total_memory_allocated_mb',
        'total_memory_allocated_gb',
    ],
    'io_metrics': [
        'total_read_local_mb',
        'total_read_local_gb',
        'total_read_remote_mb',
        'total_read_remote_gb',
        'total_read_mb',
        'total_read_gb',
        'total_write_local_mb',
        'total_write_local_gb',
        'total_write_remote_mb',
        'total_write_remote_gb',
        'total_write_mb',
        'total_write_gb',
    ],
}


def normalize_real_data_file(
    input_file: Path,
    output_file: Path,
    target_events: int = 1_000_000
) -> None:
    """
    Load real data JSON file, normalize metrics, and save to output file.
    
    Args:
        input_file: Path to input JSON file (real data format)
        output_file: Path to output JSON file
        target_events: Target number of events (default: 1,000,000)
    """
    # Load input file
    print(f"Loading: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract actual events
    event_metrics = data.get('event_metrics', {})
    actual_events = event_metrics.get('total_events', 0)
    
    if actual_events <= 0:
        raise ValueError(
            f"Invalid total_events in input file: {actual_events}. "
            "Cannot normalize metrics."
        )
    
    scaling_factor = target_events / actual_events
    
    print(f"  Actual events: {actual_events:,}")
    print(f"  Target events: {target_events:,}")
    print(f"  Scaling factor: {scaling_factor:.6f}")
    
    # Scale metrics in place
    for category, metric_list in METRICS_TO_SCALE.items():
        if category in data:
            for metric_name in metric_list:
                if metric_name in data[category]:
                    original_value = data[category][metric_name]
                    data[category][metric_name] = original_value * scaling_factor
    
    # Save output file
    print(f"Saving normalized metrics to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print("✓ Normalization complete!")


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Normalize real workflow metrics to 1M events for comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normalize a single file
  python scripts/normalize_real_metrics.py \\
      results/real/summary_const001_1M.json \\
      results/real/summary_const001_1M_normalized.json
        """
    )
    
    parser.add_argument(
        'input_file',
        type=Path,
        help='Input JSON file (real data format)'
    )
    
    parser.add_argument(
        'output_file',
        type=Path,
        help='Output JSON file (normalized metrics)'
    )
    
    parser.add_argument(
        '--target-events',
        type=int,
        default=1_000_000,
        help='Target number of events (default: 1,000,000)'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory if needed
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        normalize_real_data_file(
            args.input_file,
            args.output_file,
            args.target_events
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
