#!/usr/bin/env python3
"""
Workflow Type Sensitivity Analysis Script

This script analyzes how different workflow types respond to hybrid workflow
constructions compared to extreme cases. It aggregates data across workflow
types to identify which types benefit most from hybrid compositions.

Analysis: Workflow Type Sensitivity (Comparison #2)
- Fixed: target_job_length + failure_rate
- Variable: workflow_type (case1_real, case2_homo, case3_hetero)
- Compare: Const 1, Const 16, and best hybrid across workflow types
- Primary Metric: event_throughput
- Second Metric: network_transfer_mb_per_event
"""

import argparse
import json
import os
from typing import List, Dict, Any, Optional
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_simulation_data(file_path: str) -> Optional[Dict[str, Any]]:
    """Load and extract key metrics from a simulation JSON file.

    Args:
        file_path: Path to simulation result JSON file

    Returns:
        Dictionary with extracted metrics, or None if loading fails
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        metrics = data.get('metrics', {})
        sim_result = data.get('simulation_result', {})

        return {
            'composition_number': metrics.get('composition_number', 0),
            'event_throughput': metrics.get('event_throughput', 0.0),
            'wall_time_per_event': metrics.get('wall_time_per_event', 0.0),
            'cpu_time_per_event': metrics.get('cpu_time_per_event', 0.0),
            'total_write_remote_mb_per_event': metrics.get('total_write_remote_mb_per_event', 0.0),
            'total_read_remote_mb_per_event': metrics.get('total_read_remote_mb_per_event', 0.0),
            'network_transfer_mb_per_event': metrics.get('network_transfer_mb_per_event', 0.0),
            'cpu_utilization': metrics.get('cpu_utilization', 0.0),
            'memory_occupancy': metrics.get('memory_occupancy', 0.0),
            'total_groups': metrics.get('total_groups', 0),
            'failure_rate': sim_result.get('failure_rate', 0.0),
            'overhead_enabled': sim_result.get('overhead_enabled', True),
            'file_path': file_path
        }
    except Exception as e:
        print(f"  Warning: Failed to load {file_path}: {e}")
        return None


def collect_data_from_workflow_types(base_path: str,
                                     workflow_types: List[str],
                                     target_job_length: str,
                                     failure_rate: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Collect simulation data from multiple workflow types.

    Reads simulation result JSON files (*.json) in each workflow type directory.

    Args:
        base_path: Base path to results directory (e.g., 'results/sim/others')
        workflow_types: List of workflow types (e.g., ['case1_real', 'case2_homo', 'case3_hetero'])
        target_job_length: Target job length (e.g., '12h')
        failure_rate: Failure rate directory (e.g., 'fr0')

    Returns:
        Dictionary mapping workflow_type to composition_number to metrics
    """
    # Dictionary: workflow_type -> composition_number -> metrics
    data_by_workflow: Dict[str, Dict[int, Dict[str, Any]]] = {}

    print(f"Collecting data from workflow types: {', '.join(workflow_types)}")
    print(f"Target job length: {target_job_length}, Failure rate: {failure_rate}")

    for workflow_type in workflow_types:
        workflow_dir = Path(base_path) / workflow_type / target_job_length / failure_rate
        
        if not workflow_dir.exists():
            print(f"  Warning: Directory {workflow_dir} not found, skipping {workflow_type}")
            continue

        # Find simulation result JSON files
        json_files = list(workflow_dir.glob("*.json"))
        print(f"  Processing {workflow_type}: {len(json_files)} files found")

        data_by_composition: Dict[int, Dict[str, Any]] = {}
        
        for json_file in sorted(json_files):
            metrics = load_simulation_data(str(json_file))
            if metrics:
                comp_num = metrics['composition_number']
                data_by_composition[comp_num] = metrics

        if data_by_composition:
            data_by_workflow[workflow_type] = data_by_composition

    return data_by_workflow


def identify_best_hybrid(data_by_composition: Dict[int, Dict[str, Any]],
                        verbose: bool = False) -> Optional[int]:
    """Identify the best hybrid construction (2-15) for a workflow type.

    Uses event_throughput as the primary metric, with network_transfer_mb_per_event
    as a tiebreaker (lower network transfer is preferred).

    Args:
        data_by_composition: Dictionary mapping composition_number to metrics
        verbose: If True, print information about ties to stdout

    Returns:
        Composition number of best hybrid, or None if not found
    """
    # Collect all hybrid constructions with their metrics
    hybrid_candidates = []

    for comp_num in range(2, 16):  # Only hybrid constructions (2-15)
        if comp_num not in data_by_composition:
            continue

        metrics = data_by_composition[comp_num]
        hybrid_candidates.append({
            'comp_num': comp_num,
            'throughput': metrics['event_throughput'],
            'network_transfer': metrics['network_transfer_mb_per_event']
        })

    if not hybrid_candidates:
        return None

    # Find maximum throughput
    max_throughput = max(c['throughput'] for c in hybrid_candidates)

    # Find all candidates with maximum throughput
    tied_candidates = [c for c in hybrid_candidates if abs(c['throughput'] - max_throughput) < 1e-10]

    # If there's a tie, use network transfer as tiebreaker (lower is better)
    if len(tied_candidates) > 1:
        if verbose:
            tied_names = [f"Const {c['comp_num']}" for c in tied_candidates]
            print(f"    Tie detected: {', '.join(tied_names)} "
                  f"(throughput: {max_throughput:.6f} evt/s)")
            print(f"      Using network transfer as tiebreaker (lower is better)")

        # Sort by network transfer (ascending - lower is better), then by comp_num for consistency
        tied_candidates.sort(key=lambda x: (x['network_transfer'], x['comp_num']))
        best_comp = tied_candidates[0]['comp_num']

        if verbose:
            print(f"      Selected: Const {best_comp} "
                  f"(network transfer: {tied_candidates[0]['network_transfer']:.6f} MB/evt)")
    else:
        best_comp = tied_candidates[0]['comp_num']

    return best_comp


def plot_throughput_comparison(data_by_workflow: Dict[str, Dict[int, Dict[str, Any]]],
                               output_dir: str) -> None:
    """Plot throughput comparison across workflow types.

    Args:
        data_by_workflow: Dictionary mapping workflow_type to composition metrics
        output_dir: Output directory for plots
    """
    print(f"\n==> Creating throughput comparison plot")

    fig, ax = plt.subplots(figsize=(12, 7))

    workflow_types = sorted(data_by_workflow.keys())
    x = np.arange(len(workflow_types))
    width = 0.25

    # Extract data for Const 1, Const 16, and best hybrid
    const1_throughput = []
    const16_throughput = []
    best_hybrid_throughput = []
    best_hybrid_labels = []

    for workflow_type in workflow_types:
        workflow_data = data_by_workflow[workflow_type]
        
        # Const 1
        if 1 in workflow_data:
            const1_throughput.append(workflow_data[1]['event_throughput'])
        else:
            const1_throughput.append(0.0)

        # Const 16
        if 16 in workflow_data:
            const16_throughput.append(workflow_data[16]['event_throughput'])
        else:
            const16_throughput.append(0.0)

        # Best hybrid
        best_hybrid = identify_best_hybrid(workflow_data, verbose=False)
        if best_hybrid and best_hybrid in workflow_data:
            best_hybrid_throughput.append(workflow_data[best_hybrid]['event_throughput'])
            best_hybrid_labels.append(f"Const {best_hybrid}")
        else:
            best_hybrid_throughput.append(0.0)
            best_hybrid_labels.append("N/A")

    bars1 = ax.bar(x - width, const1_throughput, width, label='Const 1 (All Chained)',
                  color='#d62728', alpha=0.8)
    bars2 = ax.bar(x, const16_throughput, width, label='Const 16 (All Independent)',
                  color='#2ca02c', alpha=0.8)
    bars3 = ax.bar(x + width, best_hybrid_throughput, width, label='Best Hybrid',
                  color='#1f77b4', alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    # Add best hybrid labels
    for i, (workflow_type, label) in enumerate(zip(workflow_types, best_hybrid_labels)):
        if best_hybrid_throughput[i] > 0:
            ax.text(i + width, best_hybrid_throughput[i] + max(best_hybrid_throughput) * 0.02,
                   label, ha='center', va='bottom', fontsize=8, style='italic')

    ax.set_xlabel("Workflow Type", fontsize=12)
    ax.set_ylabel("Event Throughput (events/second)", fontsize=12)
    ax.set_title("Throughput Comparison Across Workflow Types", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(workflow_types, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "throughput_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_path}")


def plot_improvement_percentage(data_by_workflow: Dict[str, Dict[int, Dict[str, Any]]],
                                output_dir: str) -> None:
    """Plot event throughput improvement percentage of best hybrid over extremes.

    Calculates percentage improvement as: ((best_hybrid_throughput - extreme_throughput) / extreme_throughput) × 100

    Args:
        data_by_workflow: Dictionary mapping workflow_type to composition metrics
        output_dir: Output directory for plots
    """
    print(f"\n==> Creating improvement percentage plot")

    fig, ax = plt.subplots(figsize=(12, 7))

    workflow_types = sorted(data_by_workflow.keys())
    x = np.arange(len(workflow_types))
    width = 0.35

    improvement_over_const1 = []
    improvement_over_const16 = []

    for workflow_type in workflow_types:
        workflow_data = data_by_workflow[workflow_type]
        
        # Get best hybrid
        best_hybrid = identify_best_hybrid(workflow_data, verbose=False)
        if not best_hybrid or best_hybrid not in workflow_data:
            improvement_over_const1.append(0.0)
            improvement_over_const16.append(0.0)
            continue

        best_throughput = workflow_data[best_hybrid]['event_throughput']
        
        # Improvement over Const 1
        if 1 in workflow_data and workflow_data[1]['event_throughput'] > 0:
            const1_throughput = workflow_data[1]['event_throughput']
            improvement = ((best_throughput - const1_throughput) / const1_throughput) * 100
            improvement_over_const1.append(improvement)
        else:
            improvement_over_const1.append(0.0)

        # Improvement over Const 16
        if 16 in workflow_data and workflow_data[16]['event_throughput'] > 0:
            const16_throughput = workflow_data[16]['event_throughput']
            improvement = ((best_throughput - const16_throughput) / const16_throughput) * 100
            improvement_over_const16.append(improvement)
        else:
            improvement_over_const16.append(0.0)

    bars1 = ax.bar(x - width/2, improvement_over_const1, width, 
                  label='Over Const 1', color='#d62728', alpha=0.8)
    bars2 = ax.bar(x + width/2, improvement_over_const16, width,
                  label='Over Const 16', color='#2ca02c', alpha=0.8)

    # Add value labels on bars
    # Always show labels, even for zero or very small values
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            # For zero or near-zero values, show label slightly offset from baseline
            if abs(height) < 0.01:
                # Position label at a small offset so it's visible
                label_y = 0.3 if height >= 0 else -0.3
                ax.text(bar.get_x() + bar.get_width()/2., label_y,
                       '0.0%', ha='center', va='bottom' if height >= 0 else 'top', 
                       fontsize=9)
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', 
                       va='bottom' if height > 0 else 'top', fontsize=9)

    ax.set_xlabel("Workflow Type", fontsize=12)
    ax.set_ylabel("Throughput Improvement (%)", fontsize=12)
    ax.set_title("Best Hybrid Throughput Improvement Over Extremes", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(workflow_types, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "throughput_improvement.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_path}")


def plot_network_efficiency_comparison(data_by_workflow: Dict[str, Dict[int, Dict[str, Any]]],
                                      output_dir: str) -> None:
    """Plot network efficiency comparison across workflow types.

    Args:
        data_by_workflow: Dictionary mapping workflow_type to composition metrics
        output_dir: Output directory for plots
    """
    print(f"\n==> Creating network efficiency comparison plot")

    fig, ax = plt.subplots(figsize=(12, 7))

    workflow_types = sorted(data_by_workflow.keys())
    x = np.arange(len(workflow_types))
    width = 0.25

    const1_network = []
    const16_network = []
    best_hybrid_network = []

    for workflow_type in workflow_types:
        workflow_data = data_by_workflow[workflow_type]
        
        # Const 1
        if 1 in workflow_data:
            const1_network.append(workflow_data[1]['network_transfer_mb_per_event'])
        else:
            const1_network.append(0.0)

        # Const 16
        if 16 in workflow_data:
            const16_network.append(workflow_data[16]['network_transfer_mb_per_event'])
        else:
            const16_network.append(0.0)

        # Best hybrid
        best_hybrid = identify_best_hybrid(workflow_data, verbose=False)
        if best_hybrid and best_hybrid in workflow_data:
            best_hybrid_network.append(workflow_data[best_hybrid]['network_transfer_mb_per_event'])
        else:
            best_hybrid_network.append(0.0)

    # Get best hybrid labels for each workflow type
    best_hybrid_labels = []
    for workflow_type in workflow_types:
        workflow_data = data_by_workflow[workflow_type]
        best_hybrid = identify_best_hybrid(workflow_data, verbose=False)
        if best_hybrid:
            best_hybrid_labels.append(f"Const {best_hybrid}")
        else:
            best_hybrid_labels.append("N/A")

    bars1 = ax.bar(x - width, const1_network, width, label='Const 1 (All Chained)',
                  color='#d62728', alpha=0.8)
    bars2 = ax.bar(x, const16_network, width, label='Const 16 (All Independent)',
                  color='#2ca02c', alpha=0.8)
    bars3 = ax.bar(x + width, best_hybrid_network, width, label='Best Hybrid',
                  color='#1f77b4', alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # Add best hybrid construction labels on top of best hybrid bars
    for i, (workflow_type, label) in enumerate(zip(workflow_types, best_hybrid_labels)):
        if best_hybrid_network[i] > 0:
            ax.text(i + width, best_hybrid_network[i] + max(best_hybrid_network) * 0.02,
                   label, ha='center', va='bottom', fontsize=8, style='italic')

    ax.set_xlabel("Workflow Type", fontsize=12)
    ax.set_ylabel("Network Transfer per Event (MB)", fontsize=12)
    ax.set_title("Network Efficiency Comparison Across Workflow Types", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(workflow_types, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "network_efficiency_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_path}")


def plot_network_improvement_percentage(data_by_workflow: Dict[str, Dict[int, Dict[str, Any]]],
                                       output_dir: str) -> None:
    """Plot network transfer reduction percentage of best hybrid over extremes.

    Since lower network transfer is better, this shows reduction percentage:
    ((extreme_network - best_hybrid_network) / extreme_network) × 100
    Positive values indicate the hybrid uses less network (better).

    Args:
        data_by_workflow: Dictionary mapping workflow_type to composition metrics
        output_dir: Output directory for plots
    """
    print(f"\n==> Creating network improvement percentage plot")

    fig, ax = plt.subplots(figsize=(12, 7))

    workflow_types = sorted(data_by_workflow.keys())
    x = np.arange(len(workflow_types))
    width = 0.35

    reduction_over_const1 = []
    reduction_over_const16 = []

    for workflow_type in workflow_types:
        workflow_data = data_by_workflow[workflow_type]
        
        # Get best hybrid
        best_hybrid = identify_best_hybrid(workflow_data, verbose=False)
        if not best_hybrid or best_hybrid not in workflow_data:
            reduction_over_const1.append(0.0)
            reduction_over_const16.append(0.0)
            continue

        best_network = workflow_data[best_hybrid]['network_transfer_mb_per_event']
        
        # Reduction over Const 1 (lower network is better, so we calculate reduction)
        if 1 in workflow_data and workflow_data[1]['network_transfer_mb_per_event'] > 0:
            const1_network = workflow_data[1]['network_transfer_mb_per_event']
            reduction = ((const1_network - best_network) / const1_network) * 100
            reduction_over_const1.append(reduction)
        else:
            reduction_over_const1.append(0.0)

        # Reduction over Const 16
        if 16 in workflow_data and workflow_data[16]['network_transfer_mb_per_event'] > 0:
            const16_network = workflow_data[16]['network_transfer_mb_per_event']
            reduction = ((const16_network - best_network) / const16_network) * 100
            reduction_over_const16.append(reduction)
        else:
            reduction_over_const16.append(0.0)

    bars1 = ax.bar(x - width/2, reduction_over_const1, width, 
                  label='Over Const 1', color='#d62728', alpha=0.8)
    bars2 = ax.bar(x + width/2, reduction_over_const16, width,
                  label='Over Const 16', color='#2ca02c', alpha=0.8)

    # Add value labels on bars
    # Always show labels, even for zero or very small values
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            # For zero or near-zero values, show label slightly offset from baseline
            if abs(height) < 0.01:
                # Position label at a small offset so it's visible
                label_y = 0.3 if height >= 0 else -0.3
                ax.text(bar.get_x() + bar.get_width()/2., label_y,
                       '0.0%', ha='center', va='bottom' if height >= 0 else 'top', 
                       fontsize=9)
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', 
                       va='bottom' if height > 0 else 'top', fontsize=9)

    ax.set_xlabel("Workflow Type", fontsize=12)
    ax.set_ylabel("Network Transfer Reduction (%)", fontsize=12)
    ax.set_title("Best Hybrid Network Transfer Reduction Over Extremes", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(workflow_types, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "network_improvement_percentage.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {output_path}")


def generate_summary_table(data_by_workflow: Dict[str, Dict[int, Dict[str, Any]]],
                          output_dir: str) -> pd.DataFrame:
    """Generate summary table with metrics across workflow types.

    Args:
        data_by_workflow: Dictionary mapping workflow_type to composition metrics
        output_dir: Output directory for table

    Returns:
        DataFrame with summary metrics
    """
    print(f"\n==> Generating summary table")

    table_data = []

    for workflow_type in sorted(data_by_workflow.keys()):
        workflow_data = data_by_workflow[workflow_type]
        
        # Get Const 1, Const 16, and best hybrid
        const1_data = workflow_data.get(1)
        const16_data = workflow_data.get(16)
        best_hybrid = identify_best_hybrid(workflow_data, verbose=False)
        best_hybrid_data = workflow_data.get(best_hybrid) if best_hybrid else None

        for comp_num, metrics in [(1, const1_data), (16, const16_data), 
                                  (best_hybrid, best_hybrid_data)]:
            if metrics is None:
                continue

            row = {
                'Workflow_Type': workflow_type,
                'Construction': comp_num,
                'Event_Throughput': metrics['event_throughput'],
                'Wall_Time_Per_Event': metrics['wall_time_per_event'],
                'CPU_Time_Per_Event': metrics['cpu_time_per_event'],
                'Network_Transfer_MB_Per_Event': metrics['network_transfer_mb_per_event'],
                'CPU_Utilization': metrics['cpu_utilization'],
                'Memory_Occupancy': metrics['memory_occupancy'],
                'Total_Groups': metrics['total_groups']
            }
            table_data.append(row)

    df = pd.DataFrame(table_data)

    # Save as CSV
    csv_path = os.path.join(output_dir, "workflow_type_sensitivity_summary.csv")
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  => Saved: {csv_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Analyze workflow construction performance across workflow types'
    )
    parser.add_argument('base_path', type=str,
                       help='Base path to results directory (e.g., results/sim/others)')
    parser.add_argument('target_job_length', type=str,
                       help='Target job length (e.g., 12h)')
    parser.add_argument('failure_rate', type=str,
                       help='Failure rate directory (e.g., fr0)')
    parser.add_argument('--workflow-types', type=str, nargs='+',
                       default=['case1_real', 'case2_homo', 'case3_hetero'],
                       help='Workflow types to analyze (default: case1_real case2_homo case3_hetero)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: results/analysis/workflow_type_sensitivity/{target_job_length}/{failure_rate})')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"results/analysis/workflow_type_sensitivity/{args.target_job_length}/{args.failure_rate}"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Workflow Type Sensitivity Analysis")
    print("="*70)
    print(f"Target Job Length: {args.target_job_length}")
    print(f"Failure Rate: {args.failure_rate}")
    print(f"Workflow Types: {', '.join(args.workflow_types)}")
    print(f"Output Directory: {args.output_dir}")
    print("="*70)

    data_by_workflow = collect_data_from_workflow_types(
        args.base_path,
        args.workflow_types,
        args.target_job_length,
        args.failure_rate
    )

    if not data_by_workflow:
        print("Error: No data collected. Please check directory paths and file availability.")
        return

    print(f"\nCollected data for {len(data_by_workflow)} workflow types")

    print(f"\nIdentifying best hybrid for each workflow type:")
    for workflow_type in sorted(data_by_workflow.keys()):
        best_hybrid = identify_best_hybrid(data_by_workflow[workflow_type], verbose=True)
        if best_hybrid:
            print(f"  {workflow_type}: Const {best_hybrid}")

    plot_throughput_comparison(data_by_workflow, args.output_dir)
    plot_improvement_percentage(data_by_workflow, args.output_dir)
    plot_network_efficiency_comparison(data_by_workflow, args.output_dir)
    plot_network_improvement_percentage(data_by_workflow, args.output_dir)
    generate_summary_table(data_by_workflow, args.output_dir)

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
