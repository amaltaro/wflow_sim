#!/usr/bin/env python3
"""
Workflow Simulation Visualization Examples

This script demonstrates various visualization approaches for workflow simulation metrics.
"""

import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import argparse

class WorkflowVisualizer:
    """Class for creating visualizations from workflow simulation results."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.data = self._load_all_results()
    
    def _load_all_results(self):
        """Load all JSON results into a pandas DataFrame."""
        all_results = []
        
        for json_file in self.results_dir.rglob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                # Extract metrics
                metrics = data.get('metrics', {})
                
                # Add file metadata
                metrics['file_path'] = str(json_file)
                metrics['workflow_name'] = json_file.stem
                metrics['category'] = json_file.parent.name
                
                all_results.append(metrics)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        return pd.DataFrame(all_results)
    
    def create_resource_utilization_scatter(self):
        """Create CPU vs Memory utilization scatter plot."""
        # Sort by composition number for consistent legend ordering
        sorted_data = self.data.sort_values('composition_number')
        
        fig = px.scatter(
            sorted_data, 
            x='cpu_utilization', 
            y='memory_occupancy',
            color='composition_number',
            size='total_events',
            hover_data=['total_cpu_cores_used', 'total_memory_used_mb', 'event_throughput', 'workflow_name'],
            title="CPU Utilization vs Memory Occupancy",
            labels={
                'cpu_utilization': 'CPU Utilization',
                'memory_occupancy': 'Memory Occupancy',
                'composition_number': 'Composition'
            }
        )
        fig.update_layout(
            xaxis_title="CPU Utilization",
            yaxis_title="Memory Occupancy",
            showlegend=True
        )
        return fig
    
    def create_io_pattern_analysis(self):
        """Create I/O pattern analysis visualization."""
        # Sort by composition number for consistent ordering
        sorted_data = self.data.sort_values('composition_number').copy()
        
        # Calculate I/O ratios
        sorted_data['local_io_ratio'] = (
            sorted_data['total_write_local_mb'] + sorted_data['total_read_local_mb']
        ) / (sorted_data['total_write_local_mb'] + sorted_data['total_write_remote_mb'] + 
             sorted_data['total_read_remote_mb'] + sorted_data['total_read_local_mb'])
        
        sorted_data['remote_io_ratio'] = (
            sorted_data['total_write_remote_mb'] + sorted_data['total_read_remote_mb']
        ) / (sorted_data['total_write_local_mb'] + sorted_data['total_write_remote_mb'] + 
             sorted_data['total_read_remote_mb'] + sorted_data['total_read_local_mb'])
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('I/O Pattern Distribution', 'Network Transfer vs Local I/O',
                           'I/O per Event Analysis', 'Total I/O Breakdown'),
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Pie chart for I/O patterns
        fig.add_trace(
            go.Pie(
                labels=['Local I/O', 'Remote I/O'],
                values=[sorted_data['local_io_ratio'].mean(), sorted_data['remote_io_ratio'].mean()],
                name="I/O Pattern"
            ),
            row=1, col=1
        )
        
        # Scatter plot: Network vs Local I/O
        fig.add_trace(
            go.Scatter(
                x=sorted_data['total_write_local_mb'] + sorted_data['total_read_local_mb'],
                y=sorted_data['total_network_transfer_mb'],
                mode='markers',
                text=sorted_data['workflow_name'],
                name="Network vs Local I/O"
            ),
            row=1, col=2
        )
        
        # Bar chart: I/O per event
        fig.add_trace(
            go.Bar(
                x=sorted_data['composition_number'],
                y=sorted_data['total_write_local_mb_per_event'] + sorted_data['total_read_local_mb_per_event'],
                name="Local I/O per Event"
            ),
            row=2, col=1
        )
        
        # Bar chart: Total I/O breakdown
        fig.add_trace(
            go.Bar(
                x=sorted_data['composition_number'],
                y=sorted_data['total_write_local_mb'],
                name="Write Local"
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True, title_text="I/O Pattern Analysis")
        return fig
    
    def create_performance_comparison(self):
        """Create performance comparison dashboard."""
        # Sort by composition number
        sorted_data = self.data.sort_values('composition_number')
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Event Throughput Comparison', 'Execution Time Analysis',
                           'Resource Efficiency', 'Cost per Event'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Event throughput
        fig.add_trace(
            go.Bar(
                x=sorted_data['composition_number'],
                y=sorted_data['event_throughput'],
                name="Event Throughput",
                text=sorted_data['workflow_name'],
                hovertemplate='Comp: %{text}<br>Throughput: %{y:.6f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Execution time breakdown
        fig.add_trace(
            go.Bar(
                x=sorted_data['composition_number'],
                y=sorted_data['total_wall_time'],
                name="Wall Time",
                text=sorted_data['workflow_name'],
                hovertemplate='Comp: %{text}<br>Wall Time: %{y:.2f}s<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Resource efficiency scatter
        fig.add_trace(
            go.Scatter(
                x=sorted_data['cpu_utilization'],
                y=sorted_data['memory_occupancy'],
                mode='markers',
                text=sorted_data['workflow_name'],
                name="Resource Efficiency"
            ),
            row=2, col=1
        )
        
        # Cost per event
        fig.add_trace(
            go.Scatter(
                x=sorted_data['cpu_cores_per_event'],
                y=sorted_data['memory_mb_per_event'],
                mode='markers',
                text=sorted_data['workflow_name'],
                name="Cost per Event"
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True, title_text="Performance Comparison Dashboard")
        return fig
    
    def create_cost_analysis(self):
        """Create cost analysis visualization."""
        # Sort by composition number
        sorted_data = self.data.sort_values('composition_number').copy()
        
        # Calculate cost metrics
        sorted_data['total_cost_score'] = (
            sorted_data['cpu_cores_per_event'] * 0.1 +  # CPU cost factor
            sorted_data['memory_mb_per_event'] * 0.001 +  # Memory cost factor
            sorted_data['network_transfer_mb_per_event'] * 0.01  # Network cost factor
        )
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cost per Event Breakdown', 'Cost vs Performance Trade-off',
                           'Resource Cost Distribution', 'ROI Analysis'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "pie"}, {"type": "scatter"}]]
        )
        
        # Cost breakdown
        fig.add_trace(
            go.Bar(
                x=sorted_data['composition_number'],
                y=sorted_data['total_cost_score'],
                name="Total Cost Score",
                text=sorted_data['workflow_name'],
                hovertemplate='Comp: %{text}<br>Cost: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Cost vs Performance
        fig.add_trace(
            go.Scatter(
                x=sorted_data['total_cost_score'],
                y=sorted_data['event_throughput'],
                mode='markers',
                text=sorted_data['workflow_name'],
                name="Cost vs Performance"
            ),
            row=1, col=2
        )
        
        # Resource cost distribution
        fig.add_trace(
            go.Pie(
                labels=['CPU', 'Memory', 'Network'],
                values=[
                    sorted_data['cpu_cores_per_event'].mean() * 0.1,
                    sorted_data['memory_mb_per_event'].mean() * 0.001,
                    sorted_data['network_transfer_mb_per_event'].mean() * 0.01
                ],
                name="Resource Cost Distribution"
            ),
            row=2, col=1
        )
        
        # ROI analysis
        fig.add_trace(
            go.Scatter(
                x=sorted_data['total_wall_time'],
                y=sorted_data['event_throughput'],
                mode='markers',
                text=sorted_data['workflow_name'],
                name="ROI Analysis"
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True, title_text="Cost Analysis Dashboard")
        return fig
    
    def create_trade_off_matrix(self):
        """Create trade-off analysis matrix."""
        # Create correlation matrix
        numeric_cols = [
            'event_throughput', 'cpu_utilization', 'memory_occupancy',
            'total_wall_time', 'cpu_cores_per_event', 'memory_mb_per_event',
            'network_transfer_mb_per_event'
        ]
        
        corr_matrix = self.data[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title="Trade-off Analysis Matrix",
            xaxis_title="Metrics",
            yaxis_title="Metrics"
        )
        
        return fig
    
    def create_job_scaling_analysis(self):
        """Create job scaling efficiency visualization using metrics only."""
        # Sort by composition number
        sorted_data = self.data.sort_values('composition_number').copy()
        
        # Calculate derived metrics
        sorted_data['jobs_per_100k_events'] = (sorted_data['total_jobs'] / sorted_data['total_events']) * 100000
        sorted_data['jobs_per_group'] = sorted_data['total_jobs'] / sorted_data['total_groups']
        sorted_data['cpu_efficiency'] = sorted_data['total_cpu_used_time'] / sorted_data['total_cpu_allocated_time']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Job Scaling Efficiency', 'CPU Efficiency (Used/Allocated)',
                           'Jobs per Group Distribution', 'Composition Summary'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Jobs per 100k events scatter
        fig.add_trace(
            go.Scatter(
                x=sorted_data['total_events'],
                y=sorted_data['jobs_per_100k_events'],
                mode='markers',
                text=sorted_data['workflow_name'],
                marker=dict(
                    size=10,
                    color=sorted_data['composition_number'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Composition", x=0.46)
                ),
                name="Job Scaling",
                hovertemplate='Composition: %{text}<br>Events: %{x:,.0f}<br>Jobs/100k: %{y:.1f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # CPU efficiency
        fig.add_trace(
            go.Bar(
                x=sorted_data['composition_number'],
                y=sorted_data['cpu_efficiency'],
                text=[f'{val:.2f}' for val in sorted_data['cpu_efficiency']],
                textposition='outside',
                marker=dict(
                    color=sorted_data['cpu_efficiency'],
                    colorscale='RdYlGn',
                    showscale=True,
                    cmin=0.5,
                    cmax=1.0,
                    colorbar=dict(title="Efficiency", x=1.02)
                ),
                name="CPU Efficiency",
                hovertemplate='Comp: %{x}<br>Efficiency: %{y:.2f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Jobs per group distribution
        fig.add_trace(
            go.Bar(
                x=sorted_data['composition_number'],
                y=sorted_data['jobs_per_group'],
                name="Jobs per Group",
                marker=dict(color='lightblue'),
                text=sorted_data['workflow_name'],
                hovertemplate='Comp: %{text}<br>Jobs/Group: %{y:.1f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Composition summary (total resources)
        fig.add_trace(
            go.Bar(
                x=sorted_data['composition_number'],
                y=sorted_data['total_jobs'],
                text=[f'{int(val)}' for val in sorted_data['total_jobs']],
                textposition='outside',
                name="Total Jobs",
                marker=dict(color='steelblue'),
                hovertemplate='Comp: %{x}<br>Total Jobs: %{y}<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=900,
            showlegend=True,
            title_text="Job Scaling and Resource Efficiency Analysis"
        )
        fig.update_xaxes(title_text="Total Events", row=1, col=1)
        fig.update_yaxes(title_text="Jobs per 100k Events", row=1, col=1)
        fig.update_yaxes(title_text="Efficiency Ratio", row=1, col=2)
        fig.update_xaxes(title_text="Workflow Composition", row=2, col=1)
        fig.update_yaxes(title_text="Jobs per Group", row=2, col=1)
        fig.update_xaxes(title_text="Workflow Composition", row=2, col=2)
        fig.update_yaxes(title_text="Total Jobs", row=2, col=2)
        
        return fig
    
    def create_composition_comparison(self):
        """Create comprehensive composition comparison dashboard."""
        # Sort by composition number for consistent ordering
        sorted_data = self.data.sort_values('composition_number').copy()
        
        # Calculate derived metrics
        sorted_data['cpu_efficiency'] = sorted_data['total_cpu_used_time'] / sorted_data['total_cpu_allocated_time']
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Total Events vs Composition', 'Wall Time Distribution',
                           'Network Transfer Comparison', 'CPU Time Breakdown',
                           'Memory Usage per Composition', 'Overall Performance'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Total events
        fig.add_trace(
            go.Bar(
                x=sorted_data['composition_number'],
                y=sorted_data['total_events'],
                name="Total Events",
                marker=dict(color='darkblue')
            ),
            row=1, col=1
        )
        
        # Wall time
        fig.add_trace(
            go.Bar(
                x=sorted_data['composition_number'],
                y=sorted_data['total_wall_time'] / 3600,  # Convert to hours
                name="Wall Time (hrs)",
                marker=dict(color='orange')
            ),
            row=1, col=2
        )
        
        # Network transfer
        fig.add_trace(
            go.Bar(
                x=sorted_data['composition_number'],
                y=sorted_data['total_network_transfer_mb'] / 1000,  # Convert to GB
                name="Network (GB)",
                marker=dict(color='green')
            ),
            row=2, col=1
        )
        
        # CPU time breakdown
        fig.add_trace(
            go.Bar(
                x=sorted_data['composition_number'],
                y=sorted_data['total_cpu_used_time'] / 1e6,  # Convert to millions
                name="CPU Used (10^6s)",
                marker=dict(color='red')
            ),
            row=2, col=2
        )
        
        # Memory usage
        fig.add_trace(
            go.Bar(
                x=sorted_data['composition_number'],
                y=sorted_data['total_memory_used_mb'] / 1000,  # Convert to GB
                name="Memory (GB)",
                marker=dict(color='purple')
            ),
            row=3, col=1
        )
        
        # Overall performance scatter
        fig.add_trace(
            go.Scatter(
                x=sorted_data['event_throughput'],
                y=sorted_data['total_jobs'],
                mode='markers+text',
                text=[str(c) for c in sorted_data['composition_number']],
                textposition="top center",
                marker=dict(
                    size=15,
                    color=sorted_data['cpu_efficiency'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="CPU Efficiency", x=1.02)
                ),
                name="Performance",
                hovertemplate='Comp: %{text}<br>Throughput: %{x:.4f}<br>Jobs: %{y}<extra></extra>'
            ),
            row=3, col=2
        )
        
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text="Comprehensive Composition Comparison Dashboard"
        )
        
        # Update all axis titles
        fig.update_xaxes(title_text="Composition Number", row=1, col=1)
        fig.update_yaxes(title_text="Total Events", row=1, col=1)
        fig.update_xaxes(title_text="Composition Number", row=1, col=2)
        fig.update_yaxes(title_text="Wall Time (hours)", row=1, col=2)
        fig.update_xaxes(title_text="Composition Number", row=2, col=1)
        fig.update_yaxes(title_text="Network Transfer (GB)", row=2, col=1)
        fig.update_xaxes(title_text="Composition Number", row=2, col=2)
        fig.update_yaxes(title_text="CPU Used Time (10^6s)", row=2, col=2)
        fig.update_xaxes(title_text="Composition Number", row=3, col=1)
        fig.update_yaxes(title_text="Memory Used (GB)", row=3, col=1)
        fig.update_xaxes(title_text="Event Throughput", row=3, col=2)
        fig.update_yaxes(title_text="Total Jobs", row=3, col=2)
        
        return fig

def main():
    """Main function to demonstrate visualizations."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate visualizations from workflow simulation results'
    )
    parser.add_argument(
        'data_path',
        type=str,
        nargs='?',
        default='results',
        help='Path to directory containing simulation result JSON files (default: results)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Directory to save generated HTML visualizations (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Validate data path
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data path '{data_path}' does not exist.")
        return 1
    
    if not data_path.is_dir():
        print(f"Error: Data path '{data_path}' is not a directory.")
        return 1
    
    # Find all JSON files
    json_files = list(data_path.rglob("*.json"))
    if not json_files:
        print(f"No JSON files found in '{data_path}'.")
        return 1
    
    print(f"Found {len(json_files)} JSON files in '{data_path}'.")
    print("Parsing metrics...")
    
    # Create visualizer with specified data path
    visualizer = WorkflowVisualizer(results_dir=str(data_path))
    
    print(f"Successfully loaded {len(visualizer.data)} simulation results.")
    print("Creating visualizations...")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create and save visualizations
    visualizations = {
        'resource_utilization': visualizer.create_resource_utilization_scatter(),
        'io_patterns': visualizer.create_io_pattern_analysis(),
        'performance_comparison': visualizer.create_performance_comparison(),
        'cost_analysis': visualizer.create_cost_analysis(),
        'trade_off_matrix': visualizer.create_trade_off_matrix(),
        'job_scaling_analysis': visualizer.create_job_scaling_analysis(),
        'composition_comparison': visualizer.create_composition_comparison()
    }
    
    # Save as HTML files
    for name, fig in visualizations.items():
        output_file = output_dir / f"{name}_analysis.html"
        fig.write_html(str(output_file))
        print(f"Saved {output_file}")
    
    print("\nAll visualizations created successfully!")
    print(f"Visualizations saved to: {output_dir.absolute()}")
    print("\nTo view the visualizations, open the generated HTML files in your browser.")

if __name__ == "__main__":
    exit(main() or 0)
