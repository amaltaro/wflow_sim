"""
Unit tests for the WorkflowMetricsCalculator class.
"""

import json
import pytest
from pathlib import Path
import sys

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from workflow_metrics import WorkflowMetricsCalculator


class TestWorkflowMetricsCalculator:
    """Test cases for WorkflowMetricsCalculator."""
    
    def setup_method(self):
        """Set up test data."""
        self.sample_workflow = {
            "Comments": "Test Workflow - 1 group",
            "NumTasks": 3,
            "RequestNumEvents": 1000000,
            "Taskset1": {
                "KeepOutput": False,
                "Memory": 2000,
                "Multicore": 1,
                "RequiresGPU": "forbidden",
                "ScramArch": ["el9_amd64_gcc11"],
                "SizePerEvent": 200,
                "TimePerEvent": 10,
                "GroupName": "group_1",
                "GroupInputEvents": 1000
            },
            "Taskset2": {
                "KeepOutput": True,
                "Memory": 4000,
                "Multicore": 2,
                "RequiresGPU": "forbidden",
                "ScramArch": ["el9_amd64_gcc11"],
                "SizePerEvent": 300,
                "TimePerEvent": 20,
                "InputTaskset": "Taskset1",
                "GroupName": "group_1",
                "GroupInputEvents": 1000
            },
            "Taskset3": {
                "KeepOutput": True,
                "Memory": 3000,
                "Multicore": 2,
                "RequiresGPU": "forbidden",
                "ScramArch": ["el9_amd64_gcc11"],
                "SizePerEvent": 50,
                "TimePerEvent": 10,
                "InputTaskset": "Taskset2",
                "GroupName": "group_1",
                "GroupInputEvents": 1000
            },
            "CompositionNumber": 1
        }
    
    def test_initialization(self):
        """Test calculator initialization."""
        calculator = WorkflowMetricsCalculator(self.sample_workflow)
        assert calculator.workflow_data == self.sample_workflow
        assert calculator.metrics is None
    
    def test_calculate_total_tasksets(self):
        """Test total tasksets calculation."""
        calculator = WorkflowMetricsCalculator(self.sample_workflow)
        total_tasksets = calculator._calculate_total_tasksets()
        assert total_tasksets == 3
    
    def test_calculate_total_groups(self):
        """Test total groups calculation."""
        calculator = WorkflowMetricsCalculator(self.sample_workflow)
        total_groups = calculator._calculate_total_groups()
        assert total_groups == 1
    
    def test_calculate_total_jobs(self):
        """Test total jobs calculation."""
        calculator = WorkflowMetricsCalculator(self.sample_workflow)
        total_jobs = calculator._calculate_total_jobs()
        # Should be 1000 jobs (1000000 requested / 1000 group input events)
        assert total_jobs == 1000
    
    def test_calculate_metrics(self):
        """Test complete metrics calculation."""
        calculator = WorkflowMetricsCalculator(self.sample_workflow)
        metrics = calculator.calculate_metrics()
        
        assert metrics.workflow_id == "unknown"  # Default when not specified
        assert metrics.composition_number == 1
        assert metrics.total_tasksets == 3
        assert metrics.total_groups == 1
        assert metrics.total_jobs == 1000
        assert metrics.total_execution_time > 0
        assert metrics.resource_efficiency >= 0
        assert metrics.throughput >= 0
        assert metrics.success_rate == 1.0
    
    def test_get_groups_info(self):
        """Test group information extraction."""
        calculator = WorkflowMetricsCalculator(self.sample_workflow)
        groups = calculator._get_groups_info()
        
        assert "group_1" in groups
        assert groups["group_1"]["GroupInputEvents"] == 1000
        assert len(groups["group_1"]["tasksets"]) == 3
    
    def test_get_metrics_summary(self):
        """Test metrics summary generation."""
        calculator = WorkflowMetricsCalculator(self.sample_workflow)
        calculator.calculate_metrics()
        summary = calculator.get_metrics_summary()
        
        required_keys = [
            'workflow_id', 'total_tasksets', 'total_groups', 'total_jobs',
            'execution_time', 'resource_efficiency', 'throughput', 'success_rate'
        ]
        
        for key in required_keys:
            assert key in summary
    
    def test_print_metrics(self, capsys):
        """Test metrics printing."""
        calculator = WorkflowMetricsCalculator(self.sample_workflow)
        calculator.calculate_metrics()
        calculator.print_metrics()
        
        captured = capsys.readouterr()
        assert "WORKFLOW EXECUTION METRICS" in captured.out
        assert "Total Tasksets: 3" in captured.out
        assert "Total Groups: 1" in captured.out
    
    def test_write_metrics_to_file(self, tmp_path):
        """Test writing metrics to file."""
        calculator = WorkflowMetricsCalculator(self.sample_workflow)
        calculator.calculate_metrics()
        
        output_file = tmp_path / "test_metrics.json"
        calculator.write_metrics_to_file(output_file)
        
        assert output_file.exists()
        
        # Verify file content
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        assert data['total_tasksets'] == 3
        assert data['total_groups'] == 1
        assert data['total_jobs'] == 1000


if __name__ == "__main__":
    pytest.main([__file__])
