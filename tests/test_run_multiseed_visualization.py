"""Unit tests for scripts/run_multiseed_visualization.py helpers."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_multiseed_visualization import (  # noqa: E402
    aggregate_by_composition,
    discover_result_files,
    mean_and_sem,
    should_draw_error_bars,
)


class TestMeanAndSem:
    """Tests for mean / SEM calculation."""

    def test_single_value_sem_zero(self) -> None:
        """N=1 yields SEM 0."""
        mean, sem = mean_and_sem([10.0])
        assert mean == 10.0
        assert sem == 0.0

    def test_known_sem(self) -> None:
        """SEM matches sample std / sqrt(N)."""
        values = [1.0, 3.0, 5.0]
        mean, sem = mean_and_sem(values)
        assert mean == pytest.approx(3.0)
        expected = float(np.std(values, ddof=1) / np.sqrt(3))
        assert sem == pytest.approx(expected)


class TestShouldDrawErrorBars:
    """Tests for error-bar gating."""

    def test_fr_positive_and_multiple_runs(self) -> None:
        assert should_draw_error_bars(5.0, 10) is True

    def test_fr_zero_disables(self) -> None:
        assert should_draw_error_bars(0.0, 10) is False

    def test_single_run_disables(self) -> None:
        assert should_draw_error_bars(5.0, 1) is False


class TestAggregateByComposition:
    """Tests for grouping and aggregation."""

    def test_groups_and_means(self) -> None:
        """Two compositions with two runs each produce sorted means/SEMs."""
        records = [
            {
                "composition_number": 2,
                "cpu_time_per_event": 10.0,
                "cpu_utilization": 0.5,
                "event_throughput": 1.0,
                "total_write_remote_mb_per_event": 0.1,
                "total_turnaround_time": 100.0,
                "network_transfer_mb_per_event": 0.2,
                "memory_occupancy": 0.8,
                "total_cpu_cores_used": 10.0,
                "total_memory_used_mb": 1000.0,
            },
            {
                "composition_number": 2,
                "cpu_time_per_event": 14.0,
                "cpu_utilization": 0.7,
                "event_throughput": 1.0,
                "total_write_remote_mb_per_event": 0.1,
                "total_turnaround_time": 100.0,
                "network_transfer_mb_per_event": 0.2,
                "memory_occupancy": 0.8,
                "total_cpu_cores_used": 10.0,
                "total_memory_used_mb": 1000.0,
            },
            {
                "composition_number": 1,
                "cpu_time_per_event": 20.0,
                "cpu_utilization": 0.9,
                "event_throughput": 2.0,
                "total_write_remote_mb_per_event": 0.2,
                "total_turnaround_time": 200.0,
                "network_transfer_mb_per_event": 0.3,
                "memory_occupancy": 0.7,
                "total_cpu_cores_used": 20.0,
                "total_memory_used_mb": 2000.0,
            },
            {
                "composition_number": 1,
                "cpu_time_per_event": 20.0,
                "cpu_utilization": 0.9,
                "event_throughput": 2.0,
                "total_write_remote_mb_per_event": 0.2,
                "total_turnaround_time": 200.0,
                "network_transfer_mb_per_event": 0.3,
                "memory_occupancy": 0.7,
                "total_cpu_cores_used": 20.0,
                "total_memory_used_mb": 2000.0,
            },
        ]
        rows = aggregate_by_composition(records)
        assert [r["composition_number"] for r in rows] == [1, 2]
        assert rows[0]["n"] == 2
        assert rows[0]["cpu_time_per_event"] == pytest.approx(20.0)
        assert rows[0]["cpu_time_per_event_sem"] == pytest.approx(0.0)
        assert rows[1]["cpu_time_per_event"] == pytest.approx(12.0)
        assert rows[1]["cpu_time_per_event_sem"] > 0.0


class TestDiscoverResultFiles:
    """Tests for seed* discovery."""

    def test_finds_seed_json_skips_campaign(self, tmp_path: Path) -> None:
        """Only seed*/ trees are searched; campaign.json at root is ignored."""
        (tmp_path / "campaign.json").write_text("{}")
        seed0 = tmp_path / "seed0" / "others" / "seq_real" / "12h" / "fr5" / "100MBps"
        seed0.mkdir(parents=True)
        target = seed0 / "seq_real_const_001.json"
        target.write_text(json.dumps({"metrics": {"composition_number": 1}}))
        found = discover_result_files(tmp_path)
        assert found == [target]


if __name__ == "__main__":
    pytest.main([__file__])
