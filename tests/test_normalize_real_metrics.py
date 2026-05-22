"""Tests for normalize_real_metrics.py."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from normalize_real_metrics import normalize_real_data_file


def test_normalize_sets_total_events_to_target(tmp_path: Path) -> None:
    inp = tmp_path / "in.json"
    out = tmp_path / "out.json"
    payload = {
        "time_metrics": {
            "total_wallclock_time_with_overhead_sec": 100.0,
            "workflow_turnaround_time_sec": 50.0,
        },
        "cpu_metrics": {"total_cpu_cores_used": 10.0},
        "event_metrics": {
            "total_events": 100_000,
            "cpu_time_per_event_sec": 2.5,
        },
    }
    inp.write_text(json.dumps(payload))
    normalize_real_data_file(inp, out, target_events=1_000_000)

    result = json.loads(out.read_text())
    assert result["event_metrics"]["total_events"] == 1_000_000
    assert result["event_metrics"]["cpu_time_per_event_sec"] == 2.5
    assert result["time_metrics"]["total_wallclock_time_with_overhead_sec"] == pytest.approx(
        1000.0
    )
    assert result["time_metrics"]["workflow_turnaround_time_sec"] == 50.0
    assert result["cpu_metrics"]["total_cpu_cores_used"] == pytest.approx(100.0)


def test_normalize_rejects_zero_events(tmp_path: Path) -> None:
    inp = tmp_path / "in.json"
    out = tmp_path / "out.json"
    inp.write_text(json.dumps({"event_metrics": {"total_events": 0}}))
    with pytest.raises(ValueError, match="total_events"):
        normalize_real_data_file(inp, out)
