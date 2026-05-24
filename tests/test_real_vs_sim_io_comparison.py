"""Tests for real vs simulated I/O comparison helpers."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from plot_real_vs_sim_io_comparison import (  # noqa: E402
    IoVolumeMetrics,
    REAL_VS_SIM_COMPOSITIONS,
    io_volumes_from_workflow,
    plot_real_vs_sim_io_comparison,
)
from real_workflow_visualization import extract_construction_number  # noqa: E402


def _sample_workflow(
    *,
    pevt: float = 1.0,
    total: float = 1000.0,
) -> dict:
    return {
        "total_read_local_mb_per_event": pevt,
        "total_read_remote_mb_per_event": pevt * 2,
        "total_write_local_mb_per_event": pevt * 3,
        "total_write_remote_mb_per_event": pevt * 4,
        "total_read_local_mb": total,
        "total_read_remote_mb": total * 2,
        "total_write_local_mb": total * 3,
        "total_write_remote_mb": total * 4,
    }


@pytest.mark.parametrize(
    "file_name,expected",
    [
        ("summary_const001.json", 1),
        ("summary_const016.json", 16),
        ("seq_real_const_001.json", 1),
        ("seq_real_const_016.json", 16),
    ],
)
def test_extract_construction_number(file_name, expected):
    assert extract_construction_number(file_name) == expected


def test_io_volumes_from_workflow():
    wf = _sample_workflow(pevt=0.5, total=500.0)
    vol = io_volumes_from_workflow(wf)
    assert vol == IoVolumeMetrics(
        read_local_pevt=0.5,
        read_remote_pevt=1.0,
        write_local_pevt=1.5,
        write_remote_pevt=2.0,
        total_read_local_mb=500.0,
        total_read_remote_mb=1000.0,
        total_write_local_mb=1500.0,
        total_write_remote_mb=2000.0,
    )


def test_plot_real_vs_sim_io_comparison_writes_pngs(tmp_path):
    real = {1: _sample_workflow(), 16: _sample_workflow(pevt=2.0, total=2000.0)}
    sim = {1: _sample_workflow(pevt=1.1, total=1100.0), 16: _sample_workflow(pevt=2.2, total=2200.0)}
    plot_real_vs_sim_io_comparison(
        real,
        sim,
        str(tmp_path),
        constructions=REAL_VS_SIM_COMPOSITIONS,
    )
    assert (tmp_path / "io_patterns_real_vs_sim_local.png").is_file()
    assert (tmp_path / "io_patterns_real_vs_sim_nonlocal.png").is_file()


def test_plot_real_vs_sim_raises_without_overlap():
    with pytest.raises(ValueError, match="No overlapping"):
        plot_real_vs_sim_io_comparison({1: _sample_workflow()}, {}, str(Path("/tmp")))
