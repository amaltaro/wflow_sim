"""Tests for workflow_visualization stacked-total axis styling and volume units."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent / "scripts"))
from workflow_visualization import (
    _annotate_construction_scatter_labels,
    _comparison_xtick_labels,
    _io_patterns_horizontal_legend_below,
    _resource_util_panel_center_banner,
    _stacked_total_volume_scale_and_unit_from_max_mb,
    _style_stacked_total_data_volume_axis,
    _tight_axis_limits,
)


def test_comparison_xtick_labels_custom() -> None:
    assert _comparison_xtick_labels(2, ["StepChain", "TaskChain"]) == [
        "StepChain",
        "TaskChain",
    ]
    assert _comparison_xtick_labels(3, None) == ["1", "2", "3"]


def test_annotate_construction_scatter_labels_adds_one_per_point() -> None:
    _, ax = plt.subplots()
    xs = np.array([1.0, 2.0])
    ys = np.array([3.0, 4.0])
    _annotate_construction_scatter_labels(ax, xs, ys, label_start=1)
    assert len(ax.texts) == 2
    assert {t.get_text() for t in ax.texts} == {"1", "2"}
    plt.close()


def test_io_patterns_horizontal_legend_below_uses_figure_legend() -> None:
    fig, (ax_top, ax_bot) = plt.subplots(2, 1)
    ax_top.bar([0], [1], label="A")
    ax_bot.bar([0], [2], label="B")
    _io_patterns_horizontal_legend_below(fig, ax_bot, ncol=2)
    assert fig.legends
    assert len(fig.legends[0].get_texts()) == 1
    assert fig.legends[0].get_texts()[0].get_text() == "B"
    plt.close(fig)


def test_resource_util_panel_center_banner_adds_text() -> None:
    _, ax = plt.subplots()
    _resource_util_panel_center_banner(ax, "Network", "#9467bd")
    assert len(ax.texts) == 1
    assert ax.texts[0].get_text() == "Network"
    plt.close()


def test_volume_unit_mb_small_totals() -> None:
    s, u = _stacked_total_volume_scale_and_unit_from_max_mb(100.0)
    assert u == "MB" and s == 1.0


def test_volume_unit_gb() -> None:
    s, u = _stacked_total_volume_scale_and_unit_from_max_mb(2000.0)
    assert u == "GB"
    assert abs(s - 1.0 / 1024.0) < 1e-15


def test_volume_unit_tb() -> None:
    s, u = _stacked_total_volume_scale_and_unit_from_max_mb(1024.0**2 + 1.0)
    assert u == "TB"
    assert abs(s - 1.0 / 1024.0**2) < 1e-20


def test_volume_unit_pb() -> None:
    s, u = _stacked_total_volume_scale_and_unit_from_max_mb(1024.0**3 + 1.0)
    assert u == "PB"
    assert abs(s - 1.0 / 1024.0**3) < 1e-25


def test_volume_unit_pb_capped_past_one_pib_in_mb() -> None:
    """Beyond 1024^3 MB we stay on PB scale (largest supported label)."""
    s, u = _stacked_total_volume_scale_and_unit_from_max_mb(1024.0**5)
    assert u == "PB"
    assert abs(s - 1.0 / 1024.0**3) < 1e-30


def test_style_stacked_total_data_volume_axis_sets_minor_locator() -> None:
    _, ax = plt.subplots()
    _style_stacked_total_data_volume_axis(ax)
    assert ax.yaxis.get_minor_locator() is not None
    plt.close()


def test_tight_axis_limits_empty_defaults() -> None:
    lo, hi = _tight_axis_limits(np.array([]))
    assert lo == 0.0 and hi == 1.0


def test_tight_axis_limits_non_negative_clamp() -> None:
    lo, hi = _tight_axis_limits(np.array([0.1, 0.2, 0.15]), clamp_non_negative=True)
    assert lo >= 0.0
    assert lo < hi


def test_tight_axis_limits_throughput_can_start_above_zero() -> None:
    """Scatter x-axis should not force 0 when all throughputs are clustered away from zero."""
    v = np.array([0.018, 0.022, 0.019])
    lo, hi = _tight_axis_limits(v, clamp_non_negative=False)
    assert lo > 0.0
    assert hi > lo


def test_tight_axis_limits_constant_series() -> None:
    lo, hi = _tight_axis_limits(np.array([5.0, 5.0, 5.0]))
    assert lo < hi
    assert 5.0 >= lo and 5.0 <= hi
