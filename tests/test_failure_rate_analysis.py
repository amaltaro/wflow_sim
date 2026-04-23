"""Tests for scripts/failure_rate_analysis.py composition logic."""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
import composition_extremes as ce  # noqa: E402
import failure_rate_analysis as fra  # noqa: E402


def _fr_row(
    composition: int,
    failure_rate: float,
    tp: float,
    net: float,
    total_groups: int = 0,
) -> dict:
    return {
        "composition_number": composition,
        "failure_rate": failure_rate,
        "event_throughput": tp,
        "network_transfer_mb_per_event": net,
        "total_groups": total_groups,
    }


class TestCompositionExtremes:
    def test_five_compositions_ties_on_min_and_max_total_groups(self) -> None:
        # 1,2: fewest groups; 3: mid; 4,5: most groups
        g_min, g_max = 2, 7
        g_mid = 4
        data = {
            1: [_fr_row(1, 0.0, 1.0, 1.0, total_groups=g_min)],
            2: [_fr_row(2, 0.0, 1.0, 1.0, total_groups=g_min)],
            3: [_fr_row(3, 0.0, 1.0, 1.0, total_groups=g_mid)],
            4: [_fr_row(4, 0.0, 1.0, 1.0, total_groups=g_max)],
            5: [_fr_row(5, 0.0, 1.0, 1.0, total_groups=g_max)],
        }
        # Smallest total_groups: tie 1,2 -> lowest id = 1; largest: tie 4,5 -> highest id = 5
        assert ce.composition_extremes(data) == (1, 5)

    def test_tie_falls_back_to_index_when_equal_total_groups(self) -> None:
        data = {1: [], 5: [], 48: []}
        assert ce.composition_extremes(data) == (1, 48)

    def test_by_total_groups_prefers_min_and_max_counts(self) -> None:
        data = {
            1: [_fr_row(1, 0.0, 1.0, 1.0, total_groups=2)],
            2: [_fr_row(2, 0.0, 1.0, 1.0, total_groups=3)],
            3: [_fr_row(3, 0.0, 1.0, 1.0, total_groups=2)],
        }
        assert ce.composition_extremes(data) == (1, 2)

    def test_tie_on_max_total_groups_uses_largest_id(self) -> None:
        data2 = {
            1: [_fr_row(1, 0.0, 1.0, 1.0, total_groups=1)],
            2: [_fr_row(2, 0.0, 1.0, 1.0, total_groups=5)],
            3: [_fr_row(3, 0.0, 1.0, 1.0, total_groups=5)],
        }
        assert ce.composition_extremes(data2) == (1, 3)

    def test_single_key(self) -> None:
        data = {7: []}
        assert ce.composition_extremes(data) == (7, 7)

    def test_fr0_trumps_higher_fr_for_total_groups(self) -> None:
        data = {
            1: [
                _fr_row(1, 25.0, 1.0, 1.0, total_groups=9),
                _fr_row(1, 0.0, 1.0, 1.0, total_groups=2),
            ],
            2: [_fr_row(2, 0.0, 1.0, 1.0, total_groups=2)],
        }
        assert ce.composition_extremes(data) == (1, 2)  # both min 2, tie: ids 1 and 2

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            ce.composition_extremes({})


class TestIdentifyBestHybrid:
    def test_middles_only_excludes_extremes(self) -> None:
        data = {
            1: [_fr_row(1, 0.0, 1.0, 1.0), _fr_row(1, 5.0, 1.0, 1.0)],
            2: [_fr_row(2, 0.0, 3.0, 2.0), _fr_row(2, 5.0, 3.0, 2.0)],
            10: [_fr_row(10, 0.0, 2.0, 1.0), _fr_row(10, 5.0, 2.0, 1.0)],
            16: [_fr_row(16, 0.0, 4.0, 0.5), _fr_row(16, 5.0, 4.0, 0.5)],
        }
        assert (
            fra.identify_best_hybrid(data, 0.0, grouped_comp=1, independent_comp=16) == 2
        )
        # Const 4 not in data
        data2 = {1: data[1], 16: data[16], 2: data[2]}
        assert fra.identify_best_hybrid(data2, 0.0, 1, 16) == 2

    def test_fork_48_sweep_uses_2_to_47(self) -> None:
        data: dict = {}
        for c in (1, 2, 47, 48):
            data[c] = [_fr_row(c, 0.0, float(c), 1.0)]
        # Best throughput among 2..47 is 47
        assert (
            fra.identify_best_hybrid(
                data, 0.0, grouped_comp=1, independent_comp=48
            ) == 47
        )

    def test_no_middles_returns_none(self) -> None:
        data = {1: [_fr_row(1, 0.0, 1.0, 1.0)], 2: [_fr_row(2, 0.0, 1.0, 1.0)]}
        assert (
            fra.identify_best_hybrid(
                data, 0.0, grouped_comp=1, independent_comp=2
            ) is None
        )

    def test_single_composition_extreme_returns_none(self) -> None:
        data = {5: [_fr_row(5, 0.0, 1.0, 1.0)]}
        assert (
            fra.identify_best_hybrid(
                data, 0.0, grouped_comp=5, independent_comp=5
            ) is None
        )
