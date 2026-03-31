"""Unit tests for plot_construction_groups_overview module."""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent / "scripts"))

from plot_construction_groups_overview import build_group_id_matrix


def test_build_group_id_matrix_happy_path() -> None:
    summary = {
        "template_name": "case1_real",
        "total_compositions": 2,
        "compositions": [
            {
                "composition_number": 1,
                "num_groups": 1,
                "group_details": [{"group_id": "group_0", "tasks": ["Taskset1", "Taskset2"]}],
            },
            {
                "composition_number": 2,
                "num_groups": 2,
                "group_details": [
                    {"group_id": "group_0", "tasks": ["Taskset1"]},
                    {"group_id": "group_1", "tasks": ["Taskset2"]},
                ],
            },
        ],
    }

    matrix = build_group_id_matrix(summary)
    assert matrix.template_name == "case1_real"
    assert matrix.composition_numbers == [1, 2]
    assert matrix.taskset_names == ["Taskset1", "Taskset2"]
    assert matrix.group_id_matrix == [["group_0", "group_0"], ["group_0", "group_1"]]


def test_build_group_id_matrix_rejects_conflicting_assignment() -> None:
    summary = {
        "template_name": "x",
        "compositions": [
            {
                "composition_number": 1,
                "group_details": [
                    {"group_id": "group_0", "tasks": ["Taskset1"]},
                    {"group_id": "group_1", "tasks": ["Taskset1"]},
                ],
            }
        ],
    }

    with pytest.raises(ValueError, match="assigned to multiple groups"):
        build_group_id_matrix(summary)

