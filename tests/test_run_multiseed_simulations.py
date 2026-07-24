"""Unit tests for scripts/run_multiseed_simulations.py helpers."""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_multiseed_simulations import (  # noqa: E402
    REPO_ROOT,
    build_runner_command,
    list_construction_files,
    parse_arguments,
    print_campaign_manifest,
    seed_output_base,
    to_repo_relative,
    write_campaign_manifest,
)


class TestListConstructionFiles:
    """Tests for construction file discovery."""

    def test_lists_sorted_const_files(self, tmp_path: Path) -> None:
        """Only *_const_*.json files are returned, sorted by name."""
        (tmp_path / "seq_real_const_002.json").write_text("{}")
        (tmp_path / "seq_real_const_001.json").write_text("{}")
        (tmp_path / "notes.txt").write_text("ignore")
        (tmp_path / "compositions_summary.json").write_text("{}")

        files = list_construction_files(tmp_path)
        assert [p.name for p in files] == [
            "seq_real_const_001.json",
            "seq_real_const_002.json",
        ]

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty template dir yields an empty list."""
        assert list_construction_files(tmp_path) == []


class TestSeedOutputBase:
    """Tests for per-seed output directory naming."""

    def test_seed_subdir_name(self, tmp_path: Path) -> None:
        """Seed directories use the seed{N} naming convention."""
        assert seed_output_base(tmp_path, 0) == tmp_path / "seed0"
        assert seed_output_base(tmp_path, 9) == tmp_path / "seed9"


class TestToRepoRelative:
    """Tests for converting workflow paths to repo-relative form."""

    def test_relative_under_repo(self) -> None:
        """Paths under the repo become relative to REPO_ROOT."""
        abs_path = REPO_ROOT / "templates" / "others" / "seq_real" / "x.json"
        assert to_repo_relative(abs_path) == Path(
            "templates/others/seq_real/x.json"
        )

    def test_outside_repo_raises(self, tmp_path: Path) -> None:
        """Paths outside the repository are rejected."""
        outside = tmp_path / "outside_const_001.json"
        outside.write_text("{}")
        with pytest.raises(ValueError, match="inside the repository"):
            to_repo_relative(outside)


class TestBuildRunnerCommand:
    """Tests for workflow_runner CLI construction."""

    def test_includes_seed_and_output_base(self, tmp_path: Path) -> None:
        """Command passes --seed and --output-base to workflow_runner."""
        workflow = tmp_path / "seq_real_const_001.json"
        out_base = tmp_path / "seed3"
        cmd = build_runner_command(
            workflow_path=workflow,
            output_base=out_base,
            seed=3,
            target_wallclock_time=43200,
            job_failure_rate=5,
            data_transfer_rate_mb_per_s=100.0,
            max_job_slots=-1,
            python_executable="python",
        )
        assert cmd[:3] == ["python", "-m", "src.workflow_runner"]
        assert "--seed" in cmd
        assert cmd[cmd.index("--seed") + 1] == "3"
        assert "--output-base" in cmd
        assert cmd[cmd.index("--output-base") + 1] == str(out_base)
        assert "--failure-rate" in cmd
        assert cmd[cmd.index("--failure-rate") + 1] == "5"


class TestCampaignManifest:
    """Tests for campaign.json writing."""

    def test_writes_seeds_zero_through_runs_minus_one(
        self, tmp_path: Path
    ) -> None:
        """Manifest records seeds as 0..runs-1 and lists constructions."""
        constructions = [
            tmp_path / "a_const_001.json",
            tmp_path / "a_const_002.json",
        ]
        path = write_campaign_manifest(
            tmp_path / "rebuttal",
            use_case="seq_real",
            templates_dir=tmp_path,
            construction_files=constructions,
            target_wallclock_time=43200,
            job_failure_rate=5,
            data_transfer_rate_mb_per_s=100.0,
            max_job_slots=-1,
            runs=3,
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["runs"] == 3
        assert data["seeds"] == [0, 1, 2]
        assert data["constructions"] == [
            "a_const_001.json",
            "a_const_002.json",
        ]
        assert data["use_case"] == "seq_real"


class TestParseArguments:
    """Tests for CLI defaults and overrides."""

    def test_defaults_match_paper_scenario(self) -> None:
        """Defaults target seq_real / 12h / fr5 / 100MBps / 10 runs."""
        args = parse_arguments([])
        assert args.use_case == "seq_real"
        assert args.target_wallclock_time == 43200
        assert args.job_failure_rate == 5
        assert args.data_transfer_rate == 100.0
        assert args.runs == 10
        assert args.output_root == "results/sim/rebuttal"

    def test_runs_override(self) -> None:
        """--runs is accepted from the CLI."""
        args = parse_arguments(["--runs", "2"])
        assert args.runs == 2


class TestPrintCampaignManifest:
    """Tests for printing persisted campaign.json."""

    def test_prints_file_contents(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Printed output matches the campaign.json file body."""
        path = write_campaign_manifest(
            tmp_path / "rebuttal",
            use_case="seq_real",
            templates_dir=tmp_path,
            construction_files=[tmp_path / "a_const_001.json"],
            target_wallclock_time=43200,
            job_failure_rate=5,
            data_transfer_rate_mb_per_s=100.0,
            max_job_slots=-1,
            runs=2,
        )
        print_campaign_manifest(path)
        out = capsys.readouterr().out
        assert out.startswith("Campaign manifest:\n")
        assert path.read_text(encoding="utf-8") in out
        assert '"seeds": [' in out
        assert '"use_case": "seq_real"' in out


if __name__ == "__main__":
    pytest.main([__file__])
