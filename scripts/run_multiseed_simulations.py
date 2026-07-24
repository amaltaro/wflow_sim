#!/usr/bin/env python3
"""
Run all workflow constructions for one scenario across N RNG seeds.

For each seed in ``0 .. runs-1``, every ``*_const_*.json`` in the selected
template directory is simulated with that same seed. Results are written under:

  {output_root}/seed{seed}/.../<time>/fr<fr>/<rate>/<construction>.json

by invoking ``python -m src.workflow_runner`` with ``--output-base`` and
``--seed``. This script is self-contained (no Makefile / other-script changes).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent


def list_construction_files(templates_dir: Path) -> List[Path]:
    """Return sorted workflow construction JSON paths under ``templates_dir``."""
    files = sorted(templates_dir.glob("*_const_*.json"))
    return [p for p in files if p.is_file()]


def to_repo_relative(path: Path) -> Path:
    """Return ``path`` relative to the repository root.

    ``workflow_runner`` nests outputs from the input path; absolute paths outside
    ``templates/`` can discard ``--output-base``. Prefer repo-relative inputs.
    """
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            "Workflow path must be inside the repository so results nest under "
            f"--output-base correctly: {resolved}"
        ) from exc


def seed_output_base(output_root: Path, seed: int) -> Path:
    """Return per-seed output base directory (e.g. ``.../rebuttal/seed0``)."""
    return output_root / f"seed{seed}"


def write_campaign_manifest(
    output_root: Path,
    *,
    use_case: str,
    templates_dir: Path,
    construction_files: Sequence[Path],
    target_wallclock_time: int,
    job_failure_rate: int,
    data_transfer_rate_mb_per_s: float,
    max_job_slots: int,
    runs: int,
) -> Path:
    """Write a small campaign.json describing the multi-seed run parameters."""
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "use_case": use_case,
        "templates_dir": str(templates_dir),
        "constructions": [p.name for p in construction_files],
        "target_wallclock_time": target_wallclock_time,
        "job_failure_rate": job_failure_rate,
        "data_transfer_rate_mb_per_s": data_transfer_rate_mb_per_s,
        "max_job_slots": max_job_slots,
        "runs": runs,
        "seeds": list(range(runs)),
        "output_root": str(output_root),
        "output_layout": (
            "{output_root}/seed{seed}/others/<use_case>/<time>/fr<fr>/<rate>/"
            "<construction>.json"
        ),
    }
    path = output_root / "campaign.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def build_runner_command(
    *,
    workflow_path: Path,
    output_base: Path,
    seed: int,
    target_wallclock_time: int,
    job_failure_rate: int,
    data_transfer_rate_mb_per_s: float,
    max_job_slots: int,
    python_executable: str,
) -> List[str]:
    """Build the ``workflow_runner`` CLI invocation for one construction/seed."""
    return [
        python_executable,
        "-m",
        "src.workflow_runner",
        "--input-workflow-path",
        str(workflow_path),
        "--target-wallclock-time",
        str(target_wallclock_time),
        "--max-job-slots",
        str(max_job_slots),
        "--failure-rate",
        str(job_failure_rate),
        "--data-transfer-rate",
        str(data_transfer_rate_mb_per_s),
        "--seed",
        str(seed),
        "--output-base",
        str(output_base),
    ]


def print_campaign_manifest(manifest_path: Path) -> None:
    """Print the persisted campaign.json contents."""
    print("Campaign manifest:")
    print(manifest_path.read_text(encoding="utf-8"), end="")


def run_multiseed_campaign(
    *,
    use_case: str,
    templates_dir: Path,
    output_root: Path,
    target_wallclock_time: int,
    job_failure_rate: int,
    data_transfer_rate_mb_per_s: float,
    max_job_slots: int,
    runs: int,
    python_executable: str | None = None,
) -> None:
    """Execute all constructions for each seed in ``0 .. runs-1``."""
    if runs < 1:
        raise ValueError("runs must be >= 1")

    construction_files = list_construction_files(templates_dir)
    if not construction_files:
        raise FileNotFoundError(
            f"No *_const_*.json files found in {templates_dir}"
        )

    py = python_executable or sys.executable
    manifest_path = write_campaign_manifest(
        output_root,
        use_case=use_case,
        templates_dir=templates_dir,
        construction_files=construction_files,
        target_wallclock_time=target_wallclock_time,
        job_failure_rate=job_failure_rate,
        data_transfer_rate_mb_per_s=data_transfer_rate_mb_per_s,
        max_job_slots=max_job_slots,
        runs=runs,
    )

    total = runs * len(construction_files)
    print(f"Multi-seed campaign: {use_case}")
    print(f"  constructions: {len(construction_files)}")
    print(f"  runs/seeds:    {runs} (seeds 0..{runs - 1})")
    print(f"  wallclock:     {target_wallclock_time}s")
    print(f"  failure rate:  {job_failure_rate}%")
    print(f"  data rate:     {data_transfer_rate_mb_per_s} MB/s")
    print(f"  output root:   {output_root}")
    print(f"  manifest:      {manifest_path}")
    print(f"  total sims:    {total}")
    print("")

    completed = 0
    for seed in range(runs):
        out_base = seed_output_base(output_root, seed)
        print(f"=== Seed {seed} ({seed + 1}/{runs}) ===")

        for workflow_path in construction_files:
            completed += 1
            rel_workflow = to_repo_relative(workflow_path)
            print(
                f"  [{completed}/{total}] {rel_workflow.name} "
                f"(seed={seed})"
            )
            cmd = build_runner_command(
                workflow_path=rel_workflow,
                output_base=out_base,
                seed=seed,
                target_wallclock_time=target_wallclock_time,
                job_failure_rate=job_failure_rate,
                data_transfer_rate_mb_per_s=data_transfer_rate_mb_per_s,
                max_job_slots=max_job_slots,
                python_executable=py,
            )
            result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Simulation failed for {workflow_path.name} "
                    f"seed={seed} (exit {result.returncode})"
                )

        print("")

    print("Multi-seed campaign completed.")
    print("")
    print_campaign_manifest(manifest_path)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the multi-seed campaign."""
    parser = argparse.ArgumentParser(
        description=(
            "Run all workflow constructions for one scenario across N seeds "
            "(seed = run index 0..N-1)."
        )
    )
    parser.add_argument(
        "--use-case",
        type=str,
        default="seq_real",
        help="Workflow use case under templates/others/ (default: seq_real)",
    )
    parser.add_argument(
        "--templates-dir",
        type=str,
        default=None,
        help=(
            "Directory with *_const_*.json files "
            "(default: templates/others/<use-case>)"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/sim/rebuttal",
        help="Root directory for seed*/ trees (default: results/sim/rebuttal)",
    )
    parser.add_argument(
        "--target-wallclock-time",
        type=int,
        default=43200,
        help="Target wallclock time in seconds (default: 43200 = 12h)",
    )
    parser.add_argument(
        "--failure-rate",
        dest="job_failure_rate",
        type=int,
        default=5,
        help="Job failure rate as percentage 0-99 (default: 5)",
    )
    parser.add_argument(
        "--data-transfer-rate",
        type=float,
        default=100.0,
        help="Network data transfer rate in MB/s (default: 100.0)",
    )
    parser.add_argument(
        "--max-job-slots",
        type=int,
        default=-1,
        help="Maximum job slots (-1 = infinite, default: -1)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of seeds/runs; seeds are 0..runs-1 (default: 10)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_arguments(argv)

    if args.job_failure_rate >= 100 or args.job_failure_rate < 0:
        print("ERROR: --failure-rate must be in [0, 99].", file=sys.stderr)
        return 1
    if args.runs < 1:
        print("ERROR: --runs must be >= 1.", file=sys.stderr)
        return 1

    use_case = args.use_case
    templates_dir = (
        Path(args.templates_dir)
        if args.templates_dir
        else REPO_ROOT / "templates" / "others" / use_case
    )
    if not templates_dir.is_dir():
        print(f"ERROR: templates dir not found: {templates_dir}", file=sys.stderr)
        return 1

    try:
        run_multiseed_campaign(
            use_case=use_case,
            templates_dir=templates_dir,
            output_root=Path(args.output_root),
            target_wallclock_time=args.target_wallclock_time,
            job_failure_rate=args.job_failure_rate,
            data_transfer_rate_mb_per_s=args.data_transfer_rate,
            max_job_slots=args.max_job_slots,
            runs=args.runs,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
