#!/usr/bin/env python3
"""
Stage 3.1 closure package runner.

This is an orchestration wrapper, not a new experimental engine.

It runs:
1. optional smoke tests;
2. Stage 3.1A compatibility rerun;
3. Stage 3.1A analysis;
4. Stage 3.1B ablation suite;
5. curated artifact packaging into docs/results/.

Raw outputs remain under logs/.
Curated paper-grade outputs are copied into docs/results/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run Stage 3.1A compatibility + Stage 3.1B closure package"
    )

    ap.add_argument(
        "--mode",
        choices=["smoke", "full", "analyze-only"],
        default="smoke",
        help="Run profile. smoke is small; full is paper-grade; analyze-only reuses existing runs.",
    )

    ap.add_argument(
        "--raw-output-root",
        default="logs/stage3/stage3_1_closure_raw",
        help="Raw output root. This should remain ignored by Git.",
    )
    ap.add_argument(
        "--curated-output-root",
        default="docs/results",
        help="Tracked curated output root.",
    )

    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-stage3-1a", action="store_true")
    ap.add_argument("--skip-stage3-1b", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")

    ap.add_argument(
        "--ablations",
        default=None,
        help="Stage 3.1B ablations. Default: full for smoke, all for full.",
    )

    ap.add_argument("--stage3-1a-seeds", type=int, default=None)
    ap.add_argument("--stage3-1a-trials", type=int, default=None)
    ap.add_argument("--stage3-1b-balanced-seeds", type=int, default=None)
    ap.add_argument("--stage3-1b-balanced-trials", type=int, default=None)
    ap.add_argument("--stage3-1b-one-shot-seeds", type=int, default=None)
    ap.add_argument("--stage3-1b-one-shot-trials", type=int, default=None)

    ap.add_argument(
        "--stage3-1a-run-dir",
        default=None,
        help="Existing Stage 3.1A run dir for analyze-only mode.",
    )
    ap.add_argument(
        "--stage3-1b-suite-dir",
        default=None,
        help="Existing Stage 3.1B suite dir for analyze-only mode.",
    )

    ap.add_argument(
        "--clean-curated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clean curated stage3 output directories before packaging.",
    )

    return ap.parse_args()


# ---------------------------------------------------------------------
# Git / subprocess helpers
# ---------------------------------------------------------------------

def run_capture(cmd: List[str], cwd: Path = PROJECT_ROOT) -> str:
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )
    return p.stdout.strip()


def get_git_info() -> Dict[str, Any]:
    branch = run_capture(["git", "branch", "--show-current"])
    commit = run_capture(["git", "rev-parse", "HEAD"])
    status = run_capture(["git", "status", "--porcelain"])
    return {
        "branch": branch,
        "commit": commit,
        "working_tree_clean": status == "",
        "status_porcelain": status,
    }


def run_cmd(label: str, cmd: List[str], manifest: Dict[str, Any]) -> None:
    print()
    print("=" * 80)
    print(label)
    print("=" * 80)
    print(" ".join(cmd))
    print()

    started = datetime.now().isoformat()
    p = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    ended = datetime.now().isoformat()

    manifest.setdefault("commands", []).append(
        {
            "label": label,
            "cmd": cmd,
            "returncode": p.returncode,
            "started": started,
            "ended": ended,
        }
    )

    if p.returncode != 0:
        raise RuntimeError(f"{label} failed with return code {p.returncode}")


# ---------------------------------------------------------------------
# Paths / packaging helpers
# ---------------------------------------------------------------------

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    ensure_dir(path)


def category_for_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
        return "figures"
    if suffix in {".csv", ".tsv"}:
        return "tables"
    if suffix in {".json"}:
        return "stats"
    if suffix in {".md", ".txt"}:
        return "reports"
    return "misc"


def copy_one_file(src: Path, curated_stage_root: Path, rename: Optional[str] = None) -> Path:
    category = category_for_file(src)
    dest_dir = curated_stage_root / category
    ensure_dir(dest_dir)
    dest = dest_dir / (rename or src.name)
    shutil.copy2(src, dest)
    return dest


def copy_artifacts_flat(src_dir: Path, curated_stage_root: Path) -> List[str]:
    copied: List[str] = []
    if not src_dir.exists():
        raise FileNotFoundError(f"Artifact source does not exist: {src_dir}")

    for src in sorted(src_dir.iterdir()):
        if not src.is_file():
            continue
        dest = copy_one_file(src, curated_stage_root)
        copied.append(str(dest.relative_to(PROJECT_ROOT)))

    return copied


def write_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def latest_suite_dir(base_dir: Path, before: set[Path]) -> Path:
    after = {p.resolve() for p in base_dir.glob("stage3_1b_ablation_suite_*") if p.is_dir()}
    new_dirs = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
    if new_dirs:
        return new_dirs[0]

    all_dirs = sorted(after, key=lambda p: p.stat().st_mtime, reverse=True)
    if not all_dirs:
        raise FileNotFoundError(f"No stage3_1b_ablation_suite_* dirs found in {base_dir}")
    return all_dirs[0]


# ---------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------

def resolve_profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.mode == "smoke":
        defaults = {
            "stage3_1a_seeds": 3,
            "stage3_1a_trials": 20,
            "stage3_1b_balanced_seeds": 3,
            "stage3_1b_balanced_trials": 20,
            "stage3_1b_one_shot_seeds": 3,
            "stage3_1b_one_shot_trials": 40,
            "ablations": "full",
        }
    else:
        defaults = {
            "stage3_1a_seeds": 50,
            "stage3_1a_trials": 100,
            "stage3_1b_balanced_seeds": 50,
            "stage3_1b_balanced_trials": 100,
            "stage3_1b_one_shot_seeds": 50,
            "stage3_1b_one_shot_trials": 100,
            "ablations": "all",
        }

    return {
        "stage3_1a_seeds": args.stage3_1a_seeds or defaults["stage3_1a_seeds"],
        "stage3_1a_trials": args.stage3_1a_trials or defaults["stage3_1a_trials"],
        "stage3_1b_balanced_seeds": args.stage3_1b_balanced_seeds or defaults["stage3_1b_balanced_seeds"],
        "stage3_1b_balanced_trials": args.stage3_1b_balanced_trials or defaults["stage3_1b_balanced_trials"],
        "stage3_1b_one_shot_seeds": args.stage3_1b_one_shot_seeds or defaults["stage3_1b_one_shot_seeds"],
        "stage3_1b_one_shot_trials": args.stage3_1b_one_shot_trials or defaults["stage3_1b_one_shot_trials"],
        "ablations": args.ablations or defaults["ablations"],
    }


def run_tests(manifest: Dict[str, Any]) -> None:
    run_cmd(
        "Smoke tests: stage3/tests",
        [sys.executable, "-m", "pytest", "stage3/tests"],
        manifest,
    )


def run_stage3_1a(
    raw_root: Path,
    curated_root: Path,
    profile: Dict[str, Any],
    manifest: Dict[str, Any],
    existing_run_dir: Optional[str] = None,
) -> None:
    curated_stage = curated_root / "stage3_1a"
    raw_analysis_dir = raw_root / f"stage3_1a_analysis_{timestamp()}"

    if existing_run_dir:
        run_dir = Path(existing_run_dir)
    else:
        run_dir = raw_root / f"stage3_1a_compat_{timestamp()}"
        run_cmd(
            "Stage 3.1A compatibility rerun",
            [
                sys.executable,
                "-m",
                "stage3.analysis.run_stage3_1a",
                "--n-seeds",
                str(profile["stage3_1a_seeds"]),
                "--n-trials",
                str(profile["stage3_1a_trials"]),
                "--output-dir",
                str(run_dir),
                "--ablation",
                "full",
            ],
            manifest,
        )

    run_cmd(
        "Stage 3.1A baseline analysis",
        [
            sys.executable,
            "-m",
            "stage3.analysis.analyze_stage3_1a_baseline",
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(raw_analysis_dir),
        ],
        manifest,
    )

    copied = copy_artifacts_flat(raw_analysis_dir, curated_stage)

    # Add selected run-level stats.
    for name in ["metadata.json", "run_summary.json", "seed_summary.csv"]:
        src = run_dir / name
        if src.exists():
            dest = copy_one_file(src, curated_stage, rename=f"stage3_1a_{name}")
            copied.append(str(dest.relative_to(PROJECT_ROOT)))

    readme = f"""# Stage 3.1A Compatibility Rerun

This folder contains curated outputs from a Stage 3.1A compatibility rerun
under the Stage 3.1B closure branch.

This is not a new Stage 3.1A claim. It checks that the revised Stage 3.1B
agent/kernel still preserves the calibrated Stage 3.1A baseline behavior.

## Source

- Raw run dir: `{run_dir}`
- Raw analysis dir: `{raw_analysis_dir}`
- Seeds: {profile["stage3_1a_seeds"]}
- Trials per seed: {profile["stage3_1a_trials"]}
"""
    write_text(curated_stage / "README.md", readme)

    manifest["stage3_1a"] = {
        "run_dir": str(run_dir),
        "analysis_dir": str(raw_analysis_dir),
        "curated_dir": str(curated_stage),
        "copied_artifacts": copied,
    }


def run_stage3_1b(
    raw_root: Path,
    curated_root: Path,
    profile: Dict[str, Any],
    manifest: Dict[str, Any],
    existing_suite_dir: Optional[str] = None,
) -> None:
    curated_stage = curated_root / "stage3_1b_closure"
    suite_base = raw_root / "stage3_1b"
    ensure_dir(suite_base)

    if existing_suite_dir:
        suite_dir = Path(existing_suite_dir)
    else:
        before = {p.resolve() for p in suite_base.glob("stage3_1b_ablation_suite_*") if p.is_dir()}

        run_cmd(
            "Stage 3.1B ablation suite",
            [
                sys.executable,
                "-m",
                "stage3.analysis.run_stage3_1b_ablation_suite",
                "--ablations",
                str(profile["ablations"]),
                "--n-seeds-balanced",
                str(profile["stage3_1b_balanced_seeds"]),
                "--n-trials-balanced",
                str(profile["stage3_1b_balanced_trials"]),
                "--n-seeds-one-shot",
                str(profile["stage3_1b_one_shot_seeds"]),
                "--n-trials-one-shot",
                str(profile["stage3_1b_one_shot_trials"]),
                "--base-output-dir",
                str(suite_base),
            ],
            manifest,
        )

        suite_dir = latest_suite_dir(suite_base, before)

    analysis_dir = suite_dir / "analysis"

    # If analyze-only points to a suite without analysis, run the analyzer.
    suite_manifest = suite_dir / "manifest.json"
    if not analysis_dir.exists():
        run_cmd(
            "Stage 3.1B ablation suite analysis",
            [
                sys.executable,
                "-m",
                "stage3.analysis.analyze_stage3_1b_ablation_suite",
                "--manifest",
                str(suite_manifest),
                "--output-dir",
                str(analysis_dir),
            ],
            manifest,
        )

    copied = copy_artifacts_flat(analysis_dir, curated_stage)

    if suite_manifest.exists():
        dest = copy_one_file(
            suite_manifest,
            curated_stage,
            rename="stage3_1b_suite_manifest.json",
        )
        copied.append(str(dest.relative_to(PROJECT_ROOT)))

    readme = f"""# Stage 3.1B Closure Results

This folder contains curated paper-grade outputs for Stage 3.1B closure.

Stage 3.1B closes the valence/exposure kernel. It tests tradeoff-sensitive
path choice, persistent one-shot deformation, and ablation-localized carrier
effects.

## Source

- Raw suite dir: `{suite_dir}`
- Raw analysis dir: `{analysis_dir}`
- Ablations: {profile["ablations"]}
- Balanced seeds: {profile["stage3_1b_balanced_seeds"]}
- Balanced trials per seed: {profile["stage3_1b_balanced_trials"]}
- One-shot seeds: {profile["stage3_1b_one_shot_seeds"]}
- One-shot trials per seed: {profile["stage3_1b_one_shot_trials"]}
"""
    write_text(curated_stage / "README.md", readme)

    report = f"""# Stage 3.1B Closure Report

## Scope

This report was generated by `stage3.analysis.run_stage3_1_closure_package`.

The closure package uses existing Stage 3.1B runners and analyzers. It does
not introduce a new environment, agent, or experimental protocol.

## Raw source

- Suite: `{suite_dir}`
- Analysis: `{analysis_dir}`

## Curated package

- Figures: `docs/results/stage3_1b_closure/figures`
- Tables: `docs/results/stage3_1b_closure/tables`
- Stats: `docs/results/stage3_1b_closure/stats`
- Reports: `docs/results/stage3_1b_closure/reports`

## Interpretation boundary

Stage 3.1B closure supports the valence/exposure kernel only.

It does not claim absence inference, self-model-based visibility reasoning,
allocentric spatial cognition, or rodent-level VTE equivalence.
"""
    write_text(curated_stage / "reports" / "STAGE3_1B_CLOSURE_REPORT.md", report)

    manifest["stage3_1b"] = {
        "suite_dir": str(suite_dir),
        "analysis_dir": str(analysis_dir),
        "curated_dir": str(curated_stage),
        "copied_artifacts": copied,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    profile = resolve_profile(args)

    raw_root = PROJECT_ROOT / args.raw_output_root
    curated_root = PROJECT_ROOT / args.curated_output_root

    git_info = get_git_info()
    if not git_info["working_tree_clean"] and not args.allow_dirty:
        raise RuntimeError(
            "Working tree is not clean. Commit or stash changes first, or pass --allow-dirty.\n"
            f"{git_info['status_porcelain']}"
        )

    run_id = timestamp()
    manifest: Dict[str, Any] = {
        "stage": "stage3_1_closure",
        "mode": args.mode,
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "raw_output_root": str(raw_root),
        "curated_output_root": str(curated_root),
        "git": git_info,
        "profile": profile,
        "commands": [],
    }

    ensure_dir(raw_root)
    ensure_dir(curated_root)

    if args.clean_curated:
        if not args.skip_stage3_1a:
            reset_dir(curated_root / "stage3_1a")
        if not args.skip_stage3_1b:
            reset_dir(curated_root / "stage3_1b_closure")

    if args.mode != "analyze-only" and not args.skip_tests:
        run_tests(manifest)

    if not args.skip_stage3_1a:
        if args.mode == "analyze-only" and not args.stage3_1a_run_dir:
            raise ValueError("--stage3-1a-run-dir is required for analyze-only unless --skip-stage3-1a")
        run_stage3_1a(
            raw_root=raw_root,
            curated_root=curated_root,
            profile=profile,
            manifest=manifest,
            existing_run_dir=args.stage3_1a_run_dir,
        )

    if not args.skip_stage3_1b:
        if args.mode == "analyze-only" and not args.stage3_1b_suite_dir:
            raise ValueError("--stage3-1b-suite-dir is required for analyze-only unless --skip-stage3-1b")
        run_stage3_1b(
            raw_root=raw_root,
            curated_root=curated_root,
            profile=profile,
            manifest=manifest,
            existing_suite_dir=args.stage3_1b_suite_dir,
        )

    manifest["completed_at"] = datetime.now().isoformat()

    manifest_path = curated_root / "stage3_1_closure_manifest.json"
    write_json(manifest_path, manifest)

    print()
    print("=" * 80)
    print("Stage 3.1 closure package complete")
    print("=" * 80)
    print(f"Manifest: {manifest_path}")
    print(f"Curated root: {curated_root}")
    print(f"Raw root: {raw_root}")
    print()


if __name__ == "__main__":
    main()