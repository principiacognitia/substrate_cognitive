#!/usr/bin/env python3
"""
Stage 3.1B ablation suite runner.

Runs a compact, manifest-driven ablation package for Stage 3.1B:
- balanced_conflict baseline
- one-shot shock on open
- one-shot treat on covered

The runner follows the same in-memory orchestration pattern as
run_stage3_1b_param_sweep.py: it imports the Stage 3.1B runner module and
calls its functions directly, instead of assuming every setting exists as a
CLI override.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Stage 3.1B ablation suite")
    ap.add_argument(
        "--ablations",
        default="all",
        help="Comma-separated ablation names, or 'all' (default)",
    )
    ap.add_argument("--condition", default="balanced_conflict")
    ap.add_argument("--n-seeds-balanced", type=int, default=10)
    ap.add_argument("--n-trials-balanced", type=int, default=40)
    ap.add_argument("--n-seeds-one-shot", type=int, default=10)
    ap.add_argument("--n-trials-one-shot", type=int, default=100)
    ap.add_argument("--shock-trial", type=int, default=30)
    ap.add_argument("--shock-target-path", choices=["open", "covered"], default="open")
    ap.add_argument("--treat-target-path", choices=["open", "covered"], default="covered")
    ap.add_argument("--base-output-dir", default="logs/stage3/stage3_1_ablation_suite")
    ap.add_argument("--run-module", default="stage3.analysis.run_stage3_1b")
    ap.add_argument("--analyze-module", default="stage3.analysis.analyze_stage3_1b_ablation_suite")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--no-balanced", action="store_true")
    ap.add_argument("--no-shock", action="store_true")
    ap.add_argument("--no-treat", action="store_true")
    ap.add_argument("--no-analyze", action="store_true")
    ap.add_argument("--force-one-shot-event", action="store_true", default=True)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def parse_ablations(raw: str, available: Sequence[str]) -> List[str]:
    if raw.strip().lower() == "all":
        return list(available)
    items = [x.strip() for x in raw.split(",") if x.strip()]
    if not items:
        raise ValueError("No ablations specified")
    unknown = [x for x in items if x not in available]
    if unknown:
        raise ValueError(f"Unknown ablations: {', '.join(unknown)}")
    return items


def metric(summary: Optional[Dict[str, Any]], key: str) -> str:
    if not summary:
        return "."
    val = summary.get(key)
    if val is None:
        return "."
    try:
        return f"{float(val):.3f}"
    except Exception:
        return str(val)


def extract_condition_summary(run_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not run_summary:
        return {}
    return dict(run_summary.get("condition_summary", {}) or {})


def resolve_condition_id(runner_mod: Any, condition_name: str) -> str:
    normalized = runner_mod.normalize_condition_name(condition_name)
    canonical = runner_mod.get_canonical_conditions()
    info = canonical.get(normalized, {})
    return str(info.get("condition_id") or "R1_T2")


def main() -> None:
    args = parse_args()

    runner_mod = import_module(args.run_module)
    available_ablations = list(runner_mod.CONFIG_3_1B["ablation"].keys())
    ablations = parse_ablations(args.ablations, available_ablations)
    condition_id = resolve_condition_id(runner_mod, args.condition)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_root = Path(args.base_output_dir) / f"stage3_1b_ablation_suite_{ts}"
    suite_root.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "condition": args.condition,
        "condition_id": condition_id,
        "shock_trial": args.shock_trial,
        "shock_target_path": args.shock_target_path,
        "treat_target_path": args.treat_target_path,
        "ablations": ablations,
        "balanced_runs": [],
        "shock_runs": [],
        "treat_runs": [],
    }

    print(f"Suite root : {suite_root}")
    print(f"Condition  : {args.condition} ({condition_id})")
    print(f"Ablations  : {', '.join(ablations)}")
    print()

    original_agent_cfg = copy.deepcopy(runner_mod.AGENT_CONFIG_3_1B)

    for ablation in ablations:
        runner_mod.AGENT_CONFIG_3_1B = copy.deepcopy(original_agent_cfg)
        print(f"=== {ablation} ===")

        if not args.no_balanced:
            balanced_dir = suite_root / "balanced" / ablation
            if args.dry_run:
                print(f"balanced  ablation={ablation:<12} dry-run")
            else:
                summary = runner_mod.run_single_condition(
                    condition_name=args.condition,
                    n_seeds=args.n_seeds_balanced,
                    n_trials=args.n_trials_balanced,
                    ablation=ablation,
                    output_dir=str(balanced_dir),
                    verbose=False,
                    debug=False,
                    debug_console=False,
                    debug_junction_only=False,
                    debug_trial_window=2,
                )
                cond = extract_condition_summary(summary)
                manifest["balanced_runs"].append({
                    "ablation": ablation,
                    "run_dir": str(balanced_dir),
                })
                print(
                    f"balanced  p_open={metric(cond, 'p_open')}  "
                    f"timeout={metric(cond, 'p_commit_timeout')}  "
                    f"lat={metric(cond, 'mean_commit_latency')}"
                )

        if not args.no_shock:
            shock_dir = suite_root / "shock" / ablation
            if args.dry_run:
                print(f"shock     ablation={ablation:<12} dry-run")
            else:
                protocol = runner_mod.get_one_shot_protocol(
                    kind="shock",
                    shock_trial=args.shock_trial,
                    condition_name=args.condition,
                    condition_id=condition_id,
                    path=args.shock_target_path,
                    salience=0.9,
                    stakes=10.0,
                )
                summary = runner_mod.run_one_shot_protocol(
                    n_seeds=args.n_seeds_one_shot,
                    n_trials=args.n_trials_one_shot,
                    ablation=ablation,
                    output_dir=str(shock_dir),
                    verbose=False,
                    diagnostic_forced_shock=bool(args.force_one_shot_event),
                    diagnostic_forced_treat=False,
                    forced_shock_path=args.shock_target_path if args.force_one_shot_event else None,
                    debug=True,
                    debug_console=False,
                    debug_junction_only=True,
                    debug_trial_window=2,
                    one_shot_protocol=protocol,
                )
                cond = extract_condition_summary(summary)
                manifest["shock_runs"].append({
                    "ablation": ablation,
                    "run_dir": str(shock_dir),
                })
                print(
                    f"shock     p_open={metric(cond, 'p_open')}  "
                    f"timeout={metric(cond, 'p_commit_timeout')}  "
                    f"lat={metric(cond, 'mean_commit_latency')}"
                )

        if not args.no_treat:
            treat_dir = suite_root / "treat" / ablation
            if args.dry_run:
                print(f"treat     ablation={ablation:<12} dry-run")
            else:
                protocol = runner_mod.get_one_shot_protocol(
                    kind="treat",
                    shock_trial=args.shock_trial,
                    condition_name=args.condition,
                    condition_id=condition_id,
                    path=args.treat_target_path,
                    salience=0.9,
                    stakes=10.0,
                )
                summary = runner_mod.run_one_shot_protocol(
                    n_seeds=args.n_seeds_one_shot,
                    n_trials=args.n_trials_one_shot,
                    ablation=ablation,
                    output_dir=str(treat_dir),
                    verbose=False,
                    diagnostic_forced_shock=False,
                    diagnostic_forced_treat=bool(args.force_one_shot_event),
                    forced_shock_path=args.treat_target_path if args.force_one_shot_event else None,
                    debug=True,
                    debug_console=False,
                    debug_junction_only=True,
                    debug_trial_window=2,
                    one_shot_protocol=protocol,
                )
                cond = extract_condition_summary(summary)
                manifest["treat_runs"].append({
                    "ablation": ablation,
                    "run_dir": str(treat_dir),
                })
                print(
                    f"treat     p_open={metric(cond, 'p_open')}  "
                    f"timeout={metric(cond, 'p_commit_timeout')}  "
                    f"lat={metric(cond, 'mean_commit_latency')}"
                )

        print()

    runner_mod.AGENT_CONFIG_3_1B = copy.deepcopy(original_agent_cfg)

    manifest_path = suite_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Manifest saved: {manifest_path}")

    if args.no_analyze or args.dry_run:
        return

    analysis_dir = suite_root / "analysis"
    analyze_mod = import_module(args.analyze_module)
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            args.analyze_module,
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(analysis_dir),
        ]
        analyze_mod.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
