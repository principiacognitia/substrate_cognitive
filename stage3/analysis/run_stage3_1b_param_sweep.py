#!/usr/bin/env python3
"""
Stage 3.1B parameter sweep runner.

This version is aligned with the current Stage 3.1B codebase.
It no longer assumes that every tunable has a dedicated CLI override in
run_stage3_1b.py. Instead, it can patch the in-memory AGENT_CONFIG_3_1B
before calling the runner functions directly.

Supported built-in parameters:
- temporal_state.local_opp_immediate_seed_weight
- temporal_state.w_qpos_input
- temporal_state.k_pos
- temporal_state.rho_pos
- temporal_state.w_pos_to_opp
- temporal_state.w_qneg_input
- action_policy.local_opp_bonus_weight

Typical usage:
python -m stage3.analysis.run_stage3_1b_param_sweep \
  --param local_opp_immediate_seed_weight \
  --values 0.25,0.5,0.75,1.0,1.25,1.5,2.0 \
  --target-path covered \
  --n-seeds-balanced 5 \
  --n-trials-balanced 40 \
  --n-seeds-one-shot 5
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_PARAM_PATHS: Dict[str, Tuple[str, ...]] = {
    "local_opp_immediate_seed_weight": ("temporal_state", "local_opp_immediate_seed_weight"),
    "w_qpos_input": ("temporal_state", "w_qpos_input"),
    "k_pos": ("temporal_state", "k_pos"),
    "rho_pos": ("temporal_state", "rho_pos"),
    "w_pos_to_opp": ("temporal_state", "w_pos_to_opp"),
    "w_qneg_input": ("temporal_state", "w_qneg_input"),
    "local_opp_bonus_weight": ("action_policy", "local_opp_bonus_weight"),
}

DEFAULT_SHORT_MAP: Dict[str, str] = {
    "local_opp_immediate_seed_weight": "lseed",
    "w_qpos_input": "wqpi",
    "k_pos": "kpos",
    "rho_pos": "rhop",
    "w_pos_to_opp": "wpto",
    "w_qneg_input": "wqni",
    "local_opp_bonus_weight": "lobw",
}

CLI_OVERRIDE_PARAMS = {"w_qneg_input", "w_qpos_input", "k_pos"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Stage 3.1B sweep with compact console output")
    ap.add_argument("--param", required=True)
    ap.add_argument("--values", required=True, help="Comma-separated numeric values")
    ap.add_argument(
        "--config-path",
        default=None,
        help="Explicit dotted config path, for example temporal_state.local_opp_immediate_seed_weight",
    )
    ap.add_argument("--short-name", default=None)
    ap.add_argument("--target-path", choices=["open", "covered"], default="covered")
    ap.add_argument("--ablation", default="full")
    ap.add_argument("--n-seeds-balanced", type=int, default=5)
    ap.add_argument("--n-trials-balanced", type=int, default=40)
    ap.add_argument("--n-seeds-one-shot", type=int, default=5)
    ap.add_argument("--n-trials-one-shot", type=int, default=100)
    ap.add_argument("--shock-trial", type=int, default=30)
    ap.add_argument("--condition", default="balanced_conflict")
    ap.add_argument("--base-output-dir", default="logs/stage3/stage3_1b_sweeps")
    ap.add_argument("--run-module", default="stage3.analysis.run_stage3_1b")
    ap.add_argument("--no-balanced", action="store_true")
    ap.add_argument("--no-one-shot", action="store_true")
    ap.add_argument("--no-analyze", action="store_true")
    ap.add_argument("--analyze-module", default="stage3.analysis.analyze_stage3_1b_param_sweep")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument(
        "--force-one-shot-event",
        action="store_true",
        default=True,
        help="Force the one-shot treat onto the configured target path (default: on)",
    )
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def parse_values(raw: str) -> List[float]:
    values: List[float] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        raise ValueError("No numeric values provided")
    return values


def value_slug(value: float) -> str:
    s = f"{value:.6g}"
    return s.replace("-", "m").replace(".", "p")


def resolve_short_name(param: str, short_name: Optional[str]) -> str:
    if short_name:
        return short_name
    return DEFAULT_SHORT_MAP.get(param, param)


def resolve_config_path(param: str, explicit: Optional[str]) -> Tuple[str, ...]:
    if explicit:
        return tuple(x.strip() for x in explicit.split(".") if x.strip())
    if param in DEFAULT_PARAM_PATHS:
        return DEFAULT_PARAM_PATHS[param]
    return ("temporal_state", param)


def set_nested(mapping: Dict[str, Any], path: Sequence[str], value: Any) -> None:
    cur = mapping
    for key in path[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[path[-1]] = value


def get_nested(mapping: Dict[str, Any], path: Sequence[str], default: Any = None) -> Any:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


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


def main() -> None:
    args = parse_args()
    values = parse_values(args.values)
    short_name = resolve_short_name(args.param, args.short_name)
    config_path = resolve_config_path(args.param, args.config_path)

    runner_mod = import_module(args.run_module)
    original_agent_cfg = copy.deepcopy(runner_mod.AGENT_CONFIG_3_1B)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = Path(args.base_output_dir) / f"{short_name}_sweep_{ts}"
    sweep_root.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "param": args.param,
        "config_path": ".".join(config_path),
        "short_name": short_name,
        "target_path": args.target_path,
        "shock_trial": args.shock_trial,
        "ablation": args.ablation,
        "balanced_runs": [],
        "one_shot_runs": [],
    }

    print(f"Sweep root: {sweep_root}")
    print(f"Parameter : {args.param}")
    print(f"Cfg path  : {'.'.join(config_path)}")
    print(f"Values    : {', '.join(f'{v:.6g}' for v in values)}")
    print()

    for value in values:
        slug = value_slug(value)
        runner_mod.AGENT_CONFIG_3_1B = copy.deepcopy(original_agent_cfg)
        set_nested(runner_mod.AGENT_CONFIG_3_1B, config_path, float(value))

        effective_value = get_nested(runner_mod.AGENT_CONFIG_3_1B, config_path)
        if effective_value is None:
            raise SystemExit(f"Could not set {'.'.join(config_path)} for value={value}")

        cli_override_kwargs: Dict[str, Any] = {}
        if args.param == "w_qneg_input":
            cli_override_kwargs["w_qneg_input_override"] = float(value)
        elif args.param == "w_qpos_input":
            cli_override_kwargs["w_qpos_input_override"] = float(value)
        elif args.param == "k_pos":
            cli_override_kwargs["k_pos_override"] = float(value)

        if not args.no_balanced:
            balanced_dir = sweep_root / "balanced" / f"{short_name}_{slug}"
            if args.dry_run:
                print(f"balanced  value={value:>8.6g}  dry-run")
            else:
                summary = runner_mod.run_single_condition(
                    condition_name=args.condition,
                    n_seeds=args.n_seeds_balanced,
                    n_trials=args.n_trials_balanced,
                    ablation=args.ablation,
                    output_dir=str(balanced_dir),
                    verbose=False,
                    debug=False,
                    debug_console=False,
                    debug_junction_only=False,
                    debug_trial_window=2,
                    **cli_override_kwargs,
                )
                cond = extract_condition_summary(summary)
                manifest["balanced_runs"].append({
                    "value": value,
                    "slug": slug,
                    "run_dir": str(balanced_dir),
                })
                print(
                    f"balanced  value={value:>8.6g}  "
                    f"p_open={metric(cond, 'p_open')}  "
                    f"timeout={metric(cond, 'p_commit_timeout')}  "
                    f"lat={metric(cond, 'mean_commit_latency')}"
                )

        if not args.no_one_shot:
            one_shot_dir = sweep_root / "one_shot" / f"{short_name}_{slug}"
            if args.dry_run:
                print(f"one-shot  value={value:>8.6g}  dry-run")
            else:
                protocol = runner_mod.get_one_shot_protocol(
                    kind="treat",
                    shock_trial=args.shock_trial,
                    condition_name=args.condition,
                    condition_id="R1_T2",
                    path=args.target_path,
                    salience=0.9,
                    stakes=10.0,
                )
                summary = runner_mod.run_one_shot_protocol(
                    n_seeds=args.n_seeds_one_shot,
                    n_trials=args.n_trials_one_shot,
                    ablation=args.ablation,
                    output_dir=str(one_shot_dir),
                    verbose=False,
                    diagnostic_forced_shock=False,
                    diagnostic_forced_treat=bool(args.force_one_shot_event),
                    forced_shock_path=args.target_path if args.force_one_shot_event else None,
                    debug=True,
                    debug_console=False,
                    debug_junction_only=True,
                    debug_trial_window=2,
                    one_shot_protocol=protocol,
                    **cli_override_kwargs,
                )
                cond = extract_condition_summary(summary)
                manifest["one_shot_runs"].append({
                    "value": value,
                    "slug": slug,
                    "run_dir": str(one_shot_dir),
                })
                print(
                    f"one-shot  value={value:>8.6g}  "
                    f"p_open={metric(cond, 'p_open')}  "
                    f"timeout={metric(cond, 'p_commit_timeout')}  "
                    f"lat={metric(cond, 'mean_commit_latency')}"
                )

    runner_mod.AGENT_CONFIG_3_1B = copy.deepcopy(original_agent_cfg)

    manifest_path = sweep_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nManifest saved: {manifest_path}")

    if args.no_analyze or args.dry_run:
        return

    analysis_dir = sweep_root / "analysis"
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
