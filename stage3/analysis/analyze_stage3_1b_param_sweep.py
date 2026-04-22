#!/usr/bin/env python3
"""
Analyze Stage 3.1B parameter sweeps and build compact calibration plots.

Preferred input is manifest.json produced by run_stage3_1b_param_sweep.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class RunSpec:
    value: float
    run_dir: Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze Stage 3.1B sweep outputs")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--run-balanced", action="append", default=[], help="Format: value=run_dir")
    ap.add_argument("--run-one-shot", action="append", default=[], help="Format: value=run_dir")
    ap.add_argument("--param", default=None)
    ap.add_argument("--short-name", default="sweep")
    ap.add_argument("--target-path", choices=["open", "covered"], default="covered")
    ap.add_argument("--shock-trial", type=int, default=30)
    ap.add_argument("--output-dir", required=True)
    return ap.parse_args()


def parse_run_arg(raw: str) -> RunSpec:
    if "=" not in raw:
        raise ValueError(f"Invalid run spec: {raw}. Expected value=run_dir")
    value_str, run_dir = raw.split("=", 1)
    return RunSpec(value=float(value_str), run_dir=Path(run_dir))


def load_manifest(path: Path) -> Tuple[str, str, str, int, List[RunSpec], List[RunSpec]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    param = str(data.get("param") or "sweep")
    short_name = str(data.get("short_name") or param)
    target_path = str(data.get("target_path") or "covered")
    shock_trial = int(data.get("shock_trial") or 30)
    balanced = [RunSpec(float(x["value"]), Path(x["run_dir"])) for x in data.get("balanced_runs", [])]
    one_shot = [RunSpec(float(x["value"]), Path(x["run_dir"])) for x in data.get("one_shot_runs", [])]
    return param, short_name, target_path, shock_trial, balanced, one_shot


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def find_single(run_dir: Path, pattern: str) -> Path:
    hits = sorted(run_dir.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No file matching {pattern} in {run_dir}")
    return hits[0]


def maybe_find_single(run_dir: Path, pattern: str) -> Optional[Path]:
    hits = sorted(run_dir.glob(pattern))
    return hits[0] if hits else None


def load_condition_summary(run_dir: Path) -> Dict[str, Any]:
    rows = read_csv_rows(find_single(run_dir, "*_condition_summary.csv"))
    return rows[0] if rows else {}


def load_trials(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(find_single(run_dir, "*_all_trials.csv"))


def load_debug_rows(run_dir: Path) -> List[Dict[str, Any]]:
    path = maybe_find_single(run_dir, "*_debug_trace.jsonl")
    if path is None:
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def split_pre_post(trials: pd.DataFrame, shock_trial: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pre_df = trials[trials["trial"] < shock_trial].copy()
    shock_df = trials[trials["trial"] == shock_trial].copy()
    post_df = trials[trials["trial"] > shock_trial].copy()
    return pre_df, shock_df, post_df


def path_to_choice_label(path_name: str) -> str:
    return "open" if path_name == "open" else "covered"


def extract_first_post_shot_debug_metrics(run_dir: Path, target_path: str, shock_trial: int) -> Dict[str, float]:
    rows = load_debug_rows(run_dir)
    rows = [r for r in rows if bool(r.get("real_junction_choice_row", False)) and int(r.get("trial", -1)) > shock_trial]
    if not rows:
        return {
            "first_post_target_prob_mean": float("nan"),
            "first_post_target_prob_wins_rate": float("nan"),
            "first_post_target_choice_rate": float("nan"),
            "first_post_target_lb_mean": float("nan"),
        }

    per_seed: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        seed = int(row.get("seed", -1))
        if seed not in per_seed:
            per_seed[seed] = row

    selected = list(per_seed.values())
    if target_path == "covered":
        idx = 1
        target_sid = "path_covered"
    else:
        idx = 0
        target_sid = "path_open"

    probs: List[float] = []
    wins: List[float] = []
    choices: List[float] = []
    bonuses: List[float] = []

    for row in selected:
        action_probs = row.get("action_probs", []) or []
        local_bonus = row.get("local_bonus_values", []) or []
        target_prob = as_float(action_probs[idx] if len(action_probs) > idx else float("nan"))
        other_prob = as_float(action_probs[1 - idx] if len(action_probs) > (1 - idx) else float("nan"))
        probs.append(target_prob)
        wins.append(1.0 if pd.notna(target_prob) and pd.notna(other_prob) and target_prob > other_prob else 0.0)

        chosen_sid = str(row.get("committed_source_id") or row.get("candidate_source_id") or "")
        choices.append(1.0 if chosen_sid == target_sid else 0.0)
        bonuses.append(as_float(local_bonus[idx] if len(local_bonus) > idx else float("nan")))

    return {
        "first_post_target_prob_mean": float(pd.Series(probs).mean()),
        "first_post_target_prob_wins_rate": float(pd.Series(wins).mean()),
        "first_post_target_choice_rate": float(pd.Series(choices).mean()),
        "first_post_target_lb_mean": float(pd.Series(bonuses).mean()),
    }


def summarize_balanced_run(spec: RunSpec) -> Dict[str, Any]:
    row = load_condition_summary(spec.run_dir)
    return {
        "value": spec.value,
        "balanced_p_open": as_float(row.get("p_open")),
        "balanced_p_covered": as_float(row.get("p_covered")),
        "balanced_timeout": as_float(row.get("p_commit_timeout")),
        "balanced_latency": as_float(row.get("mean_commit_latency")),
        "balanced_pause": as_float(row.get("mean_junction_pause_duration")),
        "balanced_reorientation": as_float(row.get("mean_reorientation_count")),
        "balanced_run_dir": str(spec.run_dir),
    }


def summarize_one_shot_run(spec: RunSpec, target_path: str, shock_trial: int) -> Dict[str, Any]:
    trials = load_trials(spec.run_dir)
    pre_df, _shock_df, post_df = split_pre_post(trials, shock_trial)
    target_choice = path_to_choice_label(target_path)

    pre_p_target = float((pre_df["path_choice"] == target_choice).mean()) if len(pre_df) else float("nan")
    post_p_target = float((post_df["path_choice"] == target_choice).mean()) if len(post_df) else float("nan")
    delta_target = post_p_target - pre_p_target if pd.notna(post_p_target) and pd.notna(pre_p_target) else float("nan")
    post_timeout = float((post_df["commit_reason"] == "timeout").mean()) if len(post_df) else float("nan")
    post_latency = float(post_df["commit_latency"].mean()) if len(post_df) else float("nan")
    post_pause = float(post_df["junction_pause_duration"].mean()) if len(post_df) else float("nan")

    dbg = extract_first_post_shot_debug_metrics(spec.run_dir, target_path=target_path, shock_trial=shock_trial)

    return {
        "value": spec.value,
        "one_shot_pre_p_target": pre_p_target,
        "one_shot_post_p_target": post_p_target,
        "one_shot_delta_post_minus_pre_target": delta_target,
        "one_shot_post_timeout": post_timeout,
        "one_shot_post_latency": post_latency,
        "one_shot_post_pause": post_pause,
        "one_shot_run_dir": str(spec.run_dir),
        **dbg,
    }


def merge_summaries(balanced_specs: Sequence[RunSpec], one_shot_specs: Sequence[RunSpec], target_path: str, shock_trial: int) -> pd.DataFrame:
    rows: Dict[float, Dict[str, Any]] = {}
    for spec in balanced_specs:
        rows.setdefault(spec.value, {"value": spec.value}).update(summarize_balanced_run(spec))
    for spec in one_shot_specs:
        rows.setdefault(spec.value, {"value": spec.value}).update(summarize_one_shot_run(spec, target_path=target_path, shock_trial=shock_trial))
    return pd.DataFrame(list(rows.values())).sort_values("value").reset_index(drop=True)


def save_table(df: pd.DataFrame, output_dir: Path, short_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"Table_3_1B_{short_name}_Sweep.csv"
    df.to_csv(out_path, index=False)
    print(f"✓ Sweep table saved: {out_path}")
    return out_path


def plot_metric(df: pd.DataFrame, x_col: str, y_col: str, output_path: Path, title: str, ylabel: str) -> None:
    if y_col not in df.columns or df[y_col].dropna().empty:
        return
    plt.figure(figsize=(7, 5))
    plt.plot(df[x_col], df[y_col], marker="o")
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ Figure saved: {output_path}")


def plot_suite(df: pd.DataFrame, output_dir: Path, short_name: str, target_path: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    x_col = "value"

    specs = [
        (
            "balanced_p_open",
            output_dir / f"Figure_3_1B_{short_name}_balanced_p_open.png",
            f"Stage 3.1B: balanced_conflict P(open) vs {short_name}",
            "P(open)",
        ),
        (
            "balanced_timeout",
            output_dir / f"Figure_3_1B_{short_name}_balanced_timeout.png",
            f"Stage 3.1B: balanced_conflict timeout vs {short_name}",
            "P(timeout)",
        ),
        (
            "one_shot_post_p_target",
            output_dir / f"Figure_3_1B_{short_name}_one_shot_post_p_target.png",
            f"Stage 3.1B: one-shot post P(target={target_path}) vs {short_name}",
            f"Post P(target={target_path})",
        ),
        (
            "one_shot_delta_post_minus_pre_target",
            output_dir / f"Figure_3_1B_{short_name}_one_shot_delta_post_minus_pre_target.png",
            f"Stage 3.1B: one-shot delta target(post-pre) vs {short_name}",
            "Delta post-pre",
        ),
        (
            "one_shot_post_timeout",
            output_dir / f"Figure_3_1B_{short_name}_one_shot_post_timeout.png",
            f"Stage 3.1B: one-shot post timeout vs {short_name}",
            "Post P(timeout)",
        ),
        (
            "first_post_target_prob_mean",
            output_dir / f"Figure_3_1B_{short_name}_one_shot_first_target_prob.png",
            f"Stage 3.1B: first post-shot target probability vs {short_name}",
            "First post-shot target probability",
        ),
        (
            "first_post_target_choice_rate",
            output_dir / f"Figure_3_1B_{short_name}_one_shot_first_target_choice_rate.png",
            f"Stage 3.1B: first post-shot target choice rate vs {short_name}",
            "First post-shot target choice rate",
        ),
        (
            "first_post_target_prob_wins_rate",
            output_dir / f"Figure_3_1B_{short_name}_one_shot_first_target_prob_wins_rate.png",
            f"Stage 3.1B: first post-shot target win-rate vs {short_name}",
            "First post-shot target probability wins rate",
        ),
        (
            "first_post_target_lb_mean",
            output_dir / f"Figure_3_1B_{short_name}_one_shot_first_target_lb_mean.png",
            f"Stage 3.1B: first post-shot target local bonus vs {short_name}",
            "First post-shot target local bonus",
        ),
    ]

    for y_col, out_path, title, ylabel in specs:
        plot_metric(df, x_col, y_col, out_path, title, ylabel)


def print_summary(df: pd.DataFrame) -> None:
    cols = [c for c in [
        "value",
        "balanced_p_open",
        "balanced_timeout",
        "one_shot_post_p_target",
        "one_shot_delta_post_minus_pre_target",
        "first_post_target_prob_mean",
        "first_post_target_choice_rate",
        "first_post_target_prob_wins_rate",
        "one_shot_post_timeout",
    ] if c in df.columns]
    print("\n=== Sweep Summary ===")
    print(df[cols].to_string(index=False))

    required = [
        "first_post_target_choice_rate",
        "one_shot_delta_post_minus_pre_target",
        "one_shot_post_timeout",
        "balanced_timeout",
    ]
    if all(c in df.columns for c in required):
        work = df.dropna(subset=required).copy()
        if len(work):
            work["score"] = (
                1.00 * work["first_post_target_choice_rate"]
                + 0.75 * work["one_shot_delta_post_minus_pre_target"]
                - 0.50 * work["one_shot_post_timeout"]
                - 0.25 * work["balanced_timeout"]
            )
            best = work.sort_values("score", ascending=False).iloc[0]
            print("\n=== Heuristic Best Value ===")
            print(
                f"value={best['value']:.6g}, "
                f"first_post_target_choice_rate={best['first_post_target_choice_rate']:.3f}, "
                f"delta_post_minus_pre_target={best['one_shot_delta_post_minus_pre_target']:.3f}, "
                f"one_shot_post_timeout={best['one_shot_post_timeout']:.3f}, "
                f"balanced_timeout={best['balanced_timeout']:.3f}"
            )


def main() -> None:
    args = parse_args()
    if args.manifest:
        param, short_name, target_path, shock_trial, balanced_specs, one_shot_specs = load_manifest(Path(args.manifest))
    else:
        param = args.param or "sweep"
        short_name = args.short_name
        target_path = args.target_path
        shock_trial = args.shock_trial
        balanced_specs = [parse_run_arg(x) for x in args.run_balanced]
        one_shot_specs = [parse_run_arg(x) for x in args.run_one_shot]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = merge_summaries(balanced_specs, one_shot_specs, target_path=target_path, shock_trial=shock_trial)
    save_table(df, output_dir, short_name)
    plot_suite(df, output_dir, short_name=short_name, target_path=target_path)
    print_summary(df)

    meta = {
        "param": param,
        "short_name": short_name,
        "target_path": target_path,
        "shock_trial": shock_trial,
        "n_rows": int(len(df)),
    }
    meta_path = output_dir / "analysis_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Analysis metadata saved: {meta_path}")


if __name__ == "__main__":
    main()
