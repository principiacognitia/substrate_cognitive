#!/usr/bin/env python3
"""
Analyze Stage 3.1B ablation suite outputs.

Inputs are expected from manifest.json produced by
run_stage3_1b_ablation_suite.py.

This analyzer computes per-seed metrics for:
- balanced_conflict baseline
- one-shot shock on open
- one-shot treat on covered

and then builds ablation-level summaries with bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class RunSpec:
    ablation: str
    run_dir: Path


SEED_METRIC_COLUMNS = [
    "seed",
    "metric_value",
    "scenario",
    "metric",
    "ablation",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze Stage 3.1B ablation suite")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--bootstrap-samples", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=123)
    return ap.parse_args()


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


def path_to_choice_label(path_name: str) -> str:
    return "open" if path_name == "open" else "covered"


def split_pre_post(trials: pd.DataFrame, shock_trial: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pre_df = trials[trials["trial"] < shock_trial].copy()
    shock_df = trials[trials["trial"] == shock_trial].copy()
    post_df = trials[trials["trial"] > shock_trial].copy()
    return pre_df, shock_df, post_df


def bootstrap_ci(values: Sequence[float], n_samples: int, rng: np.random.Generator) -> Tuple[float, float, float]:
    arr = np.asarray([x for x in values if pd.notna(x)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, mean, mean
    idx = rng.integers(0, arr.size, size=(n_samples, arr.size))
    sample_means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(sample_means, [0.025, 0.975])
    return mean, float(lo), float(hi)


def load_manifest(path: Path) -> Tuple[Dict[str, Any], List[RunSpec], List[RunSpec], List[RunSpec]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    balanced = [RunSpec(str(x["ablation"]), Path(x["run_dir"])) for x in data.get("balanced_runs", [])]
    shock = [RunSpec(str(x["ablation"]), Path(x["run_dir"])) for x in data.get("shock_runs", [])]
    treat = [RunSpec(str(x["ablation"]), Path(x["run_dir"])) for x in data.get("treat_runs", [])]
    return data, balanced, shock, treat


def collect_balanced_seed_metrics(spec: RunSpec) -> pd.DataFrame:
    trials = load_trials(spec.run_dir)
    rows: List[Dict[str, Any]] = []
    grouped = trials.groupby("seed", dropna=False)
    for seed, sdf in grouped:
        rows.extend([
            {"seed": int(seed), "metric_value": float((sdf["path_choice"] == "open").mean()), "scenario": "balanced", "metric": "p_open", "ablation": spec.ablation},
            {"seed": int(seed), "metric_value": float((sdf["commit_reason"] == "timeout").mean()), "scenario": "balanced", "metric": "p_timeout", "ablation": spec.ablation},
            {"seed": int(seed), "metric_value": float(sdf["commit_latency"].mean()), "scenario": "balanced", "metric": "commit_latency", "ablation": spec.ablation},
            {"seed": int(seed), "metric_value": float(sdf["junction_pause_duration"].mean()), "scenario": "balanced", "metric": "junction_pause", "ablation": spec.ablation},
            {"seed": int(seed), "metric_value": float(sdf["reorientation_count"].mean()), "scenario": "balanced", "metric": "reorientation", "ablation": spec.ablation},
        ])
    return pd.DataFrame(rows, columns=SEED_METRIC_COLUMNS)


def first_post_shot_debug_by_seed(run_dir: Path, target_path: str, shock_trial: int) -> pd.DataFrame:
    rows = load_debug_rows(run_dir)
    rows = [r for r in rows if bool(r.get("real_junction_choice_row", False)) and int(r.get("trial", -1)) > shock_trial]
    if not rows:
        return pd.DataFrame(columns=["seed", "target_prob", "target_prob_wins", "target_choice", "target_lb"])

    first_by_seed: Dict[int, Dict[str, Any]] = {}
    for row in sorted(rows, key=lambda r: (int(r.get("seed", -1)), int(r.get("trial", -1)), int(r.get("tick", -1)))):
        seed = int(row.get("seed", -1))
        if seed not in first_by_seed:
            first_by_seed[seed] = row

    idx = 0 if target_path == "open" else 1
    target_sid = "path_open" if target_path == "open" else "path_covered"

    out_rows: List[Dict[str, Any]] = []
    for seed, row in sorted(first_by_seed.items()):
        action_probs = row.get("action_probs", []) or []
        local_bonus = row.get("local_bonus_values", []) or []
        target_prob = float(action_probs[idx]) if len(action_probs) > idx else float("nan")
        other_idx = 1 - idx
        other_prob = float(action_probs[other_idx]) if len(action_probs) > other_idx else float("nan")
        chosen_sid = str(row.get("committed_source_id") or row.get("candidate_source_id") or "")
        out_rows.append({
            "seed": seed,
            "target_prob": target_prob,
            "target_prob_wins": 1.0 if pd.notna(target_prob) and pd.notna(other_prob) and target_prob > other_prob else 0.0,
            "target_choice": 1.0 if chosen_sid == target_sid else 0.0,
            "target_lb": float(local_bonus[idx]) if len(local_bonus) > idx else float("nan"),
        })
    return pd.DataFrame(out_rows)


def collect_one_shot_seed_metrics(spec: RunSpec, target_path: str, shock_trial: int, scenario_name: str) -> pd.DataFrame:
    trials = load_trials(spec.run_dir)
    target_choice = path_to_choice_label(target_path)
    rows: List[Dict[str, Any]] = []
    grouped = trials.groupby("seed", dropna=False)
    debug_df = first_post_shot_debug_by_seed(spec.run_dir, target_path=target_path, shock_trial=shock_trial)
    debug_map = debug_df.set_index("seed").to_dict(orient="index") if not debug_df.empty else {}

    for seed, sdf in grouped:
        pre_df, _shock_df, post_df = split_pre_post(sdf, shock_trial)
        pre_p_target = float((pre_df["path_choice"] == target_choice).mean()) if len(pre_df) else float("nan")
        post_p_target = float((post_df["path_choice"] == target_choice).mean()) if len(post_df) else float("nan")
        delta = post_p_target - pre_p_target if pd.notna(pre_p_target) and pd.notna(post_p_target) else float("nan")
        post_timeout = float((post_df["commit_reason"] == "timeout").mean()) if len(post_df) else float("nan")
        post_latency = float(post_df["commit_latency"].mean()) if len(post_df) else float("nan")
        post_pause = float(post_df["junction_pause_duration"].mean()) if len(post_df) else float("nan")

        seed_int = int(seed)
        dbg = debug_map.get(seed_int, {})
        rows.extend([
            {"seed": seed_int, "metric_value": pre_p_target, "scenario": scenario_name, "metric": "pre_p_target", "ablation": spec.ablation},
            {"seed": seed_int, "metric_value": post_p_target, "scenario": scenario_name, "metric": "post_p_target", "ablation": spec.ablation},
            {"seed": seed_int, "metric_value": delta, "scenario": scenario_name, "metric": "delta_post_minus_pre_target", "ablation": spec.ablation},
            {"seed": seed_int, "metric_value": post_timeout, "scenario": scenario_name, "metric": "post_timeout", "ablation": spec.ablation},
            {"seed": seed_int, "metric_value": post_latency, "scenario": scenario_name, "metric": "post_latency", "ablation": spec.ablation},
            {"seed": seed_int, "metric_value": post_pause, "scenario": scenario_name, "metric": "post_pause", "ablation": spec.ablation},
            {"seed": seed_int, "metric_value": float(dbg.get("target_prob", float("nan"))), "scenario": scenario_name, "metric": "first_post_target_prob", "ablation": spec.ablation},
            {"seed": seed_int, "metric_value": float(dbg.get("target_prob_wins", float("nan"))), "scenario": scenario_name, "metric": "first_post_target_prob_wins", "ablation": spec.ablation},
            {"seed": seed_int, "metric_value": float(dbg.get("target_choice", float("nan"))), "scenario": scenario_name, "metric": "first_post_target_choice", "ablation": spec.ablation},
            {"seed": seed_int, "metric_value": float(dbg.get("target_lb", float("nan"))), "scenario": scenario_name, "metric": "first_post_target_lb", "ablation": spec.ablation},
        ])
    return pd.DataFrame(rows, columns=SEED_METRIC_COLUMNS)


def summarize_seed_metrics(seed_df: pd.DataFrame, bootstrap_samples: int, rng: np.random.Generator) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if seed_df.empty:
        return pd.DataFrame()
    grouped = seed_df.groupby(["scenario", "metric", "ablation"], dropna=False)
    for (scenario, metric, ablation), sdf in grouped:
        values = list(sdf["metric_value"].astype(float))
        mean, ci_low, ci_high = bootstrap_ci(values, bootstrap_samples, rng)
        rows.append({
            "scenario": scenario,
            "metric": metric,
            "ablation": ablation,
            "n_seeds": int(sdf["seed"].nunique()),
            "mean": mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })
    return pd.DataFrame(rows).sort_values(["scenario", "metric", "ablation"]).reset_index(drop=True)


def make_summary_wide(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: Dict[str, Dict[str, Any]] = {}
    for _, row in summary_df.iterrows():
        ablation = str(row["ablation"])
        rows.setdefault(ablation, {"ablation": ablation})
        prefix = f"{row['scenario']}_{row['metric']}"
        rows[ablation][f"{prefix}_mean"] = float(row["mean"])
        rows[ablation][f"{prefix}_ci_low"] = float(row["ci_low"])
        rows[ablation][f"{prefix}_ci_high"] = float(row["ci_high"])
    return pd.DataFrame(list(rows.values())).sort_values("ablation").reset_index(drop=True)


def plot_metric(summary_df: pd.DataFrame, *, scenario: str, metric: str, ablation_order: Sequence[str], out_path: Path, title: str, ylabel: str) -> None:
    sdf = summary_df[(summary_df["scenario"] == scenario) & (summary_df["metric"] == metric)].copy()
    if sdf.empty:
        return
    sdf["ablation"] = pd.Categorical(sdf["ablation"], categories=list(ablation_order), ordered=True)
    sdf = sdf.sort_values("ablation")
    x = np.arange(len(sdf))
    y = sdf["mean"].to_numpy(dtype=float)
    yerr = np.vstack([
        y - sdf["ci_low"].to_numpy(dtype=float),
        sdf["ci_high"].to_numpy(dtype=float) - y,
    ])

    plt.figure(figsize=(8, 5))
    plt.errorbar(x, y, yerr=yerr, fmt="o-", capsize=4)
    plt.xticks(x, list(sdf["ablation"]), rotation=25, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("ablation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✓ Figure saved: {out_path}")


def plot_suite(summary_df: pd.DataFrame, output_dir: Path, ablation_order: Sequence[str]) -> None:
    specs = [
        ("balanced", "p_open", "Figure_3_1B_ablation_balanced_p_open.png", "Stage 3.1B: balanced_conflict P(open) by ablation", "P(open)"),
        ("balanced", "p_timeout", "Figure_3_1B_ablation_balanced_p_timeout.png", "Stage 3.1B: balanced_conflict P(timeout) by ablation", "P(timeout)"),
        ("balanced", "commit_latency", "Figure_3_1B_ablation_balanced_commit_latency.png", "Stage 3.1B: balanced_conflict latency by ablation", "Commit latency"),
        ("shock", "delta_post_minus_pre_target", "Figure_3_1B_ablation_shock_delta.png", "Stage 3.1B: shock post-pre target shift by ablation", "Delta post-pre"),
        ("shock", "post_timeout", "Figure_3_1B_ablation_shock_post_timeout.png", "Stage 3.1B: shock post P(timeout) by ablation", "Post P(timeout)"),
        ("treat", "delta_post_minus_pre_target", "Figure_3_1B_ablation_treat_delta.png", "Stage 3.1B: treat post-pre target shift by ablation", "Delta post-pre"),
        ("treat", "first_post_target_prob", "Figure_3_1B_ablation_treat_first_target_prob.png", "Stage 3.1B: treat first post-shot target probability by ablation", "First post-shot target probability"),
        ("treat", "first_post_target_choice", "Figure_3_1B_ablation_treat_first_target_choice.png", "Stage 3.1B: treat first post-shot target choice rate by ablation", "First post-shot target choice rate"),
        ("treat", "first_post_target_lb", "Figure_3_1B_ablation_treat_first_target_lb.png", "Stage 3.1B: treat first post-shot target local bonus by ablation", "First post-shot target local bonus"),
        ("treat", "post_timeout", "Figure_3_1B_ablation_treat_post_timeout.png", "Stage 3.1B: treat post P(timeout) by ablation", "Post P(timeout)"),
    ]
    for scenario, metric, filename, title, ylabel in specs:
        plot_metric(
            summary_df,
            scenario=scenario,
            metric=metric,
            ablation_order=ablation_order,
            out_path=output_dir / filename,
            title=title,
            ylabel=ylabel,
        )


def print_summary(wide_df: pd.DataFrame) -> None:
    keep_cols = [c for c in [
        "ablation",
        "balanced_p_open_mean",
        "balanced_p_timeout_mean",
        "shock_delta_post_minus_pre_target_mean",
        "shock_post_timeout_mean",
        "treat_delta_post_minus_pre_target_mean",
        "treat_first_post_target_prob_mean",
        "treat_first_post_target_choice_mean",
        "treat_first_post_target_lb_mean",
        "treat_post_timeout_mean",
    ] if c in wide_df.columns]
    if keep_cols:
        print("\n=== Ablation Summary ===")
        print(wide_df[keep_cols].to_string(index=False))


def main() -> None:
    args = parse_args()
    manifest, balanced_specs, shock_specs, treat_specs = load_manifest(Path(args.manifest))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.bootstrap_seed)
    frames: List[pd.DataFrame] = []

    for spec in balanced_specs:
        frames.append(collect_balanced_seed_metrics(spec))
    for spec in shock_specs:
        frames.append(collect_one_shot_seed_metrics(spec, target_path=str(manifest.get("shock_target_path", "open")), shock_trial=int(manifest.get("shock_trial", 30)), scenario_name="shock"))
    for spec in treat_specs:
        frames.append(collect_one_shot_seed_metrics(spec, target_path=str(manifest.get("treat_target_path", "covered")), shock_trial=int(manifest.get("shock_trial", 30)), scenario_name="treat"))

    seed_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SEED_METRIC_COLUMNS)
    summary_df = summarize_seed_metrics(seed_df, bootstrap_samples=args.bootstrap_samples, rng=rng)
    wide_df = make_summary_wide(summary_df)

    seed_path = output_dir / "Table_3_1B_ablation_seed_metrics.csv"
    summary_long_path = output_dir / "Table_3_1B_ablation_summary_long.csv"
    summary_wide_path = output_dir / "Table_3_1B_ablation_summary_wide.csv"

    seed_df.to_csv(seed_path, index=False)
    summary_df.to_csv(summary_long_path, index=False)
    wide_df.to_csv(summary_wide_path, index=False)

    print(f"✓ Seed metrics saved: {seed_path}")
    print(f"✓ Long summary saved: {summary_long_path}")
    print(f"✓ Wide summary saved: {summary_wide_path}")

    plot_suite(summary_df, output_dir=output_dir, ablation_order=list(manifest.get("ablations", [])))
    print_summary(wide_df)

    meta = {
        "manifest": str(Path(args.manifest)),
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "n_seed_rows": int(len(seed_df)),
        "n_summary_rows": int(len(summary_df)),
    }
    meta_path = output_dir / "analysis_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"✓ Analysis metadata saved: {meta_path}")


if __name__ == "__main__":
    main()
