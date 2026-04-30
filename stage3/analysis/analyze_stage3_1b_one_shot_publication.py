#!/usr/bin/env python3
"""
Stage 3.1B one-shot publication analyzer.

This analyzer is protocol-aware:
- shock protocol is treated as the negative branch and requires h_risk/q_neg
- treat protocol is treated as the positive branch and requires h_opp/q_pos

It reads raw outputs produced by run_stage3_1b_ablation_suite.py and writes
publication-oriented tables/figures for the Stage 3.1B closure package.

This is analysis-only. It does not change the agent, environment, runner, or
experimental protocol.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WINDOW_ORDER = [
    "pre",
    "shock",
    "post_1_3",
    "post_4_10",
    "post_11_30",
    "post_31_plus",
    "post_all",
]


@dataclass
class OneShotRunSpec:
    protocol: str
    ablation: str
    run_dir: Path
    target_path: str
    expected_direction: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze Stage 3.1B one-shot publication artifacts")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--bootstrap-samples", type=int, default=5000)
    ap.add_argument("--bootstrap-seed", type=int, default=123)
    ap.add_argument("--signflip-samples", type=int, default=20000)
    ap.add_argument("--placebo-samples", type=int, default=5000)
    ap.add_argument("--zoom-pre", type=int, default=5)
    ap.add_argument("--zoom-post", type=int, default=30)
    return ap.parse_args()


def find_single(run_dir: Path, pattern: str) -> Path:
    hits = sorted(run_dir.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No file matching {pattern} in {run_dir}")
    return hits[0]


def maybe_find_single(run_dir: Path, pattern: str) -> Optional[Path]:
    hits = sorted(run_dir.glob(pattern))
    return hits[0] if hits else None


def load_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_run_specs(manifest: Dict[str, Any]) -> List[OneShotRunSpec]:
    shock_target_path = str(manifest.get("shock_target_path", "open"))
    treat_target_path = str(manifest.get("treat_target_path", "covered"))

    specs: List[OneShotRunSpec] = []

    for item in manifest.get("shock_runs", []):
        specs.append(
            OneShotRunSpec(
                protocol="shock",
                ablation=str(item["ablation"]),
                run_dir=Path(item["run_dir"]),
                target_path=shock_target_path,
                expected_direction="negative",
            )
        )

    for item in manifest.get("treat_runs", []):
        specs.append(
            OneShotRunSpec(
                protocol="treat",
                ablation=str(item["ablation"]),
                run_dir=Path(item["run_dir"]),
                target_path=treat_target_path,
                expected_direction="positive",
            )
        )

    return specs


def target_choice_label(target_path: str) -> str:
    if target_path not in {"open", "covered"}:
        raise ValueError(f"Unknown target_path: {target_path}")
    return target_path


def target_index(target_path: str) -> int:
    if target_path == "open":
        return 0
    if target_path == "covered":
        return 1
    raise ValueError(f"Unknown target_path: {target_path}")


def target_source_id(target_path: str) -> str:
    if target_path == "open":
        return "path_open"
    if target_path == "covered":
        return "path_covered"
    raise ValueError(f"Unknown target_path: {target_path}")


def load_trials(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(find_single(run_dir, "*_all_trials.csv"))


def load_steps(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(find_single(run_dir, "*_all_steps.csv"))


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


def validate_schema(spec: OneShotRunSpec, trials: pd.DataFrame, steps: pd.DataFrame) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    common_trial_required = {
        "seed",
        "trial",
        "path_choice",
        "commit_reason",
        "commit_latency",
        "junction_pause_duration",
    }
    common_step_required = {
        "seed",
        "trial",
        "tick",
        "h_risk",
        "h_opp",
        "q_neg",
        "q_pos",
    }

    protocol_required = {
        "shock": {"h_risk", "q_neg"},
        "treat": {"h_opp", "q_pos"},
    }[spec.protocol]

    checks = [
        ("trials", common_trial_required, set(trials.columns)),
        ("steps_common", common_step_required, set(steps.columns)),
        (f"steps_{spec.protocol}_semantic", protocol_required, set(steps.columns)),
    ]

    for check_name, required, available in checks:
        missing = sorted(required - available)
        ok = len(missing) == 0
        records.append(
            {
                "protocol": spec.protocol,
                "ablation": spec.ablation,
                "run_dir": str(spec.run_dir),
                "check": check_name,
                "ok": bool(ok),
                "missing_columns": ",".join(missing),
            }
        )
        if not ok:
            raise ValueError(
                f"{spec.protocol}/{spec.ablation}: missing columns for {check_name}: {missing}"
            )

    if spec.protocol == "treat":
        # A semantic guard against old negative-branch-only analyzers.
        negative_only = {"h_risk", "q_neg"}.issubset(set(steps.columns)) and not {"h_opp", "q_pos"}.issubset(set(steps.columns))
        if negative_only:
            raise ValueError(
                f"{spec.protocol}/{spec.ablation}: treat protocol cannot be analyzed as negative-only."
            )

    return records


def assign_window(trial: int, shock_trial: int) -> str:
    rel = int(trial) - int(shock_trial)
    if rel < 0:
        return "pre"
    if rel == 0:
        return "shock"
    if 1 <= rel <= 3:
        return "post_1_3"
    if 4 <= rel <= 10:
        return "post_4_10"
    if 11 <= rel <= 30:
        return "post_11_30"
    return "post_31_plus"


def add_windows(df: pd.DataFrame, shock_trial: int) -> pd.DataFrame:
    out = df.copy()
    out["rel_trial"] = out["trial"].astype(int) - int(shock_trial)
    out["window"] = out["trial"].map(lambda x: assign_window(int(x), shock_trial))
    return out


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


def sign_flip_pvalue(
    values: Sequence[float],
    expected_direction: str,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    arr = np.asarray([x for x in values if pd.notna(x)], dtype=float)
    if arr.size == 0:
        return float("nan")

    observed = float(arr.mean())
    observed_stat = -observed if expected_direction == "negative" else observed

    if arr.size <= 18:
        # Exact sign-flip enumeration is cheap up to 2^18.
        n = int(arr.size)
        stats = []
        for mask in range(2 ** n):
            signs = np.array([1.0 if (mask >> i) & 1 else -1.0 for i in range(n)])
            mean = float((arr * signs).mean())
            stats.append(-mean if expected_direction == "negative" else mean)
        null_stats = np.asarray(stats, dtype=float)
    else:
        signs = rng.choice([-1.0, 1.0], size=(n_samples, arr.size))
        means = (signs * arr).mean(axis=1)
        null_stats = -means if expected_direction == "negative" else means

    return float((np.sum(null_stats >= observed_stat) + 1.0) / (len(null_stats) + 1.0))


def collect_trial_series(
    spec: OneShotRunSpec,
    trials: pd.DataFrame,
    steps: pd.DataFrame,
    shock_trial: int,
) -> pd.DataFrame:
    target = target_choice_label(spec.target_path)

    trial_df = (
        trials.groupby("trial", dropna=False)
        .agg(
            n_trials=("trial", "size"),
            p_open=("path_choice", lambda s: float((s == "open").mean())),
            p_covered=("path_choice", lambda s: float((s == "covered").mean())),
            p_target=("path_choice", lambda s: float((s == target).mean())),
            p_timeout=("commit_reason", lambda s: float((s == "timeout").mean())),
            mean_commit_latency=("commit_latency", "mean"),
            mean_junction_pause_duration=("junction_pause_duration", "mean"),
        )
        .reset_index()
    )

    step_agg: Dict[str, Tuple[str, str]] = {}
    for col in ["h_risk", "h_opp", "q_neg", "q_pos", "safe_drive", "X_risk", "X_opp"]:
        if col in steps.columns:
            step_agg[f"mean_{col}"] = (col, "mean")

    step_df = (
        steps.groupby("trial", dropna=False)
        .agg(**step_agg)
        .reset_index()
        if step_agg
        else pd.DataFrame({"trial": sorted(trials["trial"].unique())})
    )

    out = trial_df.merge(step_df, on="trial", how="left")
    out["protocol"] = spec.protocol
    out["ablation"] = spec.ablation
    out["target_path"] = spec.target_path
    out["expected_direction"] = spec.expected_direction
    out["rel_trial"] = out["trial"].astype(int) - int(shock_trial)
    return out.sort_values(["protocol", "ablation", "trial"]).reset_index(drop=True)

def collect_trial_seed_series(
    spec: OneShotRunSpec,
    trials: pd.DataFrame,
    steps: pd.DataFrame,
    shock_trial: int,
) -> pd.DataFrame:
    """
    Seed-level trial series for publication plots.

    For p_target, each seed contributes one binary observation per trial.
    For carriers, steps are first averaged within seed/trial, then SEM is
    computed across seeds. This avoids treating within-trial ticks as
    independent observations.
    """
    target = target_choice_label(spec.target_path)

    trial_df = (
        trials.groupby(["seed", "trial"], dropna=False)
        .agg(
            p_open=("path_choice", lambda s: float((s == "open").mean())),
            p_covered=("path_choice", lambda s: float((s == "covered").mean())),
            p_target=("path_choice", lambda s: float((s == target).mean())),
            p_timeout=("commit_reason", lambda s: float((s == "timeout").mean())),
            commit_latency=("commit_latency", "mean"),
            junction_pause_duration=("junction_pause_duration", "mean"),
        )
        .reset_index()
    )

    step_agg: Dict[str, Tuple[str, str]] = {}
    for col in ["h_risk", "h_opp", "q_neg", "q_pos", "safe_drive", "X_risk", "X_opp"]:
        if col in steps.columns:
            step_agg[col] = (col, "mean")

    step_df = (
        steps.groupby(["seed", "trial"], dropna=False)
        .agg(**step_agg)
        .reset_index()
        if step_agg
        else pd.DataFrame({"seed": [], "trial": []})
    )

    out = trial_df.merge(step_df, on=["seed", "trial"], how="left")
    out["protocol"] = spec.protocol
    out["ablation"] = spec.ablation
    out["target_path"] = spec.target_path
    out["expected_direction"] = spec.expected_direction
    out["rel_trial"] = out["trial"].astype(int) - int(shock_trial)

    return out.sort_values(["protocol", "ablation", "seed", "trial"]).reset_index(drop=True)


def sem(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size <= 1:
        return float("nan")
    return float(arr.std(ddof=1) / np.sqrt(arr.size))

def collect_window_seed_metrics(
    spec: OneShotRunSpec,
    trials: pd.DataFrame,
    steps: pd.DataFrame,
    shock_trial: int,
) -> pd.DataFrame:
    target = target_choice_label(spec.target_path)
    trials_w = add_windows(trials, shock_trial)
    steps_w = add_windows(steps, shock_trial)

    rows: List[Dict[str, Any]] = []

    for seed, sdf in trials_w.groupby("seed", dropna=False):
        seed_int = int(seed)

        for window in WINDOW_ORDER:
            if window == "post_all":
                wdf = sdf[sdf["rel_trial"] > 0]
            else:
                wdf = sdf[sdf["window"] == window]

            if len(wdf) == 0:
                continue

            metrics = {
                "p_target": float((wdf["path_choice"] == target).mean()),
                "p_open": float((wdf["path_choice"] == "open").mean()),
                "p_covered": float((wdf["path_choice"] == "covered").mean()),
                "p_timeout": float((wdf["commit_reason"] == "timeout").mean()),
                "commit_latency": float(wdf["commit_latency"].mean()),
                "junction_pause_duration": float(wdf["junction_pause_duration"].mean()),
            }

            for metric, value in metrics.items():
                rows.append(
                    {
                        "protocol": spec.protocol,
                        "ablation": spec.ablation,
                        "seed": seed_int,
                        "target_path": spec.target_path,
                        "expected_direction": spec.expected_direction,
                        "window": window,
                        "metric": metric,
                        "metric_value": value,
                    }
                )

    for seed, sdf in steps_w.groupby("seed", dropna=False):
        seed_int = int(seed)

        for window in WINDOW_ORDER:
            if window == "post_all":
                wdf = sdf[sdf["rel_trial"] > 0]
            else:
                wdf = sdf[sdf["window"] == window]

            if len(wdf) == 0:
                continue

            for metric in ["h_risk", "h_opp", "q_neg", "q_pos", "safe_drive", "X_risk", "X_opp"]:
                if metric not in wdf.columns:
                    continue
                rows.append(
                    {
                        "protocol": spec.protocol,
                        "ablation": spec.ablation,
                        "seed": seed_int,
                        "target_path": spec.target_path,
                        "expected_direction": spec.expected_direction,
                        "window": window,
                        "metric": metric,
                        "metric_value": float(wdf[metric].mean()),
                    }
                )

    return pd.DataFrame(rows)


def summarize_window_metrics(seed_window_df: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    grouped = seed_window_df.groupby(
        ["protocol", "ablation", "target_path", "expected_direction", "window", "metric"],
        dropna=False,
    )

    for keys, sdf in grouped:
        protocol, ablation, target_path, expected_direction, window, metric = keys
        mean, lo, hi = bootstrap_ci(sdf["metric_value"].tolist(), n_boot, rng)
        rows.append(
            {
                "protocol": protocol,
                "ablation": ablation,
                "target_path": target_path,
                "expected_direction": expected_direction,
                "window": window,
                "metric": metric,
                "n_seeds": int(sdf["seed"].nunique()),
                "mean": mean,
                "ci_low": lo,
                "ci_high": hi,
            }
        )

    out = pd.DataFrame(rows)
    if len(out):
        out["window"] = pd.Categorical(out["window"], categories=WINDOW_ORDER, ordered=True)
        out = out.sort_values(["protocol", "ablation", "metric", "window"]).reset_index(drop=True)
    return out


def collect_first_post_junction(spec: OneShotRunSpec, debug_rows: List[Dict[str, Any]], shock_trial: int) -> pd.DataFrame:
    if not debug_rows:
        return pd.DataFrame()

    idx = target_index(spec.target_path)
    target_sid = target_source_id(spec.target_path)

    candidate_rows = [
        r for r in debug_rows
        if bool(r.get("real_junction_choice_row", False)) and int(r.get("trial", -1)) > int(shock_trial)
    ]

    first_by_seed: Dict[int, Dict[str, Any]] = {}
    for row in sorted(candidate_rows, key=lambda r: (int(r.get("seed", -1)), int(r.get("trial", -1)), int(r.get("tick", -1)))):
        seed = int(row.get("seed", -1))
        if seed not in first_by_seed:
            first_by_seed[seed] = row

    rows: List[Dict[str, Any]] = []

    for seed, row in sorted(first_by_seed.items()):
        action_probs = row.get("action_probs", []) or []
        local_bonus = row.get("local_bonus_values", []) or []

        target_prob = float(action_probs[idx]) if isinstance(action_probs, list) and len(action_probs) > idx else float("nan")
        other_prob = float(action_probs[1 - idx]) if isinstance(action_probs, list) and len(action_probs) > 1 - idx else float("nan")
        target_lb = float(local_bonus[idx]) if isinstance(local_bonus, list) and len(local_bonus) > idx else float("nan")

        chosen_sid = str(row.get("committed_source_id") or row.get("candidate_source_id") or "")

        rows.append(
            {
                "protocol": spec.protocol,
                "ablation": spec.ablation,
                "seed": seed,
                "target_path": spec.target_path,
                "expected_direction": spec.expected_direction,
                "first_post_trial": int(row.get("trial", -1)),
                "first_post_tick": int(row.get("tick", -1)),
                "target_prob": target_prob,
                "target_prob_wins": bool(pd.notna(target_prob) and pd.notna(other_prob) and target_prob > other_prob),
                "target_choice": bool(chosen_sid == target_sid),
                "target_local_bonus": target_lb,
                "mode": str(row.get("mode", "")),
                "gate_trigger": str(row.get("gate_trigger", "")),
                "h_risk": float(row.get("h_risk", float("nan"))),
                "h_opp": float(row.get("h_opp", float("nan"))),
                "q_neg": float(row.get("q_neg", float("nan"))),
                "q_pos": float(row.get("q_pos", float("nan"))),
                "safe_drive": float(row.get("safe_drive", float("nan"))),
                "one_shot_type": str(row.get("one_shot_type", "")),
            }
        )

    return pd.DataFrame(rows)


def build_effect_stats(
    seed_window_df: pd.DataFrame,
    n_boot: int,
    n_signflip: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    base = seed_window_df[seed_window_df["metric"] == "p_target"].copy()
    if base.empty:
        return pd.DataFrame()

    grouped = base.groupby(["protocol", "ablation", "target_path", "expected_direction"], dropna=False)

    for keys, sdf in grouped:
        protocol, ablation, target_path, expected_direction = keys

        pivot = sdf.pivot_table(
            index="seed",
            columns="window",
            values="metric_value",
            aggfunc="mean",
        )

        if "pre" not in pivot.columns:
            continue

        for post_window in ["post_1_3", "post_4_10", "post_11_30", "post_31_plus", "post_all"]:
            if post_window not in pivot.columns:
                continue

            delta = pivot[post_window] - pivot["pre"]
            delta = delta.dropna()
            if len(delta) == 0:
                continue

            mean, lo, hi = bootstrap_ci(delta.tolist(), n_boot, rng)
            p_value = sign_flip_pvalue(
                delta.tolist(),
                expected_direction=str(expected_direction),
                n_samples=n_signflip,
                rng=rng,
            )

            if expected_direction == "negative":
                sign_consistency = float((delta < 0).mean())
            else:
                sign_consistency = float((delta > 0).mean())

            rows.append(
                {
                    "protocol": protocol,
                    "ablation": ablation,
                    "target_path": target_path,
                    "expected_direction": expected_direction,
                    "post_window": post_window,
                    "n_seeds": int(len(delta)),
                    "pre_mean": float(pivot["pre"].mean()),
                    "post_mean": float(pivot[post_window].mean()),
                    "delta_post_minus_pre_mean": mean,
                    "delta_ci_low": lo,
                    "delta_ci_high": hi,
                    "directional_sign_consistency": sign_consistency,
                    "directional_sign_flip_p": p_value,
                }
            )

    return pd.DataFrame(rows)

def expected_sign_ok(value: float, expected_direction: str) -> bool:
    if pd.isna(value):
        return False
    if expected_direction == "negative":
        return float(value) < 0.0
    if expected_direction == "positive":
        return float(value) > 0.0
    return False


def expected_prob_ok(value: float, expected_direction: str) -> bool:
    if pd.isna(value):
        return False
    if expected_direction == "negative":
        return float(value) < 0.5
    if expected_direction == "positive":
        return float(value) > 0.5
    return False


def status_for_direction(sign_ok: bool, p_value: float, ci_low: float, ci_high: float, expected_direction: str) -> str:
    if not sign_ok:
        return "fail"

    ci_supports = False
    if expected_direction == "negative":
        ci_supports = pd.notna(ci_high) and float(ci_high) < 0.0
    elif expected_direction == "positive":
        ci_supports = pd.notna(ci_low) and float(ci_low) > 0.0

    p_supports = pd.notna(p_value) and float(p_value) <= 0.05

    if ci_supports and p_supports:
        return "pass"
    return "direction_ok"


def build_acceptance_summary(
    schema_df: pd.DataFrame,
    effect_df: pd.DataFrame,
    first_post_df: pd.DataFrame,
    placebo_df: pd.DataFrame,
    carrier_effect_df: pd.DataFrame,
    ablation_localization_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compact, paper-facing acceptance summary.

    This table is intentionally descriptive. It does not crash smoke runs if
    n=3 is underpowered. The full paper-grade run should be judged using these
    rows plus the detailed effect/window tables.
    """
    rows: List[Dict[str, Any]] = []

    if len(schema_df):
        for (protocol, ablation), sdf in schema_df.groupby(["protocol", "ablation"], dropna=False):
            ok = bool(sdf["ok"].astype(bool).all())
            missing = "; ".join([x for x in sdf["missing_columns"].astype(str).tolist() if x and x != "nan"])
            rows.append(
                {
                    "protocol": protocol,
                    "ablation": ablation,
                    "check": "schema_required_columns",
                    "status": "pass" if ok else "fail",
                    "value": 1.0 if ok else 0.0,
                    "threshold_or_expectation": "all required protocol-specific columns present",
                    "note": "" if ok else f"missing: {missing}",
                }
            )

    if len(effect_df):
        for _, row in effect_df.iterrows():
            post_window = str(row["post_window"])
            if post_window not in {"post_all", "post_11_30", "post_31_plus"}:
                continue

            expected_direction = str(row["expected_direction"])
            delta = float(row["delta_post_minus_pre_mean"])
            p_value = float(row["directional_sign_flip_p"])
            ci_low = float(row["delta_ci_low"])
            ci_high = float(row["delta_ci_high"])
            sign_ok = expected_sign_ok(delta, expected_direction)

            rows.append(
                {
                    "protocol": row["protocol"],
                    "ablation": row["ablation"],
                    "check": f"p_target_delta_{post_window}",
                    "status": status_for_direction(sign_ok, p_value, ci_low, ci_high, expected_direction),
                    "value": delta,
                    "threshold_or_expectation": (
                        "delta < 0 for shock/negative; delta > 0 for treat/positive; "
                        "pass requires directional p<=0.05 and CI excluding zero"
                    ),
                    "note": (
                        f"ci=[{ci_low:.4f}, {ci_high:.4f}], "
                        f"directional_p={p_value:.6f}, "
                        f"sign_consistency={float(row['directional_sign_consistency']):.4f}"
                    ),
                }
            )

    if len(first_post_df):
        grouped = first_post_df.groupby(["protocol", "ablation", "expected_direction"], dropna=False)
        for (protocol, ablation, expected_direction), sdf in grouped:
            mean_prob = float(sdf["target_prob"].mean())
            prob_ok = expected_prob_ok(mean_prob, str(expected_direction))
            choice_rate = float(sdf["target_choice"].astype(float).mean()) if "target_choice" in sdf.columns else float("nan")
            wins_rate = float(sdf["target_prob_wins"].astype(float).mean()) if "target_prob_wins" in sdf.columns else float("nan")

            rows.append(
                {
                    "protocol": protocol,
                    "ablation": ablation,
                    "check": "first_post_target_probability",
                    "status": "direction_ok" if prob_ok else "warn",
                    "value": mean_prob,
                    "threshold_or_expectation": (
                        "shock/negative expects target_prob < 0.5; "
                        "treat/positive expects target_prob > 0.5"
                    ),
                    "note": f"target_choice_rate={choice_rate:.4f}, target_prob_wins_rate={wins_rate:.4f}",
                }
            )
    else:
        rows.append(
            {
                "protocol": "all",
                "ablation": "all",
                "check": "first_post_target_probability",
                "status": "warn",
                "value": float("nan"),
                "threshold_or_expectation": "first post-event junction rows should be available",
                "note": "No first-post junction rows found.",
            }
        )
    if len(carrier_effect_df):
        for _, row in carrier_effect_df.iterrows():
            post_window = str(row["post_window"])
            if post_window not in {"post_1_3", "post_4_10", "post_11_30", "post_all"}:
                continue

            delta = float(row["delta_post_minus_pre_mean"])
            p_value = float(row["directional_sign_flip_p"])
            ci_low = float(row["delta_ci_low"])
            ci_high = float(row["delta_ci_high"])
            sign_ok = delta > 0.0

            status = status_for_direction(
                sign_ok=sign_ok,
                p_value=p_value,
                ci_low=ci_low,
                ci_high=ci_high,
                expected_direction="positive",
            )

            rows.append(
                {
                    "protocol": row["protocol"],
                    "ablation": row["ablation"],
                    "check": f"carrier_delta_{row['carrier_metric']}_{post_window}",
                    "status": status,
                    "value": delta,
                    "threshold_or_expectation": (
                        "carrier delta must be > 0; pass requires directional "
                        "p<=0.05 and CI excluding zero"
                    ),
                    "note": (
                        f"carrier={row['carrier_metric']}, "
                        f"ci=[{ci_low:.4f}, {ci_high:.4f}], "
                        f"directional_p={p_value:.6f}, "
                        f"sign_consistency={float(row['directional_sign_consistency']):.4f}"
                    ),
                }
            )
    if len(ablation_localization_df):
        for _, row in ablation_localization_df.iterrows():
            metric_family = str(row["metric_family"])

            if metric_family == "availability":
                rows.append(
                    {
                        "protocol": row["protocol"],
                        "ablation": row["ablation"],
                        "check": "ablation_localization_available",
                        "status": "diagnostic",
                        "value": float("nan"),
                        "threshold_or_expectation": (
                            "non-full ablation runs should exist in paper-grade --ablations all mode"
                        ),
                        "note": row["note"],
                    }
                )
                continue

            status = str(row["status"])
            rows.append(
                {
                    "protocol": row["protocol"],
                    "ablation": row["ablation"],
                    "check": (
                        f"ablation_localization_{metric_family}_"
                        f"{row['metric']}_{row['post_window']}_vs_{row['compared_ablation']}"
                    ),
                    "status": status,
                    "value": float(row["directional_margin"]),
                    "threshold_or_expectation": (
                        "full directional magnitude should exceed the compared ablation"
                    ),
                    "note": (
                        f"compared_ablation={row['compared_ablation']}, "
                        f"full_delta={float(row['full_delta']):.4f}, "
                        f"ablation_delta={float(row['ablation_delta']):.4f}, "
                        f"full_mag={float(row['full_directional_magnitude']):.4f}, "
                        f"ablation_mag={float(row['ablation_directional_magnitude']):.4f}"
                    ),
                }
            )
    if len(placebo_df):
        for _, row in placebo_df.iterrows():
            p_value = float(row["directional_placebo_p"])
            status = "pass" if pd.notna(p_value) and p_value <= 0.05 else "diagnostic"

            rows.append(
                {
                    "protocol": row["protocol"],
                    "ablation": row["ablation"],
                    "check": f"placebo_window_null_{row['post_window']}",
                    "status": status,
                    "value": p_value,
                    "threshold_or_expectation": (
                        "directional placebo p<=0.05 indicates real event-aligned "
                        "effect is stronger than random fake event boundaries"
                    ),
                    "note": (
                        f"observed_delta={float(row['observed_delta']):.4f}, "
                        f"null_mean={float(row['null_mean']):.4f}, "
                        f"null_ci=[{float(row['null_ci_low']):.4f}, {float(row['null_ci_high']):.4f}]"
                    ),
                }
            )
    return pd.DataFrame(rows)

def carrier_metrics_for_protocol(protocol: str) -> List[str]:
    if protocol == "shock":
        return ["h_risk", "q_neg"]
    if protocol == "treat":
        return ["h_opp", "q_pos"]
    return []


def build_carrier_effect_stats(
    seed_window_df: pd.DataFrame,
    *,
    n_boot: int,
    n_signflip: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Carrier-level mechanistic guard.

    Behavioral target-choice effects are not sufficient for Stage 3.1B closure.
    The negative protocol must move through h_risk/q_neg; the positive protocol
    must move through h_opp/q_pos.

    Carrier deltas are always expected to be positive:
    post-window carrier mean minus pre-window carrier mean > 0.
    """
    rows: List[Dict[str, Any]] = []

    if seed_window_df.empty:
        return pd.DataFrame()

    grouped = seed_window_df.groupby(
        ["protocol", "ablation", "target_path"],
        dropna=False,
    )

    for (protocol, ablation, target_path), sdf in grouped:
        protocol = str(protocol)
        carriers = carrier_metrics_for_protocol(protocol)
        if not carriers:
            continue

        for metric in carriers:
            mdf = sdf[sdf["metric"].astype(str) == metric].copy()
            if mdf.empty:
                continue

            pivot = mdf.pivot_table(
                index="seed",
                columns="window",
                values="metric_value",
                aggfunc="mean",
            )

            if "pre" not in pivot.columns:
                continue

            for post_window in ["post_1_3", "post_4_10", "post_11_30", "post_31_plus", "post_all"]:
                if post_window not in pivot.columns:
                    continue

                delta = (pivot[post_window] - pivot["pre"]).dropna()
                if len(delta) == 0:
                    continue

                mean, lo, hi = bootstrap_ci(delta.tolist(), n_boot, rng)
                p_value = sign_flip_pvalue(
                    delta.tolist(),
                    expected_direction="positive",
                    n_samples=n_signflip,
                    rng=rng,
                )
                sign_consistency = float((delta > 0).mean())

                rows.append(
                    {
                        "protocol": protocol,
                        "ablation": ablation,
                        "target_path": target_path,
                        "carrier_metric": metric,
                        "post_window": post_window,
                        "n_seeds": int(len(delta)),
                        "pre_mean": float(pivot["pre"].mean()),
                        "post_mean": float(pivot[post_window].mean()),
                        "delta_post_minus_pre_mean": mean,
                        "delta_ci_low": lo,
                        "delta_ci_high": hi,
                        "directional_sign_consistency": sign_consistency,
                        "directional_sign_flip_p": p_value,
                    }
                )

    return pd.DataFrame(rows)

ABLATION_LOCALIZATION_COLUMNS = [
    "protocol",
    "ablation",
    "compared_ablation",
    "target_path",
    "expected_direction",
    "metric_family",
    "metric",
    "post_window",
    "full_delta",
    "ablation_delta",
    "full_directional_magnitude",
    "ablation_directional_magnitude",
    "directional_margin",
    "localized",
    "status",
    "note",
]


def directional_magnitude(delta: float, expected_direction: str) -> float:
    if pd.isna(delta):
        return float("nan")
    if expected_direction == "negative":
        return -float(delta)
    if expected_direction == "positive":
        return float(delta)
    return float("nan")


def localization_row(
    *,
    protocol: str,
    compared_ablation: str,
    target_path: str,
    expected_direction: str,
    metric_family: str,
    metric: str,
    post_window: str,
    full_delta: float,
    ablation_delta: float,
) -> Dict[str, Any]:
    full_mag = directional_magnitude(full_delta, expected_direction)
    ablation_mag = directional_magnitude(ablation_delta, expected_direction)
    margin = full_mag - ablation_mag

    localized = bool(pd.notna(margin) and margin > 0.0)
    status = "pass" if localized else "diagnostic"

    return {
        "protocol": protocol,
        "ablation": "full",
        "compared_ablation": compared_ablation,
        "target_path": target_path,
        "expected_direction": expected_direction,
        "metric_family": metric_family,
        "metric": metric,
        "post_window": post_window,
        "full_delta": float(full_delta),
        "ablation_delta": float(ablation_delta),
        "full_directional_magnitude": float(full_mag),
        "ablation_directional_magnitude": float(ablation_mag),
        "directional_margin": float(margin),
        "localized": localized,
        "status": status,
        "note": "Full directional magnitude minus ablation directional magnitude.",
    }


def build_ablation_localization_stats(
    effect_df: pd.DataFrame,
    carrier_effect_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Diagnostic guard against overfitting to a single full-condition trace.

    This compares the directional magnitude of full against each non-full
    ablation. It is deliberately diagnostic at this stage: after a paper-grade
    --ablations all run, the resulting rows can be promoted into hard acceptance
    criteria if the pattern is stable.

    Behavioral direction:
    - shock: more negative delta is stronger
    - treat: more positive delta is stronger

    Carrier direction:
    - carrier deltas should be positive for both shock and treat
    """
    rows: List[Dict[str, Any]] = []

    behavior_windows = {"post_11_30", "post_all"}
    carrier_windows = {"post_1_3", "post_4_10", "post_11_30", "post_all"}

    if not effect_df.empty:
        bdf = effect_df[effect_df["post_window"].isin(behavior_windows)].copy()

        grouped = bdf.groupby(
            ["protocol", "target_path", "expected_direction", "post_window"],
            dropna=False,
        )

        for (protocol, target_path, expected_direction, post_window), gdf in grouped:
            full = gdf[gdf["ablation"].astype(str) == "full"]
            others = gdf[gdf["ablation"].astype(str) != "full"]

            if full.empty or others.empty:
                continue

            full_delta = float(full.iloc[0]["delta_post_minus_pre_mean"])

            for _, row in others.iterrows():
                rows.append(
                    localization_row(
                        protocol=str(protocol),
                        compared_ablation=str(row["ablation"]),
                        target_path=str(target_path),
                        expected_direction=str(expected_direction),
                        metric_family="behavior",
                        metric="p_target",
                        post_window=str(post_window),
                        full_delta=full_delta,
                        ablation_delta=float(row["delta_post_minus_pre_mean"]),
                    )
                )

    if not carrier_effect_df.empty:
        cdf = carrier_effect_df[carrier_effect_df["post_window"].isin(carrier_windows)].copy()

        grouped = cdf.groupby(
            ["protocol", "target_path", "carrier_metric", "post_window"],
            dropna=False,
        )

        for (protocol, target_path, carrier_metric, post_window), gdf in grouped:
            full = gdf[gdf["ablation"].astype(str) == "full"]
            others = gdf[gdf["ablation"].astype(str) != "full"]

            if full.empty or others.empty:
                continue

            full_delta = float(full.iloc[0]["delta_post_minus_pre_mean"])

            for _, row in others.iterrows():
                rows.append(
                    localization_row(
                        protocol=str(protocol),
                        compared_ablation=str(row["ablation"]),
                        target_path=str(target_path),
                        expected_direction="positive",
                        metric_family="carrier",
                        metric=str(carrier_metric),
                        post_window=str(post_window),
                        full_delta=full_delta,
                        ablation_delta=float(row["delta_post_minus_pre_mean"]),
                    )
                )

    if not rows:
        rows.append(
            {
                "protocol": "all",
                "ablation": "full",
                "compared_ablation": "",
                "target_path": "",
                "expected_direction": "",
                "metric_family": "availability",
                "metric": "non_full_ablation_runs",
                "post_window": "",
                "full_delta": float("nan"),
                "ablation_delta": float("nan"),
                "full_directional_magnitude": float("nan"),
                "ablation_directional_magnitude": float("nan"),
                "directional_margin": float("nan"),
                "localized": False,
                "status": "diagnostic",
                "note": "No non-full ablation rows found. Expected in smoke mode with --ablations full.",
            }
        )

    return pd.DataFrame(rows, columns=ABLATION_LOCALIZATION_COLUMNS)

def compute_target_delta_for_event(
    trials: pd.DataFrame,
    *,
    event_trial: int,
    target_path: str,
    post_window: str,
) -> float:
    target = target_choice_label(target_path)

    trial_values = trials["trial"].astype(int).to_numpy()
    target_hit = (trials["path_choice"].astype(str) == target).astype(float).to_numpy()

    pre_mask = trial_values < int(event_trial)

    if post_window == "post_11_30":
        post_mask = (
            (trial_values >= int(event_trial) + 11)
            & (trial_values <= int(event_trial) + 30)
        )
    elif post_window == "post_all":
        post_mask = trial_values > int(event_trial)
    else:
        raise ValueError(f"Unsupported placebo post window: {post_window}")

    if not pre_mask.any() or not post_mask.any():
        return float("nan")

    return float(target_hit[post_mask].mean() - target_hit[pre_mask].mean())


def valid_placebo_trials(
    trials: pd.DataFrame,
    *,
    true_event_trial: int,
    post_window: str,
    exclusion_radius: int = 5,
    min_pre_trials: int = 10,
) -> List[int]:
    trial_values = sorted(int(x) for x in trials["trial"].dropna().unique())
    if not trial_values:
        return []

    min_trial = min(trial_values)
    max_trial = max(trial_values)

    candidates: List[int] = []

    for event_trial in trial_values:
        if event_trial - min_trial < min_pre_trials:
            continue

        if post_window == "post_11_30":
            if event_trial + 30 > max_trial:
                continue
        elif post_window == "post_all":
            if event_trial + 5 > max_trial:
                continue
        else:
            continue

        if abs(event_trial - int(true_event_trial)) <= exclusion_radius:
            continue

        candidates.append(event_trial)

    # Fallback for very short smoke/debug runs.
    if not candidates:
        for event_trial in trial_values:
            if event_trial - min_trial < max(3, min_pre_trials // 2):
                continue
            if post_window == "post_11_30" and event_trial + 30 > max_trial:
                continue
            if post_window == "post_all" and event_trial + 3 > max_trial:
                continue
            candidates.append(event_trial)

    return candidates

def precompute_seed_placebo_deltas(
    trials: pd.DataFrame,
    *,
    true_event_trial: int,
    target_path: str,
    post_window: str,
) -> np.ndarray:
    """
    Precompute all valid fake-event deltas for one seed.

    This avoids repeated DataFrame filtering inside the placebo Monte Carlo loop.
    """
    candidates = valid_placebo_trials(
        trials,
        true_event_trial=true_event_trial,
        post_window=post_window,
    )

    if not candidates:
        return np.asarray([], dtype=float)

    deltas: List[float] = []
    for fake_event in candidates:
        delta = compute_target_delta_for_event(
            trials,
            event_trial=int(fake_event),
            target_path=target_path,
            post_window=post_window,
        )
        if pd.notna(delta):
            deltas.append(float(delta))

    return np.asarray(deltas, dtype=float)

def directional_placebo_pvalue(
    observed_delta: float,
    null_deltas: Sequence[float],
    expected_direction: str,
) -> float:
    arr = np.asarray([x for x in null_deltas if pd.notna(x)], dtype=float)
    if arr.size == 0 or pd.isna(observed_delta):
        return float("nan")

    if expected_direction == "negative":
        return float((np.sum(arr <= float(observed_delta)) + 1.0) / (arr.size + 1.0))

    if expected_direction == "positive":
        return float((np.sum(arr >= float(observed_delta)) + 1.0) / (arr.size + 1.0))

    return float("nan")

def build_placebo_window_stats(
    specs: Sequence[OneShotRunSpec],
    trials_by_key: Dict[Tuple[str, str], pd.DataFrame],
    effect_df: pd.DataFrame,
    *,
    shock_trial: int,
    n_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Placebo event-time null.

    Optimized implementation:
    - precompute valid fake-event deltas per seed;
    - sample from those precomputed arrays;
    - avoid DataFrame filtering inside the Monte Carlo loop.
    """
    rows: List[Dict[str, Any]] = []
    core_windows = ["post_11_30", "post_all"]

    for spec in specs:
        trials = trials_by_key.get((spec.protocol, spec.ablation))
        if trials is None or trials.empty:
            continue

        seed_groups = {
            int(seed): sdf[["trial", "path_choice"]].copy()
            for seed, sdf in trials.groupby("seed", dropna=False)
        }

        for post_window in core_windows:
            obs_rows = effect_df[
                (effect_df["protocol"].astype(str) == spec.protocol)
                & (effect_df["ablation"].astype(str) == spec.ablation)
                & (effect_df["post_window"].astype(str) == post_window)
            ]

            if obs_rows.empty:
                continue

            observed_delta = float(obs_rows.iloc[0]["delta_post_minus_pre_mean"])

            deltas_by_seed: Dict[int, np.ndarray] = {}
            for seed, sdf in seed_groups.items():
                arr = precompute_seed_placebo_deltas(
                    sdf,
                    true_event_trial=shock_trial,
                    target_path=spec.target_path,
                    post_window=post_window,
                )
                if arr.size:
                    deltas_by_seed[seed] = arr

            active_seeds = sorted(deltas_by_seed.keys())

            print(
                "  placebo null:",
                f"{spec.protocol}/{spec.ablation}/{post_window}",
                f"active_seeds={len(active_seeds)}",
                f"samples={n_samples}",
            )

            if not active_seeds:
                rows.append(
                    {
                        "protocol": spec.protocol,
                        "ablation": spec.ablation,
                        "target_path": spec.target_path,
                        "expected_direction": spec.expected_direction,
                        "post_window": post_window,
                        "n_seeds": 0,
                        "n_placebo_samples": 0,
                        "observed_delta": observed_delta,
                        "null_mean": float("nan"),
                        "null_ci_low": float("nan"),
                        "null_ci_high": float("nan"),
                        "directional_placebo_p": float("nan"),
                        "note": "No valid placebo windows.",
                    }
                )
                continue

            null_matrix = np.empty((int(n_samples), len(active_seeds)), dtype=float)

            for col_idx, seed in enumerate(active_seeds):
                seed_deltas = deltas_by_seed[seed]
                sample_idx = rng.integers(0, seed_deltas.size, size=int(n_samples))
                null_matrix[:, col_idx] = seed_deltas[sample_idx]

            null_values = null_matrix.mean(axis=1)

            null_mean = float(null_values.mean())
            null_lo, null_hi = [
                float(x) for x in np.quantile(null_values, [0.025, 0.975])
            ]
            p_value = directional_placebo_pvalue(
                observed_delta=observed_delta,
                null_deltas=null_values,
                expected_direction=spec.expected_direction,
            )

            rows.append(
                {
                    "protocol": spec.protocol,
                    "ablation": spec.ablation,
                    "target_path": spec.target_path,
                    "expected_direction": spec.expected_direction,
                    "post_window": post_window,
                    "n_seeds": int(len(active_seeds)),
                    "n_placebo_samples": int(null_values.size),
                    "observed_delta": observed_delta,
                    "null_mean": null_mean,
                    "null_ci_low": null_lo,
                    "null_ci_high": null_hi,
                    "directional_placebo_p": p_value,
                    "note": (
                        "Placebo event-time null; conservative because fake "
                        "events are sampled from the same realized sequence."
                    ),
                }
            )

    return pd.DataFrame(rows)

def plot_carrier_effect_by_window(carrier_df: pd.DataFrame, output_path: Path) -> None:
    if carrier_df.empty:
        return

    windows = ["post_1_3", "post_4_10", "post_11_30", "post_all"]
    sdf = carrier_df[carrier_df["post_window"].isin(windows)].copy()
    if sdf.empty:
        return

    sdf["post_window"] = pd.Categorical(sdf["post_window"], categories=windows, ordered=True)

    fig, ax = plt.subplots(figsize=(11, 5))

    for (protocol, ablation, metric), gdf in sdf.groupby(
        ["protocol", "ablation", "carrier_metric"],
        dropna=False,
    ):
        gdf = gdf.sort_values("post_window")
        ax.plot(
            gdf["post_window"].astype(str),
            gdf["delta_post_minus_pre_mean"],
            marker="o",
            label=f"{protocol}:{ablation}:{metric}",
        )

    ax.axhline(0.0, linestyle="--")
    ax.set_title("Stage 3.1B: carrier-specific post-event deltas")
    ax.set_xlabel("Post-event window")
    ax.set_ylabel("Delta carrier value: window - pre")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"✓ Figure saved: {output_path}")

def plot_ablation_localization(ablation_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if ablation_df.empty or set(ablation_df["metric_family"].astype(str)) == {"availability"}:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No non-full ablation rows available.\nExpected in smoke mode with --ablations full.",
            ha="center",
            va="center",
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        print(f"✓ Figure saved: {output_path}")
        return

    sdf = ablation_df[ablation_df["metric_family"].astype(str) != "availability"].copy()
    sdf = sdf.sort_values(
        ["protocol", "metric_family", "metric", "post_window", "compared_ablation"]
    ).reset_index(drop=True)

    labels = [
        f"{r.protocol}:{r.metric}:{r.post_window}:vs_{r.compared_ablation}"
        for r in sdf.itertuples(index=False)
    ]

    x = np.arange(len(sdf))
    y = sdf["directional_margin"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x, y)
    ax.axhline(0.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Full directional magnitude - ablation magnitude")
    ax.set_title("Stage 3.1B: one-shot ablation-localization diagnostics")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"✓ Figure saved: {output_path}")

def plot_placebo_window_null(placebo_df: pd.DataFrame, output_path: Path) -> None:
    if placebo_df.empty:
        return

    sdf = placebo_df.sort_values(["protocol", "ablation", "post_window"]).reset_index(drop=True)
    labels = [
        f"{r.protocol}:{r.ablation}:{r.post_window}"
        for r in sdf.itertuples(index=False)
    ]

    x = np.arange(len(sdf))
    observed = sdf["observed_delta"].to_numpy(dtype=float)
    null_mean = sdf["null_mean"].to_numpy(dtype=float)

    yerr = np.vstack([
        null_mean - sdf["null_ci_low"].to_numpy(dtype=float),
        sdf["null_ci_high"].to_numpy(dtype=float) - null_mean,
    ])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.errorbar(x, null_mean, yerr=yerr, fmt="o", capsize=4, label="placebo null mean ± 95% interval")
    ax.scatter(x, observed, marker="x", s=80, label="observed real-event delta")
    ax.axhline(0.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Delta P(target)")
    ax.set_title("Stage 3.1B: one-shot real-event effect vs placebo event-time null")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"✓ Figure saved: {output_path}")

def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✓ Table saved: {path}")

def plot_zoom_sem(
    seed_trial_series: pd.DataFrame,
    *,
    protocol: str,
    variable: str,
    selected_ablations: Sequence[str],
    shock_trial: int,
    zoom_pre: int,
    zoom_post: int,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    sdf = seed_trial_series[
        (seed_trial_series["protocol"].astype(str) == protocol)
        & (seed_trial_series["ablation"].astype(str).isin(list(selected_ablations)))
        & (seed_trial_series["rel_trial"] >= -zoom_pre)
        & (seed_trial_series["rel_trial"] <= zoom_post)
    ].copy()

    if sdf.empty or variable not in sdf.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for ablation in selected_ablations:
        adf = sdf[sdf["ablation"].astype(str) == ablation].copy()
        if adf.empty:
            continue

        summary = (
            adf.groupby("rel_trial", dropna=False)
            .agg(
                mean=(variable, "mean"),
                sem=(variable, sem),
                n_seeds=("seed", "nunique"),
            )
            .reset_index()
            .sort_values("rel_trial")
        )

        x = summary["rel_trial"].to_numpy(dtype=float)
        y = summary["mean"].to_numpy(dtype=float)
        y_sem = summary["sem"].fillna(0.0).to_numpy(dtype=float)

        line = ax.plot(x, y, marker="o", linewidth=2, label=ablation)[0]
        color = line.get_color()
        ax.fill_between(x, y - y_sem, y + y_sem, alpha=0.18, color=color)

    ax.axvline(0, linestyle="--", linewidth=1.5, label="one-shot event")
    ax.set_title(title)
    ax.set_xlabel("Trial relative to one-shot event")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"✓ Figure saved: {output_path}")

def plot_zoom(
    trial_series: pd.DataFrame,
    *,
    protocol: str,
    variables: Sequence[str],
    shock_trial: int,
    zoom_pre: int,
    zoom_post: int,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    sdf = trial_series[trial_series["protocol"] == protocol].copy()
    if sdf.empty:
        return

    sdf = sdf[(sdf["rel_trial"] >= -zoom_pre) & (sdf["rel_trial"] <= zoom_post)].copy()
    if sdf.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for ablation, adf in sdf.groupby("ablation", dropna=False):
        adf = adf.sort_values("rel_trial")
        for var in variables:
            if var not in adf.columns:
                continue
            ax.plot(
                adf["rel_trial"],
                adf[var],
                marker="o",
                label=f"{ablation}:{var}",
            )

    ax.axvline(0, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Trial relative to one-shot event")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"✓ Figure saved: {output_path}")


def plot_effect_by_ablation(effect_df: pd.DataFrame, output_path: Path) -> None:
    if effect_df.empty:
        return

    sdf = effect_df[effect_df["post_window"] == "post_all"].copy()
    if sdf.empty:
        return

    sdf = sdf.sort_values(["protocol", "ablation"]).reset_index(drop=True)
    labels = [f"{r.protocol}:{r.ablation}" for r in sdf.itertuples(index=False)]
    x = np.arange(len(sdf))
    y = sdf["delta_post_minus_pre_mean"].to_numpy(dtype=float)
    yerr = np.vstack([
        y - sdf["delta_ci_low"].to_numpy(dtype=float),
        sdf["delta_ci_high"].to_numpy(dtype=float) - y,
    ])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=4)
    ax.axhline(0.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Delta P(target): post_all - pre")
    ax.set_title("Stage 3.1B: one-shot target-choice effect by ablation")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"✓ Figure saved: {output_path}")


def plot_carryover_decay(effect_df: pd.DataFrame, output_path: Path) -> None:
    if effect_df.empty:
        return

    windows = ["post_1_3", "post_4_10", "post_11_30", "post_31_plus"]
    sdf = effect_df[effect_df["post_window"].isin(windows)].copy()
    if sdf.empty:
        return

    sdf["post_window"] = pd.Categorical(sdf["post_window"], categories=windows, ordered=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    for (protocol, ablation), gdf in sdf.groupby(["protocol", "ablation"], dropna=False):
        gdf = gdf.sort_values("post_window")
        ax.plot(
            gdf["post_window"].astype(str),
            gdf["delta_post_minus_pre_mean"],
            marker="o",
            label=f"{protocol}:{ablation}",
        )

    ax.axhline(0.0, linestyle="--")
    ax.set_title("Stage 3.1B: one-shot carryover decay by post-event window")
    ax.set_xlabel("Post-event window")
    ax.set_ylabel("Delta P(target): window - pre")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"✓ Figure saved: {output_path}")


def write_report(
    *,
    output_dir: Path,
    manifest_path: Path,
    schema_df: pd.DataFrame,
    effect_df: pd.DataFrame,
    first_post_df: pd.DataFrame,
    acceptance_df: pd.DataFrame,
) -> None:
    path = output_dir / "Stage3_1B_OneShot_Publication_Report.md"

    lines: List[str] = []
    lines.append("# Stage 3.1B One-Shot Publication Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "This report summarizes protocol-aware one-shot analysis for Stage 3.1B closure. "
        "Shock is treated as the negative branch (`h_risk/q_neg`), while treat is treated "
        "as the positive branch (`h_opp/q_pos`)."
    )
    lines.append("")
    lines.append(f"- Source manifest: `{manifest_path}`")
    lines.append("")
    lines.append("## Schema validation")
    lines.append("")
    if len(schema_df):
        lines.append("```")
        lines.append(schema_df.to_string(index=False))
        lines.append("```")
    else:
        lines.append("No schema rows.")
    lines.append("")

    lines.append("## Compact acceptance summary")
    lines.append("")
    if len(acceptance_df):
        compact = acceptance_df[
            [
                "protocol",
                "ablation",
                "check",
                "status",
                "value",
                "threshold_or_expectation",
                "note",
            ]
        ].copy()
        lines.append("```")
        lines.append(compact.to_string(index=False))
        lines.append("```")
    else:
        lines.append("No acceptance summary rows.")
    lines.append("")

    lines.append("## Directional effect statistics")
    lines.append("")
    if len(effect_df):
        compact = effect_df[
            [
                "protocol",
                "ablation",
                "post_window",
                "n_seeds",
                "delta_post_minus_pre_mean",
                "delta_ci_low",
                "delta_ci_high",
                "directional_sign_consistency",
                "directional_sign_flip_p",
            ]
        ].copy()
        lines.append("```")
        lines.append(compact.to_string(index=False))
        lines.append("```")
    else:
        lines.append("No effect statistics.")
    lines.append("")

    lines.append("## First post-event junction")
    lines.append("")
    if len(first_post_df):
        compact = first_post_df[
            [
                "protocol",
                "ablation",
                "seed",
                "target_path",
                "first_post_trial",
                "target_prob",
                "target_prob_wins",
                "target_choice",
                "target_local_bonus",
                "mode",
                "gate_trigger",
            ]
        ].copy()
        lines.append("```")
        lines.append(compact.to_string(index=False))
        lines.append("```")
    else:
        lines.append("No first post-event junction rows.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ Report saved: {path}")


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shock_trial = int(manifest.get("shock_trial", 30))
    rng = np.random.default_rng(args.bootstrap_seed)

    specs = build_run_specs(manifest)

    # Early summary to confirm manifest loading before proceeding with heavy computations.
    print(f"Loaded suite manifest: {manifest_path}")
    print(f"One-shot specs: {len(specs)}")

    schema_records: List[Dict[str, Any]] = []
    trial_series_frames: List[pd.DataFrame] = []
    seed_trial_series_frames: List[pd.DataFrame] = []
    seed_window_frames: List[pd.DataFrame] = []
    first_post_frames: List[pd.DataFrame] = []
    trials_by_key: Dict[Tuple[str, str], pd.DataFrame] = {}

    for spec in specs:
        print(f"Loading one-shot run: {spec.protocol}/{spec.ablation} from {spec.run_dir}")
        trials = load_trials(spec.run_dir)
        steps = load_steps(spec.run_dir)
        debug_rows = load_debug_rows(spec.run_dir)
        trials_by_key[(spec.protocol, spec.ablation)] = trials.copy()

        schema_records.extend(validate_schema(spec, trials, steps))

        seed_trial_series_frames.append(
            collect_trial_seed_series(spec, trials, steps, shock_trial=shock_trial)
        )
        seed_window_frames.append(
            collect_window_seed_metrics(spec, trials, steps, shock_trial=shock_trial)
        )
        first_post_frames.append(
            collect_first_post_junction(spec, debug_rows, shock_trial=shock_trial)
        )

    schema_df = pd.DataFrame(schema_records)
    trial_series_df = pd.concat(trial_series_frames, ignore_index=True) if trial_series_frames else pd.DataFrame()
    seed_trial_series_df = pd.concat(seed_trial_series_frames, ignore_index=True) if seed_trial_series_frames else pd.DataFrame()
    seed_window_df = pd.concat(seed_window_frames, ignore_index=True) if seed_window_frames else pd.DataFrame()
    first_post_df = pd.concat(first_post_frames, ignore_index=True) if first_post_frames else pd.DataFrame()

    print("Computing behavioral effect statistics...")

    window_summary_df = summarize_window_metrics(
        seed_window_df,
        n_boot=args.bootstrap_samples,
        rng=rng,
    )
    effect_df = build_effect_stats(
        seed_window_df,
        n_boot=args.bootstrap_samples,
        n_signflip=args.signflip_samples,
        rng=rng,
    )

    print("Computing carrier effect statistics...")

    carrier_effect_df = build_carrier_effect_stats(
        seed_window_df,
        n_boot=args.bootstrap_samples,
        n_signflip=args.signflip_samples,
        rng=rng,
    )

    print("Computing ablation-localization diagnostics...")

    ablation_localization_df = build_ablation_localization_stats(
        effect_df=effect_df,
        carrier_effect_df=carrier_effect_df,
    )

    print("Computing placebo event-window nulls...")

    placebo_df = build_placebo_window_stats(
        specs=specs,
        trials_by_key=trials_by_key,
        effect_df=effect_df,
        shock_trial=shock_trial,
        n_samples=args.placebo_samples,
        rng=rng,
    )

    acceptance_df = build_acceptance_summary(
        schema_df=schema_df,
        effect_df=effect_df,
        first_post_df=first_post_df,
        placebo_df=placebo_df,
        carrier_effect_df=carrier_effect_df,
        ablation_localization_df=ablation_localization_df,
    )

    save_table(schema_df, output_dir / "Table_3_1B_one_shot_schema_validation.csv")
    save_table(trial_series_df, output_dir / "Table_3_1B_one_shot_trial_series.csv")
    save_table(seed_trial_series_df, output_dir / "Table_3_1B_one_shot_seed_trial_series.csv")
    save_table(seed_window_df, output_dir / "Table_3_1B_one_shot_window_seed_metrics.csv")
    save_table(window_summary_df, output_dir / "Table_3_1B_one_shot_window_summary.csv")
    save_table(effect_df, output_dir / "Table_3_1B_one_shot_effect_stats.csv")
    save_table(first_post_df, output_dir / "Table_3_1B_one_shot_first_post_junction.csv")
    save_table(carrier_effect_df, output_dir / "Table_3_1B_one_shot_carrier_effect_stats.csv")
    save_table(ablation_localization_df, output_dir / "Table_3_1B_one_shot_ablation_localization.csv")
    save_table(placebo_df, output_dir / "Table_3_1B_one_shot_placebo_window_stats.csv")
    save_table(acceptance_df, output_dir / "Table_3_1B_one_shot_acceptance_summary.csv")

    acceptance_json_path = output_dir / "one_shot_acceptance_summary.json"
    acceptance_json_path.write_text(
        json.dumps(acceptance_df.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"✓ Stats saved: {acceptance_json_path}")

    plot_zoom_sem(
        seed_trial_series_df,
        protocol="shock",
        variable="p_target",
        selected_ablations=["full", "nox", "one_shot_off"],
        shock_trial=shock_trial,
        zoom_pre=args.zoom_pre,
        zoom_post=args.zoom_post,
        title="Stage 3.1B: shock target choice around one-shot event",
        ylabel="P(target path) ± SEM",
        output_path=output_dir / "Figure_3_1B_Shock_Target_Choice_SEM_Zoom.png",
    )

    plot_zoom_sem(
        seed_trial_series_df,
        protocol="treat",
        variable="p_target",
        selected_ablations=["full", "novp", "one_shot_off"],
        shock_trial=shock_trial,
        zoom_pre=args.zoom_pre,
        zoom_post=args.zoom_post,
        title="Stage 3.1B: treat target choice around one-shot event",
        ylabel="P(target path) ± SEM",
        output_path=output_dir / "Figure_3_1B_Treat_Target_Choice_SEM_Zoom.png",
    )

    plot_zoom_sem(
        seed_trial_series_df,
        protocol="shock",
        variable="q_neg",
        selected_ablations=["full", "nox", "one_shot_off"],
        shock_trial=shock_trial,
        zoom_pre=args.zoom_pre,
        zoom_post=args.zoom_post,
        title="Stage 3.1B: shock q_neg carrier around one-shot event",
        ylabel="q_neg ± SEM",
        output_path=output_dir / "Figure_3_1B_Shock_QNeg_SEM_Zoom.png",
    )

    plot_zoom_sem(
        seed_trial_series_df,
        protocol="shock",
        variable="h_risk",
        selected_ablations=["full", "nox", "one_shot_off"],
        shock_trial=shock_trial,
        zoom_pre=args.zoom_pre,
        zoom_post=args.zoom_post,
        title="Stage 3.1B: shock h_risk carrier around one-shot event",
        ylabel="h_risk ± SEM",
        output_path=output_dir / "Figure_3_1B_Shock_HRisk_SEM_Zoom.png",
    )

    plot_zoom_sem(
        seed_trial_series_df,
        protocol="treat",
        variable="q_pos",
        selected_ablations=["full", "novp", "one_shot_off"],
        shock_trial=shock_trial,
        zoom_pre=args.zoom_pre,
        zoom_post=args.zoom_post,
        title="Stage 3.1B: treat q_pos carrier around one-shot event",
        ylabel="q_pos ± SEM",
        output_path=output_dir / "Figure_3_1B_Treat_QPos_SEM_Zoom.png",
    )

    plot_zoom_sem(
        seed_trial_series_df,
        protocol="treat",
        variable="h_opp",
        selected_ablations=["full", "novp", "one_shot_off"],
        shock_trial=shock_trial,
        zoom_pre=args.zoom_pre,
        zoom_post=args.zoom_post,
        title="Stage 3.1B: treat h_opp carrier around one-shot event",
        ylabel="h_opp ± SEM",
        output_path=output_dir / "Figure_3_1B_Treat_HOpp_SEM_Zoom.png",
    )
    plot_zoom(
        trial_series_df,
        protocol="shock",
        variables=["mean_h_risk", "mean_q_neg"],
        shock_trial=shock_trial,
        zoom_pre=args.zoom_pre,
        zoom_post=args.zoom_post,
        title="Stage 3.1B: shock negative carrier around one-shot event",
        ylabel="Mean negative carrier",
        output_path=output_dir / "Figure_3_1B_Shock_Risk_Carrier_Around_Event_Zoom.png",
    )
    plot_zoom(
        trial_series_df,
        protocol="treat",
        variables=["mean_h_opp", "mean_q_pos"],
        shock_trial=shock_trial,
        zoom_pre=args.zoom_pre,
        zoom_post=args.zoom_post,
        title="Stage 3.1B: treat positive carrier around one-shot event",
        ylabel="Mean positive carrier",
        output_path=output_dir / "Figure_3_1B_Treat_Opportunity_Carrier_Around_Event_Zoom.png",
    )
    plot_zoom(
        trial_series_df,
        protocol="shock",
        variables=["p_target"],
        shock_trial=shock_trial,
        zoom_pre=args.zoom_pre,
        zoom_post=args.zoom_post,
        title="Stage 3.1B: shock target choice around one-shot event",
        ylabel="P(target path)",
        output_path=output_dir / "Figure_3_1B_Shock_Target_Choice_Around_Event_Zoom.png",
    )
    plot_zoom(
        trial_series_df,
        protocol="treat",
        variables=["p_target"],
        shock_trial=shock_trial,
        zoom_pre=args.zoom_pre,
        zoom_post=args.zoom_post,
        title="Stage 3.1B: treat target choice around one-shot event",
        ylabel="P(target path)",
        output_path=output_dir / "Figure_3_1B_Treat_Target_Choice_Around_Event_Zoom.png",
    )
    plot_effect_by_ablation(
        effect_df,
        output_path=output_dir / "Figure_3_1B_OneShot_Effect_By_Ablation.png",
    )
    plot_carryover_decay(
        effect_df,
        output_path=output_dir / "Figure_3_1B_OneShot_Carryover_Decay_By_Window.png",
    )
    plot_carrier_effect_by_window(
        carrier_effect_df,
        output_path=output_dir / "Figure_3_1B_OneShot_Carrier_Effect_By_Window.png",
    )
    plot_ablation_localization(
        ablation_localization_df,
        output_path=output_dir / "Figure_3_1B_OneShot_Ablation_Localization.png",
    )    
    plot_placebo_window_null(
        placebo_df,
        output_path=output_dir / "Figure_3_1B_OneShot_Placebo_Window_Null.png",
    )

    meta = {
        "manifest": str(manifest_path),
        "shock_trial": shock_trial,
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "signflip_samples": int(args.signflip_samples),
        "n_specs": int(len(specs)),
        "n_trial_series_rows": int(len(trial_series_df)),
        "n_seed_trial_series_rows": int(len(seed_trial_series_df)),
        "n_seed_window_rows": int(len(seed_window_df)),
        "n_first_post_rows": int(len(first_post_df)),
        "n_carrier_effect_rows": int(len(carrier_effect_df)),
        "n_ablation_localization_rows": int(len(ablation_localization_df)),
        "n_placebo_rows": int(len(placebo_df)),
        "placebo_samples": int(args.placebo_samples),
    }
    meta_path = output_dir / "one_shot_publication_analysis_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓ Analysis metadata saved: {meta_path}")

    write_report(
        output_dir=output_dir,
        manifest_path=manifest_path,
        schema_df=schema_df,
        effect_df=effect_df,
        first_post_df=first_post_df,
        acceptance_df=acceptance_df,
    )


if __name__ == "__main__":
    main()