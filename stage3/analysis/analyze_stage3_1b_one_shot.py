"""
Stage 3.1B: One-Shot Graphical Analysis with fired vs non-fired subgroup analysis.

Строит графический и табличный анализ one-shot protocol из уже сохранённых логов.

Ожидаемые файлы в run directory:
- one_shot_full_run_summary.json
- one_shot_full_all_trials.csv
- one_shot_full_all_steps.csv
- optionally: one_shot_full_seed_summary.csv

Поддерживает два режима входа:
1. --input-dir указывает на конкретную папку one_shot_* run
2. --input-dir указывает на logs/stage3/stage3_1b,
   тогда скрипт сам находит последний one_shot_* run

Default paths:
- input-dir  = logs/stage3/stage3_1b
- output-dir = logs/figures/stage3/stage3_1b_one_shot_analysis

Usage:
    python -m stage3.analysis.analyze_stage3_1b_one_shot

    python -m stage3.analysis.analyze_stage3_1b_one_shot ^
        --input-dir logs/stage3/stage3_1b\\one_shot_full_20260408_194951 ^
        --output-dir logs/figures/stage3/stage3_1b_one_shot_analysis
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Stage 3.1B one-shot run")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="logs/stage3/stage3_1b",
        help="One-shot run directory or base logs directory containing one_shot_* runs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/figures/stage3/stage3_1b_one_shot_analysis",
        help="Directory for figures and summary tables"
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        help="Ablation suffix (default: full)"
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=5,
        help="Rolling window for P(open) by trial smoothing"
    )
    return parser.parse_args()


def is_one_shot_run_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("one_shot_")


def find_latest_one_shot_run(input_dir: Path) -> Path:
    if input_dir.is_dir():
        if any(input_dir.glob("*_run_summary.json")) and any(input_dir.glob("*_all_trials.csv")):
            return input_dir

        candidates = [p for p in input_dir.iterdir() if is_one_shot_run_dir(p)]
        if candidates:
            candidates = sorted(candidates, key=lambda p: p.name)
            return candidates[-1]

    raise FileNotFoundError(f"Could not locate one-shot run directory in {input_dir}")


def find_single_file(run_dir: Path, pattern: str) -> Path:
    matches = sorted(run_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern} in {run_dir}")
    return matches[-1]


def try_find_optional_file(run_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(run_dir.glob(pattern))
    if not matches:
        return None
    return matches[-1]


def load_run_data(run_dir: Path, ablation: str):
    run_summary_path = find_single_file(run_dir, f"*_{ablation}_run_summary.json")
    all_trials_path = find_single_file(run_dir, f"*_{ablation}_all_trials.csv")
    all_steps_path = find_single_file(run_dir, f"*_{ablation}_all_steps.csv")
    seed_summary_path = try_find_optional_file(run_dir, f"*_{ablation}_seed_summary.csv")

    with open(run_summary_path, "r", encoding="utf-8") as f:
        run_summary = json.load(f)

    all_trials = pd.read_csv(all_trials_path)
    all_steps = pd.read_csv(all_steps_path)
    seed_summary = pd.read_csv(seed_summary_path) if seed_summary_path is not None else None

    return run_summary, all_trials, all_steps, seed_summary


def infer_shock_trial(all_trials: pd.DataFrame) -> int:
    if "one_shot_trial" in all_trials.columns:
        vals = [int(v) for v in all_trials["one_shot_trial"].dropna().unique().tolist() if int(v) >= 0]
        if vals:
            return vals[0]
    return 30


def split_pre_shock_post(all_trials: pd.DataFrame, shock_trial: int):
    pre_df = all_trials[all_trials["trial"] < shock_trial].copy()
    shock_df = all_trials[all_trials["trial"] == shock_trial].copy()
    post_df = all_trials[all_trials["trial"] > shock_trial].copy()
    return pre_df, shock_df, post_df


def block_summary(df: pd.DataFrame, label: str) -> dict:
    if len(df) == 0:
        return {
            "block": label,
            "n_trials": 0,
            "p_open": np.nan,
            "p_covered": np.nan,
            "mean_commit_latency": np.nan,
            "mean_junction_pause_duration": np.nan,
            "mean_reorientation_count": np.nan,
            "mean_junction_deliberation_proxy": np.nan,
            "mode_at_junction_top": "",
            "mode_at_junction_top_p": np.nan,
        }

    mode_dist = (
        df["mode_at_junction"].value_counts(normalize=True).to_dict()
        if "mode_at_junction" in df.columns else {}
    )
    mode_top = max(mode_dist, key=mode_dist.get) if mode_dist else ""
    mode_top_p = float(mode_dist.get(mode_top, 0.0)) if mode_top else np.nan

    return {
        "block": label,
        "n_trials": int(len(df)),
        "p_open": float((df["path_choice"] == "open").mean()),
        "p_covered": float((df["path_choice"] == "covered").mean()),
        "mean_commit_latency": float(df["commit_latency"].mean()) if "commit_latency" in df.columns else np.nan,
        "mean_junction_pause_duration": float(df["junction_pause_duration"].mean()) if "junction_pause_duration" in df.columns else np.nan,
        "mean_reorientation_count": float(df["reorientation_count"].mean()) if "reorientation_count" in df.columns else np.nan,
        "mean_junction_deliberation_proxy": float(df["junction_deliberation_proxy"].mean()) if "junction_deliberation_proxy" in df.columns else np.nan,
        "mode_at_junction_top": mode_top,
        "mode_at_junction_top_p": mode_top_p,
    }


def compute_trial_level_choice(all_trials: pd.DataFrame) -> pd.DataFrame:
    trial_df = (
        all_trials
        .groupby("trial", dropna=False)
        .agg(
            n_trials=("trial", "size"),
            p_open=("path_choice", lambda s: (s == "open").mean()),
            p_covered=("path_choice", lambda s: (s == "covered").mean()),
            mean_commit_latency=("commit_latency", "mean"),
            mean_junction_pause_duration=("junction_pause_duration", "mean"),
            mean_reorientation_count=("reorientation_count", "mean"),
            mean_junction_deliberation_proxy=("junction_deliberation_proxy", "mean"),
        )
        .reset_index()
        .sort_values("trial")
    )
    return trial_df


def compute_hrisk_by_trial(all_steps: pd.DataFrame) -> pd.DataFrame:
    required = {"trial", "h_risk"}
    if not required.issubset(set(all_steps.columns)):
        raise ValueError(f"Missing required step-level columns: {sorted(required)}")

    agg_dict = {
        "mean_h_risk": ("h_risk", "mean"),
    }
    if "h_opp" in all_steps.columns:
        agg_dict["mean_h_opp"] = ("h_opp", "mean")
    if "h_time" in all_steps.columns:
        agg_dict["mean_h_time"] = ("h_time", "mean")

    hrisk_df = (
        all_steps
        .groupby("trial", dropna=False)
        .agg(**agg_dict)
        .reset_index()
        .sort_values("trial")
    )

    if "mean_h_opp" not in hrisk_df.columns:
        hrisk_df["mean_h_opp"] = np.nan
    if "mean_h_time" not in hrisk_df.columns:
        hrisk_df["mean_h_time"] = np.nan

    return hrisk_df


def compute_mode_pre_post(pre_df: pd.DataFrame, post_df: pd.DataFrame) -> pd.DataFrame:
    modes = sorted(
        set(pre_df.get("mode_at_junction", pd.Series(dtype=str)).dropna().tolist()) |
        set(post_df.get("mode_at_junction", pd.Series(dtype=str)).dropna().tolist())
    )

    rows = []
    for mode in modes:
        pre_p = float((pre_df["mode_at_junction"] == mode).mean()) if len(pre_df) else np.nan
        post_p = float((post_df["mode_at_junction"] == mode).mean()) if len(post_df) else np.nan
        rows.append({
            "mode_at_junction": mode,
            "pre_p": pre_p,
            "post_p": post_p,
            "delta_post_minus_pre": post_p - pre_p,
        })
    return pd.DataFrame(rows)


def compute_latency_pre_post(pre_df: pd.DataFrame, post_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    metrics = [
        "commit_latency",
        "junction_pause_duration",
        "reorientation_count",
        "junction_deliberation_proxy",
    ]

    for metric in metrics:
        if metric not in pre_df.columns or metric not in post_df.columns:
            continue
        pre_val = float(pre_df[metric].mean()) if len(pre_df) else np.nan
        post_val = float(post_df[metric].mean()) if len(post_df) else np.nan
        rows.append({
            "metric": f"mean_{metric}",
            "pre": pre_val,
            "post": post_val,
            "delta_post_minus_pre": post_val - pre_val,
        })

    return pd.DataFrame(rows)


def compute_seed_effect(all_trials: pd.DataFrame, shock_trial: int) -> pd.DataFrame:
    rows = []

    for seed, seed_df in all_trials.groupby("seed", dropna=False):
        pre = seed_df[seed_df["trial"] < shock_trial]
        shock = seed_df[seed_df["trial"] == shock_trial]
        post = seed_df[seed_df["trial"] > shock_trial]

        shock_choice = shock["path_choice"].iloc[0] if len(shock) else ""
        one_shot_fired = bool(shock["one_shot_fired"].astype(bool).any()) if "one_shot_fired" in shock.columns else False

        pre_p = float((pre["path_choice"] == "open").mean()) if len(pre) else np.nan
        post_p = float((post["path_choice"] == "open").mean()) if len(post) else np.nan

        rows.append({
            "seed": seed,
            "shock_choice": shock_choice,
            "one_shot_fired": one_shot_fired,
            "subgroup": "fired" if one_shot_fired else "non_fired",
            "pre_p_open": pre_p,
            "post_p_open": post_p,
            "delta_post_minus_pre": post_p - pre_p if pd.notna(pre_p) and pd.notna(post_p) else np.nan,
        })

    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def subset_by_seed_effect(df: pd.DataFrame, seed_effect_df: pd.DataFrame, subgroup: str) -> pd.DataFrame:
    seeds = seed_effect_df.loc[seed_effect_df["subgroup"] == subgroup, "seed"].tolist()
    return df[df["seed"].isin(seeds)].copy()


def compute_group_package(
    all_trials: pd.DataFrame,
    all_steps: pd.DataFrame,
    shock_trial: int,
    label: str
) -> Dict[str, pd.DataFrame]:
    pre_df, shock_df, post_df = split_pre_shock_post(all_trials, shock_trial)

    pre_post_df = pd.DataFrame([
        block_summary(pre_df, "pre"),
        block_summary(shock_df, "shock"),
        block_summary(post_df, "post"),
    ])

    trial_df = compute_trial_level_choice(all_trials) if len(all_trials) else pd.DataFrame()
    hrisk_df = compute_hrisk_by_trial(all_steps) if len(all_steps) else pd.DataFrame()
    mode_df = compute_mode_pre_post(pre_df, post_df) if len(all_trials) else pd.DataFrame()
    latency_df = compute_latency_pre_post(pre_df, post_df) if len(all_trials) else pd.DataFrame()
    checks = compute_acceptance_checks(pre_df, shock_df, post_df, hrisk_df, shock_trial) if len(all_trials) else {}

    return {
        "label": label,
        "pre_df": pre_df,
        "shock_df": shock_df,
        "post_df": post_df,
        "pre_post_df": pre_post_df,
        "trial_df": trial_df,
        "hrisk_df": hrisk_df,
        "mode_df": mode_df,
        "latency_df": latency_df,
        "checks": checks,
    }


def compute_acceptance_checks(pre_df: pd.DataFrame, shock_df: pd.DataFrame, post_df: pd.DataFrame, hrisk_df: pd.DataFrame, shock_trial: int) -> dict:
    pre_p_open = float((pre_df["path_choice"] == "open").mean()) if len(pre_df) else np.nan
    post_p_open = float((post_df["path_choice"] == "open").mean()) if len(post_df) else np.nan

    one_shot_logged = False
    if len(shock_df) and "one_shot_fired" in shock_df.columns:
        one_shot_logged = bool(shock_df["one_shot_fired"].astype(bool).any())

    pre_hrisk = hrisk_df[hrisk_df["trial"] < shock_trial]["mean_h_risk"].mean() if len(hrisk_df) else np.nan
    post_hrisk = hrisk_df[hrisk_df["trial"] > shock_trial]["mean_h_risk"].mean() if len(hrisk_df) else np.nan

    return {
        "shock_trial": shock_trial,
        "pre_p_open": pre_p_open,
        "post_p_open": post_p_open,
        "delta_post_minus_pre": post_p_open - pre_p_open if pd.notna(pre_p_open) and pd.notna(post_p_open) else np.nan,
        "post_shock_p_open_lt_pre": bool(post_p_open < pre_p_open) if pd.notna(pre_p_open) and pd.notna(post_p_open) else None,
        "one_shot_logged": bool(one_shot_logged),
        "pre_h_risk": pre_hrisk,
        "post_h_risk": post_hrisk,
        "hrisk_post_gt_pre": bool(post_hrisk > pre_hrisk) if pd.notna(pre_hrisk) and pd.notna(post_hrisk) else None,
    }


def save_table(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot_pre_post(pre_post_df: pd.DataFrame, title: str, output_path: Path):
    x = np.arange(len(pre_post_df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, pre_post_df["p_open"], width=width, label="p_open")
    ax.bar(x + width / 2, pre_post_df["p_covered"], width=width, label="p_covered")

    ax.set_xticks(x)
    ax.set_xticklabels(pre_post_df["block"].tolist())
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title(title)

    for i, row in enumerate(pre_post_df.itertuples(index=False)):
        if pd.notna(row.p_open):
            ax.text(i - width / 2, row.p_open + 0.02, f"{row.p_open:.3f}", ha="center", va="bottom")
        if pd.notna(row.p_covered):
            ax.text(i + width / 2, row.p_covered + 0.02, f"{row.p_covered:.3f}", ha="center", va="bottom")

    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_pre_post_by_subgroup(group_packages: Dict[str, Dict], output_path: Path):
    subgroups = ["fired", "non_fired"]
    blocks = ["pre", "shock", "post"]
    x = np.arange(len(blocks))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))

    offsets = [-0.5 * width, 0.5 * width]
    for subgroup, offset in zip(subgroups, offsets):
        df = group_packages[subgroup]["pre_post_df"]
        p_open = [df.loc[df["block"] == block, "p_open"].iloc[0] if len(df.loc[df["block"] == block]) else np.nan for block in blocks]
        ax.bar(x + offset, p_open, width=width, label=f"{subgroup}_p_open")

    ax.set_xticks(x)
    ax.set_xticklabels(blocks)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(open)")
    ax.set_title("Stage 3.1B: One-shot pre / shock / post by subgroup")
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_hrisk_by_trial(hrisk_df: pd.DataFrame, shock_trial: int, title: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(hrisk_df["trial"], hrisk_df["mean_h_risk"], marker="o")
    ax.axvline(shock_trial, linestyle="--")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Mean h_risk")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_hrisk_by_subgroup(group_packages: Dict[str, Dict], shock_trial: int, output_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))

    for subgroup in ["fired", "non_fired"]:
        hrisk_df = group_packages[subgroup]["hrisk_df"]
        if len(hrisk_df):
            ax.plot(hrisk_df["trial"], hrisk_df["mean_h_risk"], marker="o", label=subgroup)

    ax.axvline(shock_trial, linestyle="--")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Mean h_risk")
    ax.set_title("Stage 3.1B: h_risk around shock by subgroup")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_popen_by_trial(trial_df: pd.DataFrame, shock_trial: int, rolling_window: int, title: str, output_path: Path):
    plot_df = trial_df.copy()
    plot_df["p_open_roll"] = plot_df["p_open"].rolling(window=rolling_window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(plot_df["trial"], plot_df["p_open"], marker="o", label="p_open")
    ax.plot(plot_df["trial"], plot_df["p_open_roll"], label=f"rolling_mean_{rolling_window}")
    ax.axvline(shock_trial, linestyle="--")
    ax.set_xlabel("Trial")
    ax.set_ylabel("P(open)")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_popen_by_subgroup(group_packages: Dict[str, Dict], shock_trial: int, rolling_window: int, output_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))

    for subgroup in ["fired", "non_fired"]:
        trial_df = group_packages[subgroup]["trial_df"]
        if len(trial_df):
            plot_df = trial_df.copy()
            plot_df["p_open_roll"] = plot_df["p_open"].rolling(window=rolling_window, min_periods=1).mean()
            ax.plot(plot_df["trial"], plot_df["p_open_roll"], label=f"{subgroup}_rolling_mean_{rolling_window}")

    ax.axvline(shock_trial, linestyle="--")
    ax.set_xlabel("Trial")
    ax.set_ylabel("P(open)")
    ax.set_ylim(0, 1)
    ax.set_title("Stage 3.1B: P(open) by trial, fired vs non-fired")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_mode_pre_post(mode_df: pd.DataFrame, title: str, output_path: Path):
    if len(mode_df) == 0:
        return

    x = np.arange(len(mode_df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, mode_df["pre_p"], width=width, label="pre")
    ax.bar(x + width / 2, mode_df["post_p"], width=width, label="post")

    ax.set_xticks(x)
    ax.set_xticklabels(mode_df["mode_at_junction"].tolist(), rotation=20)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title(title)

    for i, row in enumerate(mode_df.itertuples(index=False)):
        if pd.notna(row.pre_p):
            ax.text(i - width / 2, row.pre_p + 0.02, f"{row.pre_p:.3f}", ha="center", va="bottom")
        if pd.notna(row.post_p):
            ax.text(i + width / 2, row.post_p + 0.02, f"{row.post_p:.3f}", ha="center", va="bottom")

    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_latency_pre_post(latency_df: pd.DataFrame, title: str, output_path: Path):
    if len(latency_df) == 0:
        return

    x = np.arange(len(latency_df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, latency_df["pre"], width=width, label="pre")
    ax.bar(x + width / 2, latency_df["post"], width=width, label="post")

    ax.set_xticks(x)
    ax.set_xticklabels(latency_df["metric"].tolist(), rotation=25, ha="right")
    ax.set_ylabel("Value")
    ax.set_title(title)

    for i, row in enumerate(latency_df.itertuples(index=False)):
        if pd.notna(row.pre):
            ax.text(i - width / 2, row.pre + 0.02, f"{row.pre:.3f}", ha="center", va="bottom")
        if pd.notna(row.post):
            ax.text(i + width / 2, row.post + 0.02, f"{row.post:.3f}", ha="center", va="bottom")

    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_seed_delta(seed_effect_df: pd.DataFrame, output_path: Path):
    plot_df = seed_effect_df.copy().sort_values(["subgroup", "delta_post_minus_pre", "seed"]).reset_index(drop=True)
    colors = plot_df["subgroup"].map({"fired": "tab:red", "non_fired": "tab:blue"}).tolist()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(plot_df)), plot_df["delta_post_minus_pre"], color=colors)
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Seeds (sorted)")
    ax.set_ylabel("delta_post_minus_pre")
    ax.set_title("Stage 3.1B: Seed-level post minus pre by shock status")

    fired_proxy = plt.Rectangle((0, 0), 1, 1, color="tab:red")
    nonfired_proxy = plt.Rectangle((0, 0), 1, 1, color="tab:blue")
    ax.legend([fired_proxy, nonfired_proxy], ["fired", "non_fired"])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    output_dir: Path,
    run_summary: dict,
    overall_package: Dict,
    group_packages: Dict[str, Dict],
    seed_effect_df: pd.DataFrame,
    overall_checks: dict,
):
    out_path = output_dir / "Stage3_1B_OneShot_Report.md"

    lines = []
    lines.append("# Stage 3.1B One-Shot Report")
    lines.append("")
    lines.append("## Run summary")
    lines.append("")
    lines.append("```")
    lines.append(json.dumps(run_summary, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Overall: pre / shock / post")
    lines.append("")
    lines.append("```")
    lines.append(overall_package["pre_post_df"].to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Overall: mode at junction pre vs post")
    lines.append("")
    lines.append("```")
    lines.append(overall_package["mode_df"].to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Overall: latency / pause / reorientation / proxy")
    lines.append("")
    lines.append("```")
    lines.append(overall_package["latency_df"].to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Seed-level effect")
    lines.append("")
    lines.append("```")
    lines.append(seed_effect_df.to_string(index=False))
    lines.append("```")
    lines.append("")

    for subgroup in ["fired", "non_fired"]:
        lines.append(f"## Subgroup: {subgroup}")
        lines.append("")
        lines.append("```")
        lines.append(group_packages[subgroup]["pre_post_df"].to_string(index=False))
        lines.append("```")
        lines.append("")
        lines.append("```")
        lines.append(group_packages[subgroup]["mode_df"].to_string(index=False))
        lines.append("```")
        lines.append("")
        lines.append("```")
        lines.append(group_packages[subgroup]["latency_df"].to_string(index=False))
        lines.append("```")
        lines.append("")

    lines.append("## Acceptance checks")
    lines.append("")
    lines.append("```")
    lines.append(json.dumps({
        "overall": overall_checks,
        "fired": group_packages["fired"]["checks"],
        "non_fired": group_packages["non_fired"]["checks"],
    }, indent=2, ensure_ascii=False))
    lines.append("```")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    run_dir = find_latest_one_shot_run(input_dir)
    print(f"Using one-shot run directory: {run_dir}")

    run_summary, all_trials, all_steps, seed_summary = load_run_data(run_dir, args.ablation)

    shock_trial = infer_shock_trial(all_trials)
    seed_effect_df = compute_seed_effect(all_trials, shock_trial)

    fired_trials = subset_by_seed_effect(all_trials, seed_effect_df, "fired")
    non_fired_trials = subset_by_seed_effect(all_trials, seed_effect_df, "non_fired")

    fired_steps = subset_by_seed_effect(all_steps, seed_effect_df, "fired")
    non_fired_steps = subset_by_seed_effect(all_steps, seed_effect_df, "non_fired")

    overall_package = compute_group_package(all_trials, all_steps, shock_trial, "overall")
    group_packages = {
        "fired": compute_group_package(fired_trials, fired_steps, shock_trial, "fired"),
        "non_fired": compute_group_package(non_fired_trials, non_fired_steps, shock_trial, "non_fired"),
    }
    overall_checks = overall_package["checks"]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save tables: overall
    save_table(overall_package["pre_post_df"], output_dir / "one_shot_pre_post.csv")
    save_table(overall_package["trial_df"], output_dir / "one_shot_trial_summary_by_trial.csv")
    save_table(overall_package["hrisk_df"], output_dir / "one_shot_hrisk_by_trial.csv")
    save_table(overall_package["mode_df"], output_dir / "one_shot_mode_pre_post.csv")
    save_table(overall_package["latency_df"], output_dir / "one_shot_latency_pre_post.csv")
    save_table(seed_effect_df, output_dir / "one_shot_seed_effect.csv")

    # Save tables: subgroup
    save_table(group_packages["fired"]["pre_post_df"], output_dir / "one_shot_pre_post_fired.csv")
    save_table(group_packages["non_fired"]["pre_post_df"], output_dir / "one_shot_pre_post_non_fired.csv")
    save_table(group_packages["fired"]["trial_df"], output_dir / "one_shot_trial_summary_by_trial_fired.csv")
    save_table(group_packages["non_fired"]["trial_df"], output_dir / "one_shot_trial_summary_by_trial_non_fired.csv")
    save_table(group_packages["fired"]["hrisk_df"], output_dir / "one_shot_hrisk_by_trial_fired.csv")
    save_table(group_packages["non_fired"]["hrisk_df"], output_dir / "one_shot_hrisk_by_trial_non_fired.csv")
    save_table(group_packages["fired"]["mode_df"], output_dir / "one_shot_mode_pre_post_fired.csv")
    save_table(group_packages["non_fired"]["mode_df"], output_dir / "one_shot_mode_pre_post_non_fired.csv")
    save_table(group_packages["fired"]["latency_df"], output_dir / "one_shot_latency_pre_post_fired.csv")
    save_table(group_packages["non_fired"]["latency_df"], output_dir / "one_shot_latency_pre_post_non_fired.csv")

    if seed_summary is not None:
        save_table(seed_summary, output_dir / "one_shot_seed_summary_copy.csv")

    with open(output_dir / "one_shot_acceptance_check.json", "w", encoding="utf-8") as f:
        json.dump({
            "overall": overall_checks,
            "fired": group_packages["fired"]["checks"],
            "non_fired": group_packages["non_fired"]["checks"],
        }, f, indent=2, ensure_ascii=False)

    # Overall plots
    plot_pre_post(
        overall_package["pre_post_df"],
        "Stage 3.1B: One-shot pre / shock / post",
        output_dir / "Figure_3_1B_One_Shot_Pre_Post.png"
    )
    plot_hrisk_by_trial(
        overall_package["hrisk_df"],
        shock_trial,
        "Stage 3.1B: h_risk around shock",
        output_dir / "Figure_3_1B_h_Risk_Around_Shock.png"
    )
    plot_popen_by_trial(
        overall_package["trial_df"],
        shock_trial,
        args.rolling_window,
        "Stage 3.1B: P(open) by trial",
        output_dir / "Figure_3_1B_P_Open_By_Trial.png"
    )
    plot_mode_pre_post(
        overall_package["mode_df"],
        "Stage 3.1B: mode_at_junction pre vs post",
        output_dir / "Figure_3_1B_Mode_At_Junction_Pre_Post.png"
    )
    plot_latency_pre_post(
        overall_package["latency_df"],
        "Stage 3.1B: pre vs post metrics",
        output_dir / "Figure_3_1B_Latency_Pre_Post.png"
    )

    # Subgroup plots
    plot_pre_post_by_subgroup(
        group_packages,
        output_dir / "Figure_3_1B_One_Shot_Pre_Post_By_Subgroup.png"
    )
    plot_hrisk_by_subgroup(
        group_packages,
        shock_trial,
        output_dir / "Figure_3_1B_h_Risk_Around_Shock_By_Subgroup.png"
    )
    plot_popen_by_subgroup(
        group_packages,
        shock_trial,
        args.rolling_window,
        output_dir / "Figure_3_1B_P_Open_By_Trial_By_Subgroup.png"
    )
    plot_seed_delta(
        seed_effect_df,
        output_dir / "Figure_3_1B_SeedDelta_By_Shock_Status.png"
    )

    write_report(
        output_dir=output_dir,
        run_summary=run_summary,
        overall_package=overall_package,
        group_packages=group_packages,
        seed_effect_df=seed_effect_df,
        overall_checks=overall_checks,
    )

    print("\n=== One-shot acceptance checks ===")
    print(json.dumps({
        "overall": overall_checks,
        "fired": group_packages["fired"]["checks"],
        "non_fired": group_packages["non_fired"]["checks"],
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()