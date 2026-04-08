"""
Stage 3.1B: One-Shot Summary

Собирает summary по one-shot run и сохраняет:
- one_shot_summary.csv
- one_shot_summary.md
- one_shot_acceptance_check.json
- one_shot_pre_post.csv
- one_shot_mode_pre_post.csv
- one_shot_latency_pre_post.csv
- one_shot_hrisk_by_trial.csv (если step logs доступны)
- one_shot_hRisk_note.txt (если step logs недоступны)

По умолчанию:
- input-dir  = logs/stage3/stage3_1b
- output-dir = logs/figures/stage3/stage3_1b_one_shot_summary

Поддерживает:
1. указание конкретной run-папки one_shot_*
2. указание корневой папки logs/stage3/stage3_1b с auto-discovery latest one-shot run

Ожидаемые файлы в run-папке:
- one_shot_full_run_summary.json
- one_shot_full_all_trials.csv
- optionally: step-level CSV with h_risk and trial columns

Usage:
    python -m stage3.analysis.summarize_stage3_1b_one_shot

    python -m stage3.analysis.summarize_stage3_1b_one_shot ^
        --input-dir logs/stage3/stage3_1b ^
        --output-dir logs/figures/stage3/stage3_1b_one_shot_summary
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize Stage 3.1B one-shot run")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="logs/stage3/stage3_1b",
        help="Run directory or root directory containing one_shot_* timestamped runs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/figures/stage3/stage3_1b_one_shot_summary",
        help="Directory for summary outputs"
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        help="Ablation suffix to look for (default: full)"
    )
    return parser.parse_args()


def is_one_shot_run_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("one_shot_")


def find_latest_one_shot_run(input_dir: Path) -> Path:
    """
    Если input_dir уже run-dir, используем его.
    Иначе ищем последний one_shot_* run directory внутри input_dir.
    """
    if input_dir.is_dir():
        # Case 1: user passed the run directory itself
        if any(input_dir.glob("*_run_summary.json")) and any(input_dir.glob("*_all_trials.csv")):
            return input_dir

        # Case 2: user passed base logs directory
        candidates = [p for p in input_dir.iterdir() if is_one_shot_run_dir(p)]
        if candidates:
            candidates = sorted(candidates, key=lambda p: p.name)
            return candidates[-1]

    raise FileNotFoundError(
        f"Could not find one-shot run directory in {input_dir}"
    )


def find_single_file(run_dir: Path, pattern: str) -> Path:
    files = sorted(run_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {run_dir}")
    return files[-1]


def try_find_optional_file(run_dir: Path, patterns: List[str]) -> Optional[Path]:
    for pattern in patterns:
        files = sorted(run_dir.glob(pattern))
        if files:
            return files[-1]
    return None


def load_run_data(run_dir: Path, ablation: str):
    run_summary_path = find_single_file(run_dir, f"*_{ablation}_run_summary.json")
    all_trials_path = find_single_file(run_dir, f"*_{ablation}_all_trials.csv")

    with open(run_summary_path, "r", encoding="utf-8") as f:
        run_summary = json.load(f)

    all_trials = pd.read_csv(all_trials_path)

    # Optional step log for h_risk
    step_log_path = try_find_optional_file(
        run_dir,
        [
            f"*_{ablation}_all_steps.csv",
            f"*_{ablation}_step_log.csv",
            f"*_{ablation}_steps.csv",
            f"*_{ablation}_all_step_logs.csv",
        ]
    )

    step_df = None
    if step_log_path is not None:
        try:
            step_df = pd.read_csv(step_log_path)
        except Exception:
            step_df = None

    return run_summary_path, all_trials_path, run_summary, all_trials, step_df


def infer_one_shot_trial(all_trials: pd.DataFrame) -> int:
    if "one_shot_trial" in all_trials.columns:
        vals = [v for v in all_trials["one_shot_trial"].dropna().unique().tolist() if int(v) >= 0]
        if vals:
            return int(vals[0])

    # Fallback to canonical default from spec
    return 30


def split_pre_post(all_trials: pd.DataFrame, shock_trial: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pre_df = all_trials[all_trials["trial"] < shock_trial].copy()
    shock_df = all_trials[all_trials["trial"] == shock_trial].copy()
    post_df = all_trials[all_trials["trial"] > shock_trial].copy()
    return pre_df, shock_df, post_df


def compute_block_summary(df: pd.DataFrame, label: str) -> dict:
    if len(df) == 0:
        return {
            "block": label,
            "n_trials": 0,
            "p_open": 0.0,
            "p_covered": 0.0,
            "mean_commit_latency": 0.0,
            "mean_junction_pause_duration": 0.0,
            "mean_reorientation_count": 0.0,
            "mean_junction_deliberation_proxy": 0.0,
            "mode_at_junction_top": "",
            "mode_at_junction_top_p": 0.0,
        }

    mode_dist = (
        df["mode_at_junction"].value_counts(normalize=True).to_dict()
        if "mode_at_junction" in df.columns else {}
    )
    mode_top = max(mode_dist, key=mode_dist.get) if mode_dist else ""
    mode_top_p = float(mode_dist.get(mode_top, 0.0)) if mode_top else 0.0

    return {
        "block": label,
        "n_trials": int(len(df)),
        "p_open": float((df["path_choice"] == "open").mean()),
        "p_covered": float((df["path_choice"] == "covered").mean()),
        "mean_commit_latency": float(df["commit_latency"].mean()) if "commit_latency" in df.columns else 0.0,
        "mean_junction_pause_duration": float(df["junction_pause_duration"].mean()) if "junction_pause_duration" in df.columns else 0.0,
        "mean_reorientation_count": float(df["reorientation_count"].mean()) if "reorientation_count" in df.columns else 0.0,
        "mean_junction_deliberation_proxy": float(df["junction_deliberation_proxy"].mean()) if "junction_deliberation_proxy" in df.columns else 0.0,
        "mode_at_junction_top": mode_top,
        "mode_at_junction_top_p": mode_top_p,
    }


def compute_mode_pre_post(pre_df: pd.DataFrame, post_df: pd.DataFrame) -> pd.DataFrame:
    modes = sorted(set(pre_df.get("mode_at_junction", pd.Series(dtype=str)).dropna().tolist()) |
                   set(post_df.get("mode_at_junction", pd.Series(dtype=str)).dropna().tolist()))

    rows = []
    for mode in modes:
        rows.append({
            "mode_at_junction": mode,
            "pre_p": float((pre_df["mode_at_junction"] == mode).mean()) if len(pre_df) else 0.0,
            "post_p": float((post_df["mode_at_junction"] == mode).mean()) if len(post_df) else 0.0,
            "delta_post_minus_pre": (
                float((post_df["mode_at_junction"] == mode).mean()) -
                float((pre_df["mode_at_junction"] == mode).mean())
            ) if len(pre_df) and len(post_df) else 0.0
        })
    return pd.DataFrame(rows)


def compute_latency_pre_post(pre_df: pd.DataFrame, post_df: pd.DataFrame) -> pd.DataFrame:
    rows = [{
        "metric": "mean_commit_latency",
        "pre": float(pre_df["commit_latency"].mean()) if len(pre_df) else 0.0,
        "post": float(post_df["commit_latency"].mean()) if len(post_df) else 0.0,
        "delta_post_minus_pre": (
            float(post_df["commit_latency"].mean()) - float(pre_df["commit_latency"].mean())
        ) if len(pre_df) and len(post_df) else 0.0,
    }, {
        "metric": "mean_junction_pause_duration",
        "pre": float(pre_df["junction_pause_duration"].mean()) if len(pre_df) else 0.0,
        "post": float(post_df["junction_pause_duration"].mean()) if len(post_df) else 0.0,
        "delta_post_minus_pre": (
            float(post_df["junction_pause_duration"].mean()) - float(pre_df["junction_pause_duration"].mean())
        ) if len(pre_df) and len(post_df) else 0.0,
    }, {
        "metric": "mean_reorientation_count",
        "pre": float(pre_df["reorientation_count"].mean()) if len(pre_df) else 0.0,
        "post": float(post_df["reorientation_count"].mean()) if len(post_df) else 0.0,
        "delta_post_minus_pre": (
            float(post_df["reorientation_count"].mean()) - float(pre_df["reorientation_count"].mean())
        ) if len(pre_df) and len(post_df) else 0.0,
    }]
    return pd.DataFrame(rows)


def compute_hrisk_by_trial(step_df: Optional[pd.DataFrame], output_dir: Path) -> Optional[pd.DataFrame]:
    """
    h_risk trajectory можно построить только если есть step-level log с колонками:
    - trial
    - h_risk
    """
    if step_df is None:
        note = output_dir / "one_shot_hRisk_note.txt"
        note.write_text(
            "h_risk trajectory unavailable: no step-level CSV found.\n"
            "Current runner appears to save trial-level summaries only.\n",
            encoding="utf-8"
        )
        return None

    required = {"trial", "h_risk"}
    if not required.issubset(set(step_df.columns)):
        note = output_dir / "one_shot_hRisk_note.txt"
        note.write_text(
            "h_risk trajectory unavailable: step-level CSV exists, but required columns are missing.\n"
            f"Required: {sorted(required)}\n"
            f"Found: {sorted(step_df.columns.tolist())}\n",
            encoding="utf-8"
        )
        return None

    hrisk_df = (
        step_df
        .groupby("trial", dropna=False)["h_risk"]
        .mean()
        .reset_index()
        .rename(columns={"h_risk": "mean_h_risk"})
        .sort_values("trial")
    )
    hrisk_df.to_csv(output_dir / "one_shot_hrisk_by_trial.csv", index=False)
    return hrisk_df


def compute_acceptance_checks(pre_df: pd.DataFrame, shock_df: pd.DataFrame, post_df: pd.DataFrame, hrisk_df: Optional[pd.DataFrame], shock_trial: int):
    p_open_pre = float((pre_df["path_choice"] == "open").mean()) if len(pre_df) else 0.0
    p_open_post = float((post_df["path_choice"] == "open").mean()) if len(post_df) else 0.0

    one_shot_logged = False
    if "one_shot_fired" in shock_df.columns:
        one_shot_logged = bool(shock_df["one_shot_fired"].astype(bool).any())
    elif "one_shot_active" in shock_df.columns:
        one_shot_logged = bool(shock_df["one_shot_active"].astype(bool).any())

    persistence_exists = p_open_post < p_open_pre

    hrisk_post_gt_pre = None
    if hrisk_df is not None and len(hrisk_df):
        pre_hr = hrisk_df[hrisk_df["trial"] < shock_trial]["mean_h_risk"].mean()
        post_hr = hrisk_df[hrisk_df["trial"] > shock_trial]["mean_h_risk"].mean()
        if pd.notna(pre_hr) and pd.notna(post_hr):
            hrisk_post_gt_pre = bool(post_hr > pre_hr)

    return {
        "shock_trial": shock_trial,
        "pre_p_open": p_open_pre,
        "post_p_open": p_open_post,
        "post_shock_p_open_lt_pre": persistence_exists,
        "one_shot_logged": one_shot_logged,
        "hrisk_post_gt_pre": hrisk_post_gt_pre,
    }


def df_to_codeblock(df: pd.DataFrame) -> str:
    return "```\n" + df.to_string(index=False) + "\n```"


def save_outputs(
    output_dir: Path,
    run_summary: dict,
    all_trials: pd.DataFrame,
    pre_post_df: pd.DataFrame,
    mode_df: pd.DataFrame,
    latency_df: pd.DataFrame,
    checks: dict,
    hrisk_df: Optional[pd.DataFrame],
):
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_row = pd.DataFrame([{
        "condition_id": run_summary.get("condition_summary", {}).get("condition_id", ""),
        "ablation": run_summary.get("condition_summary", {}).get("ablation", ""),
        "n_seeds": run_summary.get("condition_summary", {}).get("n_seeds", 0),
        "n_trials_total": run_summary.get("condition_summary", {}).get("n_trials_total", 0),
        "shock_trial": checks["shock_trial"],
        "pre_p_open": checks["pre_p_open"],
        "post_p_open": checks["post_p_open"],
        "delta_post_minus_pre": checks["post_p_open"] - checks["pre_p_open"],
        "post_shock_p_open_lt_pre": checks["post_shock_p_open_lt_pre"],
        "one_shot_logged": checks["one_shot_logged"],
        "hrisk_post_gt_pre": checks["hrisk_post_gt_pre"],
    }])

    summary_row.to_csv(output_dir / "one_shot_summary.csv", index=False)
    pre_post_df.to_csv(output_dir / "one_shot_pre_post.csv", index=False)
    mode_df.to_csv(output_dir / "one_shot_mode_pre_post.csv", index=False)
    latency_df.to_csv(output_dir / "one_shot_latency_pre_post.csv", index=False)

    with open(output_dir / "one_shot_acceptance_check.json", "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("# Stage 3.1B One-Shot Summary")
    lines.append("")
    lines.append("## Summary row")
    lines.append("")
    lines.append(df_to_codeblock(summary_row))
    lines.append("")
    lines.append("## Pre / Shock / Post")
    lines.append("")
    lines.append(df_to_codeblock(pre_post_df))
    lines.append("")
    lines.append("## Mode at junction: pre vs post")
    lines.append("")
    lines.append(df_to_codeblock(mode_df))
    lines.append("")
    lines.append("## Latency / pause / reorientation: pre vs post")
    lines.append("")
    lines.append(df_to_codeblock(latency_df))
    lines.append("")
    lines.append("## Acceptance checks")
    lines.append("")
    lines.append("```")
    lines.append(json.dumps(checks, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")

    if hrisk_df is not None:
        lines.append("## h_risk by trial")
        lines.append("")
        lines.append(df_to_codeblock(hrisk_df.head(20)))
    else:
        lines.append("## h_risk by trial")
        lines.append("")
        lines.append("Unavailable from current outputs; see one_shot_hRisk_note.txt")

    with open(output_dir / "one_shot_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    run_dir = find_latest_one_shot_run(input_dir)
    print(f"Using one-shot run directory: {run_dir}")

    _, _, run_summary, all_trials, step_df = load_run_data(run_dir, args.ablation)

    shock_trial = infer_one_shot_trial(all_trials)
    pre_df, shock_df, post_df = split_pre_post(all_trials, shock_trial)

    pre_post_rows = [
        compute_block_summary(pre_df, "pre"),
        compute_block_summary(shock_df, "shock"),
        compute_block_summary(post_df, "post"),
    ]
    pre_post_df = pd.DataFrame(pre_post_rows)

    mode_df = compute_mode_pre_post(pre_df, post_df)
    latency_df = compute_latency_pre_post(pre_df, post_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    hrisk_df = compute_hrisk_by_trial(step_df, output_dir)

    checks = compute_acceptance_checks(pre_df, shock_df, post_df, hrisk_df, shock_trial)

    save_outputs(
        output_dir=output_dir,
        run_summary=run_summary,
        all_trials=all_trials,
        pre_post_df=pre_post_df,
        mode_df=mode_df,
        latency_df=latency_df,
        checks=checks,
        hrisk_df=hrisk_df,
    )

    print("\n=== One-shot acceptance checks ===")
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()