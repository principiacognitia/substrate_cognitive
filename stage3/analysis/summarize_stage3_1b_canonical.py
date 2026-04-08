"""
Stage 3.1B: Canonical Conditions Summary

Собирает summary для трёх canonical conditions:
- reward_dominant
- balanced_conflict
- threat_dominant

По умолчанию:
- input-dir  = logs/stage3/stage3_1b
- output-dir = logs/figures/stage3/stage3_1b_canonical_summary

Поддерживает:
1. auto-discovery latest timestamped run dirs inside input-dir
2. explicit input-dir if user points at a folder containing the three run dirs

Сохраняет:
- canonical_summary.csv
- canonical_summary.md
- canonical_acceptance_check.json
- canonical_seed_summary.csv (если seed_summary найдены)

Usage:
    python -m stage3.analysis.summarize_stage3_1b_canonical

    python -m stage3.analysis.summarize_stage3_1b_canonical ^
        --input-dir logs/stage3/stage3_1b ^
        --output-dir logs/figures/stage3/stage3_1b_canonical_summary
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd


CANONICAL_PREFIXES = {
    "reward_dominant": "reward_dominant_full_",
    "balanced_conflict": "balanced_conflict_full_",
    "threat_dominant": "threat_dominant_full_",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize Stage 3.1B canonical conditions")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="logs/stage3/stage3_1b",
        help="Base directory containing timestamped canonical run subdirs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/figures/stage3/stage3_1b_canonical_summary",
        help="Directory for summary outputs"
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        help="Ablation suffix to look for (default: full)"
    )
    return parser.parse_args()


def find_latest_run_dir(base_dir: Path, prefix: str) -> Path:
    candidates = [
        p for p in base_dir.iterdir()
        if p.is_dir() and p.name.startswith(prefix)
    ]
    if not candidates:
        raise FileNotFoundError(f"No run directories found for prefix {prefix} in {base_dir}")
    candidates = sorted(candidates, key=lambda p: p.name)
    return candidates[-1]


def load_run_summary(run_dir: Path, ablation: str) -> dict:
    files = sorted(run_dir.glob(f"*_{ablation}_run_summary.json"))
    if not files:
        raise FileNotFoundError(f"No run_summary.json found in {run_dir}")
    with open(files[-1], "r", encoding="utf-8") as f:
        return json.load(f)


def load_seed_summary(run_dir: Path, ablation: str) -> Optional[pd.DataFrame]:
    files = sorted(run_dir.glob(f"*_{ablation}_seed_summary.csv"))
    if not files:
        return None
    return pd.read_csv(files[-1])


def extract_mode_top(distribution):
    if isinstance(distribution, str):
        try:
            distribution = json.loads(distribution)
        except Exception:
            distribution = {}

    if not distribution:
        return "", 0.0

    top = max(distribution, key=distribution.get)
    return top, float(distribution[top])


def build_summary_row(condition_name: str, run_summary: dict) -> dict:
    cs = run_summary.get("condition_summary", {})
    mode_dist = cs.get("mode_at_junction_distribution", {})
    mode_top, mode_top_p = extract_mode_top(mode_dist)

    return {
        "condition_name": condition_name,
        "condition_id": cs.get("condition_id", ""),
        "ablation": cs.get("ablation", ""),
        "n_seeds": cs.get("n_seeds", 0),
        "n_trials_total": cs.get("n_trials_total", 0),
        "p_open": cs.get("p_open", 0.0),
        "p_covered": cs.get("p_covered", 0.0),
        "mean_junction_pause_duration": cs.get("mean_junction_pause_duration", 0.0),
        "mean_commit_latency": cs.get("mean_commit_latency", 0.0),
        "mean_reorientation_count": cs.get("mean_reorientation_count", 0.0),
        "mean_junction_deliberation_proxy": cs.get("mean_junction_deliberation_proxy", 0.0),
        "p_commit_bound": cs.get("p_commit_bound", 0.0),
        "p_commit_timeout": cs.get("p_commit_timeout", 0.0),
        "mode_at_junction_top": mode_top,
        "mode_at_junction_top_p": mode_top_p,
        "mode_at_junction_distribution": json.dumps(mode_dist, ensure_ascii=False),
    }


def compute_acceptance_checks(df: pd.DataFrame) -> dict:
    by_name = {row["condition_name"]: row for _, row in df.iterrows()}

    order_ok = (
        by_name["reward_dominant"]["p_open"] >
        by_name["balanced_conflict"]["p_open"] >
        by_name["threat_dominant"]["p_open"]
    )

    balanced = by_name["balanced_conflict"]

    balanced_peak_ok = all([
        balanced["mean_junction_pause_duration"] >= df["mean_junction_pause_duration"].max(),
        balanced["mean_commit_latency"] >= df["mean_commit_latency"].max(),
        balanced["mean_reorientation_count"] >= df["mean_reorientation_count"].max(),
        balanced["mean_junction_deliberation_proxy"] >= df["mean_junction_deliberation_proxy"].max(),
    ])

    return {
        "canonical_order_ok": bool(order_ok),
        "balanced_peak_ok": bool(balanced_peak_ok),
    }


def df_to_codeblock(df: pd.DataFrame) -> str:
    return "```\n" + df.to_string(index=False) + "\n```"


def save_outputs(
    output_dir: Path,
    summary_df: pd.DataFrame,
    acceptance: dict,
    seed_df: Optional[pd.DataFrame],
):
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(output_dir / "canonical_summary.csv", index=False)
    with open(output_dir / "canonical_acceptance_check.json", "w", encoding="utf-8") as f:
        json.dump(acceptance, f, indent=2, ensure_ascii=False)

    if seed_df is not None:
        seed_df.to_csv(output_dir / "canonical_seed_summary.csv", index=False)

    lines = []
    lines.append("# Stage 3.1B Canonical Summary")
    lines.append("")
    lines.append(df_to_codeblock(summary_df))
    lines.append("")
    lines.append("## Acceptance smoke checks")
    lines.append("")
    lines.append(f"- canonical order `p_open(reward) > p_open(balanced) > p_open(threat)`: **{acceptance['canonical_order_ok']}**")
    lines.append(f"- balanced conflict has peak deliberation metrics: **{acceptance['balanced_peak_ok']}**")
    lines.append("")

    with open(output_dir / "canonical_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()

    base_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    run_dirs = {
        name: find_latest_run_dir(base_dir, prefix)
        for name, prefix in CANONICAL_PREFIXES.items()
    }

    rows: List[dict] = []
    seed_frames: List[pd.DataFrame] = []

    for condition_name, run_dir in run_dirs.items():
        run_summary = load_run_summary(run_dir, args.ablation)
        row = build_summary_row(condition_name, run_summary)
        rows.append(row)

        seed_df = load_seed_summary(run_dir, args.ablation)
        if seed_df is not None:
            seed_df = seed_df.copy()
            seed_df["condition_name"] = condition_name
            seed_frames.append(seed_df)

    summary_df = pd.DataFrame(rows)
    desired_order = ["reward_dominant", "balanced_conflict", "threat_dominant"]
    summary_df["condition_name"] = pd.Categorical(summary_df["condition_name"], desired_order, ordered=True)
    summary_df = summary_df.sort_values("condition_name").reset_index(drop=True)

    acceptance = compute_acceptance_checks(summary_df)

    seed_df = pd.concat(seed_frames, ignore_index=True) if seed_frames else None
    save_outputs(output_dir, summary_df, acceptance, seed_df)

    print("\n=== Canonical summary ===")
    print(summary_df.to_string(index=False))
    print("\n=== Acceptance checks ===")
    print(json.dumps(acceptance, indent=2, ensure_ascii=False))
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()