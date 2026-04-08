"""
Stage 3.1B canonical-condition summarizer.

Reads three canonical run_summary.json files and optionally the matching
seed_summary.csv files, then produces:
- canonical_summary.csv
- canonical_summary.md
- canonical_seed_summary.csv (optional, if seed CSVs are provided)

Usage example:
python summarize_stage3_1b_canonical.py \
  --balanced-json balanced_conflict_full_run_summary.json \
  --reward-json reward_dominant_full_run_summary.json \
  --threat-json threat_dominant_full_run_summary.json \
  --balanced-seeds balanced_conflict_full_seed_summary.csv \
  --reward-seeds reward_dominant_full_seed_summary.csv \
  --threat-seeds threat_dominant_full_seed_summary.csv \
  --output-dir canonical_summary_out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_condition_row(summary: Dict[str, Any], label: str) -> Dict[str, Any]:
    cs = summary["condition_summary"]
    mode_dist = cs.get("mode_at_junction_distribution", {}) or {}

    return {
        "condition_name": label,
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
        "mode_at_junction_top": max(mode_dist, key=mode_dist.get) if mode_dist else "",
        "mode_at_junction_top_p": mode_dist.get(max(mode_dist, key=mode_dist.get), 0.0) if mode_dist else 0.0,
        "mode_at_junction_distribution": json.dumps(mode_dist, ensure_ascii=False),
    }


def format_float(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def build_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---" for _ in cols]) + "|"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(format_float(row[c]) for c in cols) + " |")
    return "\n".join([header, sep] + rows)


def compute_acceptance_notes(df: pd.DataFrame) -> Dict[str, Any]:
    lookup = {row["condition_name"]: row for _, row in df.iterrows()}

    reward = lookup.get("reward_dominant")
    balanced = lookup.get("balanced_conflict")
    threat = lookup.get("threat_dominant")

    if reward is None or balanced is None or threat is None:
        return {"canonical_order_ok": False, "balanced_peak_ok": False}

    canonical_order_ok = (
        reward["p_open"] > balanced["p_open"] > threat["p_open"]
    )

    balanced_peak_ok = (
        balanced["mean_junction_pause_duration"] > reward["mean_junction_pause_duration"]
        and balanced["mean_junction_pause_duration"] > threat["mean_junction_pause_duration"]
        and balanced["mean_commit_latency"] > reward["mean_commit_latency"]
        and balanced["mean_commit_latency"] > threat["mean_commit_latency"]
        and balanced["mean_reorientation_count"] > reward["mean_reorientation_count"]
        and balanced["mean_reorientation_count"] > threat["mean_reorientation_count"]
        and balanced["mean_junction_deliberation_proxy"] > reward["mean_junction_deliberation_proxy"]
        and balanced["mean_junction_deliberation_proxy"] > threat["mean_junction_deliberation_proxy"]
    )

    return {
        "canonical_order_ok": canonical_order_ok,
        "balanced_peak_ok": balanced_peak_ok,
    }


def load_seed_table(path: Optional[Path], label: str) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    df = pd.read_csv(path)
    df["condition_name"] = label
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Stage 3.1B canonical conditions")
    parser.add_argument("--balanced-json", required=True)
    parser.add_argument("--reward-json", required=True)
    parser.add_argument("--threat-json", required=True)
    parser.add_argument("--balanced-seeds")
    parser.add_argument("--reward-seeds")
    parser.add_argument("--threat-seeds")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = {
        "balanced_conflict": load_json(Path(args.balanced_json)),
        "reward_dominant": load_json(Path(args.reward_json)),
        "threat_dominant": load_json(Path(args.threat_json)),
    }

    rows = [extract_condition_row(summary, label) for label, summary in summaries.items()]
    df = pd.DataFrame(rows)

    order = ["reward_dominant", "balanced_conflict", "threat_dominant"]
    df["_order"] = df["condition_name"].map({name: i for i, name in enumerate(order)})
    df = df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

    summary_csv = out_dir / "canonical_summary.csv"
    summary_md = out_dir / "canonical_summary.md"
    acceptance_json = out_dir / "canonical_acceptance_check.json"

    df.to_csv(summary_csv, index=False)

    notes = compute_acceptance_notes(df)

    md_parts = [
        "# Stage 3.1B Canonical Summary",
        "",
        build_markdown_table(df),
        "",
        "## Acceptance smoke checks",
        "",
        f"- canonical order `p_open(reward) > p_open(balanced) > p_open(threat)`: **{notes['canonical_order_ok']}**",
        f"- balanced conflict has peak deliberation metrics: **{notes['balanced_peak_ok']}**",
        "",
    ]
    summary_md.write_text("\n".join(md_parts), encoding="utf-8")

    with open(acceptance_json, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2, ensure_ascii=False)

    seed_frames = []
    for path_str, label in [
        (args.balanced_seeds, "balanced_conflict"),
        (args.reward_seeds, "reward_dominant"),
        (args.threat_seeds, "threat_dominant"),
    ]:
        df_seed = load_seed_table(Path(path_str) if path_str else None, label)
        if df_seed is not None:
            seed_frames.append(df_seed)

    if seed_frames:
        seed_df = pd.concat(seed_frames, ignore_index=True)
        seed_out = out_dir / "canonical_seed_summary.csv"
        seed_df.to_csv(seed_out, index=False)

    print(f"Saved: {summary_csv}")
    print(f"Saved: {summary_md}")
    print(f"Saved: {acceptance_json}")
    if seed_frames:
        print(f"Saved: {out_dir / 'canonical_seed_summary.csv'}")

    print("\nCanonical condition summary:")
    print(df[[
        "condition_name",
        "condition_id",
        "p_open",
        "p_covered",
        "mean_junction_pause_duration",
        "mean_commit_latency",
        "mean_reorientation_count",
        "mean_junction_deliberation_proxy",
        "p_commit_timeout",
        "mode_at_junction_top",
    ]].to_string(index=False))

    print("\nAcceptance smoke checks:")
    print(json.dumps(notes, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
