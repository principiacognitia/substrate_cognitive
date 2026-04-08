"""
Stage 3.1B: Full 3x3 Matrix Summary

Собирает summary по матрице reward × threat и сохраняет:
- matrix_summary.csv
- matrix_summary.md
- matrix_acceptance_check.json
- matrix_p_open.csv
- matrix_mean_pause.csv
- matrix_mean_commit_latency.csv
- matrix_p_commit_timeout.csv
- matrix_deliberation_proxy.csv

По умолчанию:
- input-dir  = logs/stage3/stage3_1b
- output-dir = logs/figures/stage3/stage3_1b_matrix_summary

Поддерживает два источника:
1. aggregated condition_summary.csv из run_stage3_1b full-grid run
2. набор individual *_run_summary.json / *_condition_summary.csv

Usage:
    python -m stage3.analysis.summarize_stage3_1b_matrix

    python -m stage3.analysis.summarize_stage3_1b_matrix ^
        --input-dir logs/stage3/stage3_1b ^
        --output-dir logs/figures/stage3/stage3_1b_matrix_summary
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


CONDITION_RE = re.compile(r"R(?P<r>[0-2])_T(?P<t>[1-3])")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize Stage 3.1B full matrix")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="logs/stage3/stage3_1b",
        help="Directory with Stage 3.1B outputs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/figures/stage3/stage3_1b_matrix_summary",
        help="Directory for summary tables"
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        help="Ablation suffix to look for (default: full)"
    )
    return parser.parse_args()


def extract_reward_threat_levels(condition_id: str):
    m = CONDITION_RE.fullmatch(condition_id)
    if not m:
        return None, None
    return int(m.group("r")), int(m.group("t"))


def is_matrix_condition(condition_id: str) -> bool:
    return CONDITION_RE.fullmatch(str(condition_id)) is not None


def find_aggregated_condition_summary(input_dir: Path, ablation: str) -> Optional[Path]:
    candidates = sorted(input_dir.glob(f"*{ablation}*condition_summary.csv"))
    for path in candidates:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "condition_id" in df.columns and len(df) >= 9:
            valid = df["condition_id"].astype(str).apply(is_matrix_condition)
            if valid.sum() >= 9:
                return path
    return None


def load_from_aggregated_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["condition_id"].astype(str).apply(is_matrix_condition)].copy()
    return df


def load_from_individual_run_summaries(input_dir: Path, ablation: str) -> pd.DataFrame:
    rows: List[Dict] = []

    for path in sorted(input_dir.glob(f"*_{ablation}_run_summary.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cs = data.get("condition_summary", {})
        cid = cs.get("condition_id", "")
        if not is_matrix_condition(cid):
            continue

        row = dict(cs)
        row["_source_file"] = path.name
        rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No matrix condition summaries found in {input_dir} for ablation={ablation}"
        )

    return pd.DataFrame(rows)


def load_matrix_summary(input_dir: Path, ablation: str) -> pd.DataFrame:
    aggregated = find_aggregated_condition_summary(input_dir, ablation)
    if aggregated is not None:
        print(f"Using aggregated condition summary: {aggregated}")
        df = load_from_aggregated_csv(aggregated)
    else:
        print("Aggregated condition summary not found; scanning individual run_summary.json files")
        df = load_from_individual_run_summaries(input_dir, ablation)

    if "p_open_mean" in df.columns and "p_open" not in df.columns:
        df["p_open"] = df["p_open_mean"]
    if "p_open_std" in df.columns and "p_open_std_across_seeds" not in df.columns:
        df["p_open_std_across_seeds"] = df["p_open_std"]

    reward_levels = []
    threat_levels = []

    for cid in df["condition_id"].astype(str):
        r, t = extract_reward_threat_levels(cid)
        reward_levels.append(r)
        threat_levels.append(t)

    df["reward_idx"] = reward_levels
    df["threat_idx"] = threat_levels

    df = df.sort_values(["reward_idx", "threat_idx"]).reset_index(drop=True)
    return df


def build_pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    pivot = df.pivot(index="threat_idx", columns="reward_idx", values=value_col)
    pivot = pivot.sort_index().sort_index(axis=1)
    pivot.index = [f"T{idx}" for idx in pivot.index]
    pivot.columns = [f"R{idx}" for idx in pivot.columns]
    return pivot


def compute_acceptance_checks(df: pd.DataFrame) -> Dict:
    checks: Dict[str, object] = {}

    # 1. monotonic increase in p_open with reward at fixed threat
    reward_monotonic = {}
    for t in sorted(df["threat_idx"].unique()):
        sub = df[df["threat_idx"] == t].sort_values("reward_idx")
        vals = sub["p_open"].tolist()
        reward_monotonic[f"T{t}"] = bool(vals[0] <= vals[1] <= vals[2])

    # 2. monotonic decrease in p_open with threat at fixed reward
    threat_monotonic = {}
    for r in sorted(df["reward_idx"].unique()):
        sub = df[df["reward_idx"] == r].sort_values("threat_idx")
        vals = sub["p_open"].tolist()
        threat_monotonic[f"R{r}"] = bool(vals[0] >= vals[1] >= vals[2])

    checks["reward_monotonicity_per_threat"] = reward_monotonic
    checks["threat_monotonicity_per_reward"] = threat_monotonic
    checks["reward_axis_order_ok"] = all(reward_monotonic.values())
    checks["threat_axis_order_ok"] = all(threat_monotonic.values())

    # 3. center peak for deliberation
    center_row = df[df["condition_id"] == "R1_T2"]
    if len(center_row) == 1:
        center_pause = float(center_row["mean_junction_pause_duration"].iloc[0])
        center_latency = float(center_row["mean_commit_latency"].iloc[0])
        center_reorient = float(center_row["mean_reorientation_count"].iloc[0])
        center_proxy = float(center_row["mean_junction_deliberation_proxy"].iloc[0])

        checks["center_pause_peak"] = bool(center_pause >= df["mean_junction_pause_duration"].max())
        checks["center_latency_peak"] = bool(center_latency >= df["mean_commit_latency"].max())
        checks["center_reorientation_peak"] = bool(center_reorient >= df["mean_reorientation_count"].max())
        checks["center_proxy_peak"] = bool(center_proxy >= df["mean_junction_deliberation_proxy"].max())

        checks["balanced_conflict_peak_ok"] = all([
            checks["center_pause_peak"],
            checks["center_latency_peak"],
            checks["center_reorientation_peak"],
            checks["center_proxy_peak"],
        ])
    else:
        checks["balanced_conflict_peak_ok"] = False

    return checks


def save_csvs(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "matrix_summary.csv", index=False)

    build_pivot(df, "p_open").to_csv(output_dir / "matrix_p_open.csv")
    build_pivot(df, "mean_junction_pause_duration").to_csv(output_dir / "matrix_mean_pause.csv")
    build_pivot(df, "mean_commit_latency").to_csv(output_dir / "matrix_mean_commit_latency.csv")
    build_pivot(df, "p_commit_timeout").to_csv(output_dir / "matrix_p_commit_timeout.csv")
    build_pivot(df, "mean_junction_deliberation_proxy").to_csv(output_dir / "matrix_deliberation_proxy.csv")

    print(f"✓ Saved: {output_dir / 'matrix_summary.csv'}")
    print(f"✓ Saved: {output_dir / 'matrix_p_open.csv'}")
    print(f"✓ Saved: {output_dir / 'matrix_mean_pause.csv'}")
    print(f"✓ Saved: {output_dir / 'matrix_mean_commit_latency.csv'}")
    print(f"✓ Saved: {output_dir / 'matrix_p_commit_timeout.csv'}")
    print(f"✓ Saved: {output_dir / 'matrix_deliberation_proxy.csv'}")


def save_markdown(df: pd.DataFrame, checks: Dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Stage 3.1B Full Matrix Summary")
    lines.append("")
    lines.append(df.to_markdown(index=False))
    lines.append("")
    lines.append("## Acceptance checks")
    lines.append("")
    lines.append(f"- reward axis monotonicity per threat: **{checks['reward_axis_order_ok']}**")
    lines.append(f"- threat axis monotonicity per reward: **{checks['threat_axis_order_ok']}**")
    lines.append(f"- balanced conflict center peak: **{checks['balanced_conflict_peak_ok']}**")
    lines.append("")

    out_path = output_dir / "matrix_summary.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✓ Saved: {out_path}")


def save_checks(checks: Dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "matrix_acceptance_check.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved: {out_path}")


def print_console_summary(df: pd.DataFrame, checks: Dict):
    cols = [
        "condition_id",
        "p_open",
        "p_covered",
        "mean_junction_pause_duration",
        "mean_commit_latency",
        "mean_reorientation_count",
        "mean_junction_deliberation_proxy",
        "p_commit_timeout",
    ]
    print("\n=== Stage 3.1B Matrix Summary ===")
    print(df[cols].to_string(index=False))

    print("\n=== Acceptance checks ===")
    print(json.dumps(checks, indent=2, ensure_ascii=False))


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    df = load_matrix_summary(input_dir, args.ablation)
    checks = compute_acceptance_checks(df)

    save_csvs(df, output_dir)
    save_markdown(df, checks, output_dir)
    save_checks(checks, output_dir)
    print_console_summary(df, checks)


if __name__ == "__main__":
    main()