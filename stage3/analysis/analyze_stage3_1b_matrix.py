"""
Stage 3.1B: Matrix Heatmap Analysis

Строит heatmap-визуализацию full 3x3 reward × threat matrix
из уже сохранённых condition summaries / matrix_summary.csv.

Поддерживает два режима входа:
1. input-dir указывает на папку с matrix_summary.csv
2. input-dir указывает на папку logs/stage3/stage3_1b,
   тогда скрипт сам находит последний grid_3x3_* run

Default paths:
    input-dir  = logs/stage3/stage3_1b
    output-dir = logs/figures/stage3/stage3_1b_matrix_analysis

Usage:
    python -m stage3.analysis.analyze_stage3_1b_matrix

    python -m stage3.analysis.analyze_stage3_1b_matrix ^
        --input-dir logs/stage3/stage3_1b\grid_3x3_full_20260408_130512 ^
        --output-dir logs/figures/stage3/stage3_1b_matrix_analysis
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CONDITION_RE = re.compile(r"R(?P<r>[0-2])_T(?P<t>[1-3])")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Stage 3.1B full matrix as heatmaps")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="logs/stage3/stage3_1b",
        help="Matrix summary dir or base Stage 3.1B logs dir"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/figures/stage3/stage3_1b_matrix_analysis",
        help="Directory for heatmaps and report"
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        help="Ablation suffix (default: full)"
    )
    return parser.parse_args()


def parse_condition_id(condition_id: str) -> Tuple[Optional[int], Optional[int]]:
    m = CONDITION_RE.fullmatch(str(condition_id))
    if not m:
        return None, None
    return int(m.group("r")), int(m.group("t"))


def is_matrix_condition(condition_id: str) -> bool:
    return CONDITION_RE.fullmatch(str(condition_id)) is not None


def find_latest_grid_run(base_dir: Path) -> Path:
    candidates = [
        p for p in base_dir.iterdir()
        if p.is_dir() and p.name.startswith("grid_3x3_")
    ]
    if not candidates:
        raise FileNotFoundError(f"No grid_3x3_* run directories found in {base_dir}")
    candidates = sorted(candidates, key=lambda p: p.name)
    return candidates[-1]


def resolve_input_source(input_dir: Path, ablation: str) -> Path:
    """
    Возвращает путь к CSV с summary по условиям.
    Поддерживает:
    - matrix_summary.csv
    - *condition_summary.csv в run dir
    - auto-discovery latest grid run inside base logs dir
    """
    # Case 1: input-dir already contains matrix_summary.csv
    matrix_summary = input_dir / "matrix_summary.csv"
    if matrix_summary.exists():
        return matrix_summary

    # Case 2: input-dir is a specific run directory containing condition summary
    condition_candidates = sorted(input_dir.glob(f"*_{ablation}_condition_summary.csv"))
    for path in condition_candidates:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "condition_id" in df.columns and df["condition_id"].astype(str).apply(is_matrix_condition).sum() >= 9:
            return path

    # Case 3: input-dir is the Stage 3.1B base dir; find latest grid run
    latest_grid = find_latest_grid_run(input_dir)
    condition_candidates = sorted(latest_grid.glob(f"*_{ablation}_condition_summary.csv"))
    if not condition_candidates:
        raise FileNotFoundError(f"No condition summary found in latest grid run: {latest_grid}")
    return condition_candidates[-1]


def load_matrix_df(source_path: Path) -> pd.DataFrame:
    df = pd.read_csv(source_path)

    if "condition_id" not in df.columns:
        raise ValueError(f"condition_id column missing in {source_path}")

    df = df[df["condition_id"].astype(str).apply(is_matrix_condition)].copy()

    # Normalize legacy/alternative column names if needed
    if "p_open_mean" in df.columns and "p_open" not in df.columns:
        df["p_open"] = df["p_open_mean"]
    if "p_covered_mean" in df.columns and "p_covered" not in df.columns:
        df["p_covered"] = df["p_covered_mean"]

    reward_idx = []
    threat_idx = []

    for cid in df["condition_id"].astype(str):
        r, t = parse_condition_id(cid)
        reward_idx.append(r)
        threat_idx.append(t)

    df["reward_idx"] = reward_idx
    df["threat_idx"] = threat_idx

    if "mode_at_junction_distribution" not in df.columns:
        df["mode_at_junction_distribution"] = "{}"

    return df.sort_values(["reward_idx", "threat_idx"]).reset_index(drop=True)


def parse_mode_distribution(value) -> Dict[str, float]:
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        try:
            return json.loads(value.replace("'", '"'))
        except Exception:
            try:
                return eval(value, {"__builtins__": {}})
            except Exception:
                return {}
    return {}


def add_mode_columns(df: pd.DataFrame) -> pd.DataFrame:
    explore_rates = []
    exploit_rates = []
    exploit_safe_rates = []
    absence_check_rates = []
    dominant_modes = []

    for raw in df["mode_at_junction_distribution"]:
        dist = parse_mode_distribution(raw)
        explore_rates.append(float(dist.get("explore", 0.0)))
        exploit_rates.append(float(dist.get("exploit", 0.0)))
        exploit_safe_rates.append(float(dist.get("exploit_safe", 0.0)))
        absence_check_rates.append(float(dist.get("absence_check", 0.0)))

        if dist:
            dominant_modes.append(max(dist, key=dist.get))
        else:
            dominant_modes.append("")

    df = df.copy()
    df["explore_rate"] = explore_rates
    df["exploit_rate"] = exploit_rates
    df["exploit_safe_rate"] = exploit_safe_rates
    df["absence_check_rate"] = absence_check_rates
    df["dominant_mode"] = dominant_modes
    return df


def build_pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    pivot = df.pivot(index="threat_idx", columns="reward_idx", values=value_col)
    pivot = pivot.sort_index().sort_index(axis=1)
    pivot.index = [f"T{idx}" for idx in pivot.index]
    pivot.columns = [f"R{idx}" for idx in pivot.columns]
    return pivot


def draw_numeric_heatmap(
    pivot_df: pd.DataFrame,
    title: str,
    output_path: Path,
    fmt: str = ".3f",
    cmap: str = "viridis",
):
    values = pivot_df.values.astype(float)

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    im = ax.imshow(values, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels(list(pivot_df.columns))
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(list(pivot_df.index))

    ax.set_xlabel("Reward level")
    ax.set_ylabel("Threat level")
    ax.set_title(title)

    # annotate each cell
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    midpoint = (vmin + vmax) / 2.0 if not math.isclose(vmin, vmax) else vmin

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            color = "white" if val <= midpoint else "black"
            ax.text(j, i, format(val, fmt), ha="center", va="center", color=color, fontsize=10)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Value", rotation=270, labelpad=12)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def draw_mode_label_heatmap(df: pd.DataFrame, output_path: Path):
    pivot = df.pivot(index="threat_idx", columns="reward_idx", values="dominant_mode")
    pivot = pivot.sort_index().sort_index(axis=1)
    pivot.index = [f"T{idx}" for idx in pivot.index]
    pivot.columns = [f"R{idx}" for idx in pivot.columns]

    # encode for background
    mode_to_num = {
        "exploit": 0,
        "explore": 1,
        "exploit_safe": 2,
        "absence_check": 3,
        "": -1,
    }
    num = (
        pivot.apply(lambda col: col.map(lambda x: mode_to_num.get(x, -1)))
             .to_numpy(dtype=float)
    )

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    im = ax.imshow(num, cmap="Pastel1", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))

    ax.set_xlabel("Reward level")
    ax.set_ylabel("Threat level")
    ax.set_title("Stage 3.1B: Dominant mode_at_junction")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, str(pivot.iloc[i, j]), ha="center", va="center", color="black", fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def compute_report_notes(df: pd.DataFrame) -> Dict[str, str]:
    max_p_open_row = df.loc[df["p_open"].idxmax()]
    min_p_open_row = df.loc[df["p_open"].idxmin()]
    max_proxy_row = df.loc[df["mean_junction_deliberation_proxy"].idxmax()]
    max_pause_row = df.loc[df["mean_junction_pause_duration"].idxmax()]
    max_latency_row = df.loc[df["mean_commit_latency"].idxmax()]

    notes = {
        "max_p_open": f"{max_p_open_row['condition_id']} ({max_p_open_row['p_open']:.4f})",
        "min_p_open": f"{min_p_open_row['condition_id']} ({min_p_open_row['p_open']:.4f})",
        "max_proxy": f"{max_proxy_row['condition_id']} ({max_proxy_row['mean_junction_deliberation_proxy']:.4f})",
        "max_pause": f"{max_pause_row['condition_id']} ({max_pause_row['mean_junction_pause_duration']:.4f})",
        "max_latency": f"{max_latency_row['condition_id']} ({max_latency_row['mean_commit_latency']:.4f})",
    }
    return notes


def compute_acceptance_checks(df: pd.DataFrame) -> Dict:
    checks: Dict[str, object] = {}

    reward_monotonic = {}
    for t in sorted(df["threat_idx"].unique()):
        sub = df[df["threat_idx"] == t].sort_values("reward_idx")
        vals = sub["p_open"].tolist()
        reward_monotonic[f"T{t}"] = bool(vals[0] <= vals[1] <= vals[2])

    threat_monotonic = {}
    for r in sorted(df["reward_idx"].unique()):
        sub = df[df["reward_idx"] == r].sort_values("threat_idx")
        vals = sub["p_open"].tolist()
        threat_monotonic[f"R{r}"] = bool(vals[0] >= vals[1] >= vals[2])

    checks["reward_monotonicity_per_threat"] = reward_monotonic
    checks["threat_monotonicity_per_reward"] = threat_monotonic
    checks["reward_axis_order_ok"] = all(reward_monotonic.values())
    checks["threat_axis_order_ok"] = all(threat_monotonic.values())

    center = df[df["condition_id"] == "R1_T2"]
    if len(center) == 1:
        center_pause = float(center["mean_junction_pause_duration"].iloc[0])
        center_latency = float(center["mean_commit_latency"].iloc[0])
        center_reorient = float(center["mean_reorientation_count"].iloc[0])
        center_proxy = float(center["mean_junction_deliberation_proxy"].iloc[0])

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


def save_report(df: pd.DataFrame, checks: Dict, output_dir: Path):
    notes = compute_report_notes(df)
    out_path = output_dir / "Stage3_1B_Matrix_Heatmap_Report.md"

    lines = []
    lines.append("# Stage 3.1B Matrix Heatmap Report")
    lines.append("")
    lines.append("## Condition summary")
    lines.append("")
    lines.append("```")
    lines.append(df.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Key extrema")
    lines.append("")
    lines.append(f"- max `p_open`: {notes['max_p_open']}")
    lines.append(f"- min `p_open`: {notes['min_p_open']}")
    lines.append(f"- max deliberation proxy: {notes['max_proxy']}")
    lines.append(f"- max pause: {notes['max_pause']}")
    lines.append(f"- max commit latency: {notes['max_latency']}")
    lines.append("")
    lines.append("## Acceptance checks")
    lines.append("")
    lines.append("```")
    lines.append(json.dumps(checks, indent=2, ensure_ascii=False))
    lines.append("```")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    source_path = resolve_input_source(input_dir, args.ablation)
    print(f"Using matrix source: {source_path}")

    df = load_matrix_df(source_path)
    df = add_mode_columns(df)

    checks = compute_acceptance_checks(df)

    # Save normalized matrix dataframe too
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "matrix_analysis_input_normalized.csv", index=False)

    # Main heatmaps
    draw_numeric_heatmap(
        build_pivot(df, "p_open"),
        "Stage 3.1B: P(open)",
        output_dir / "Figure_3_1B_ChoiceHeatmap.png",
        fmt=".3f",
        cmap="viridis",
    )

    draw_numeric_heatmap(
        build_pivot(df, "mean_junction_deliberation_proxy"),
        "Stage 3.1B: Deliberation proxy",
        output_dir / "Figure_3_1C_DeliberationHeatmap.png",
        fmt=".3f",
        cmap="magma",
    )

    draw_numeric_heatmap(
        build_pivot(df, "mean_commit_latency"),
        "Stage 3.1B: Mean commit latency",
        output_dir / "Figure_3_1C2_CommitLatencyHeatmap.png",
        fmt=".3f",
        cmap="plasma",
    )

    draw_numeric_heatmap(
        build_pivot(df, "p_commit_timeout"),
        "Stage 3.1B: P(commit timeout)",
        output_dir / "Figure_3_1C3_TimeoutHeatmap.png",
        fmt=".3f",
        cmap="cividis",
    )

    draw_numeric_heatmap(
        build_pivot(df, "explore_rate"),
        "Stage 3.1B: mode_at_junction = explore",
        output_dir / "Figure_3_1D_ModeAtJunction_ExploreRate.png",
        fmt=".3f",
        cmap="Blues",
    )

    draw_numeric_heatmap(
        build_pivot(df, "exploit_rate"),
        "Stage 3.1B: mode_at_junction = exploit",
        output_dir / "Figure_3_1D2_ModeAtJunction_ExploitRate.png",
        fmt=".3f",
        cmap="Oranges",
    )

    draw_mode_label_heatmap(
        df,
        output_dir / "Figure_3_1D3_ModeAtJunction_DominantLabel.png",
    )

    save_report(df, checks, output_dir)

    with open(output_dir / "matrix_heatmap_acceptance_check.json", "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2, ensure_ascii=False)

    print("\n=== Matrix heatmap acceptance checks ===")
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()