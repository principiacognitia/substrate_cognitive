#!/usr/bin/env python3
"""
Stage 3.1B matrix publication analyzer.

This is the publication-facing matrix layer for Stage 3.1B closure.

It is intentionally separate from one-shot analysis:
- matrix: reward x threat conflict surface
- one-shot: event-aligned shock/treat carryover
- ablation suite: localization diagnostics

Inputs:
- a grid_3x3_* run directory produced by stage3.analysis.run_stage3_1b
- or a directory containing matrix_summary.csv / *_condition_summary.csv

Outputs:
- normalized cell summary table
- seed-level cell metrics table when all_trials is available
- publication heatmaps with Stage 3.1B naming
- acceptance stats/report
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITION_RE = re.compile(r"R(?P<r>[0-2])_T(?P<t>[1-3])")

MATRIX_HEATMAPS = [
    (
        "p_open",
        "Stage 3.1B matrix: P(open)",
        "Figure_3_1B_Matrix_P_Open_Heatmap.png",
        ".3f",
        "viridis",
    ),
    (
        "p_commit_timeout",
        "Stage 3.1B matrix: P(commit timeout)",
        "Figure_3_1B_Matrix_P_Timeout_Heatmap.png",
        ".3f",
        "cividis",
    ),
    (
        "mean_commit_latency",
        "Stage 3.1B matrix: mean commit latency",
        "Figure_3_1B_Matrix_Commit_Latency_Heatmap.png",
        ".3f",
        "plasma",
    ),
    (
        "mean_junction_deliberation_proxy",
        "Stage 3.1B matrix: deliberation proxy",
        "Figure_3_1B_Matrix_Deliberation_Proxy_Heatmap.png",
        ".3f",
        "magma",
    ),
    (
        "mean_junction_pause_duration",
        "Stage 3.1B matrix: junction pause",
        "Figure_3_1B_Matrix_Junction_Pause_Heatmap.png",
        ".3f",
        "inferno",
    ),
    (
        "mean_reorientation_count",
        "Stage 3.1B matrix: reorientation count",
        "Figure_3_1B_Matrix_Reorientation_Heatmap.png",
        ".3f",
        "magma",
    ),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Analyze Stage 3.1B 3x3 matrix for publication artifacts"
    )
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--ablation", default="full")
    return ap.parse_args()


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
        if p.is_dir() and p.name.startswith("grid_")
    ]
    if not candidates:
        raise FileNotFoundError(f"No grid_* run directories found in {base_dir}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def resolve_condition_summary(input_dir: Path, ablation: str) -> Path:
    matrix_summary = input_dir / "matrix_summary.csv"
    if matrix_summary.exists():
        return matrix_summary

    condition_candidates = sorted(input_dir.glob(f"*_{ablation}_condition_summary.csv"))
    for path in condition_candidates:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if (
            "condition_id" in df.columns
            and df["condition_id"].astype(str).apply(is_matrix_condition).sum() >= 9
        ):
            return path

    latest_grid = find_latest_grid_run(input_dir)
    condition_candidates = sorted(latest_grid.glob(f"*_{ablation}_condition_summary.csv"))
    for path in condition_candidates:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if (
            "condition_id" in df.columns
            and df["condition_id"].astype(str).apply(is_matrix_condition).sum() >= 9
        ):
            return path

    raise FileNotFoundError(
        f"No matrix condition summary found under {input_dir} for ablation={ablation}"
    )


def resolve_trials_csv(input_dir: Path, ablation: str) -> Optional[Path]:
    direct = sorted(input_dir.glob(f"*_{ablation}_all_trials.csv"))
    if direct:
        return direct[0]

    if input_dir.name.startswith("grid_"):
        candidates = sorted(input_dir.glob("*_all_trials.csv"))
        return candidates[0] if candidates else None

    try:
        latest_grid = find_latest_grid_run(input_dir)
    except FileNotFoundError:
        return None

    candidates = sorted(latest_grid.glob("*_all_trials.csv"))
    return candidates[0] if candidates else None


def normalize_matrix_df(df: pd.DataFrame) -> pd.DataFrame:
    if "condition_id" not in df.columns:
        raise ValueError("condition_id column missing from matrix summary")

    df = df[df["condition_id"].astype(str).apply(is_matrix_condition)].copy()

    aliases = {
        "p_open_mean": "p_open",
        "p_covered_mean": "p_covered",
        "p_timeout_mean": "p_commit_timeout",
        "commit_latency_mean": "mean_commit_latency",
        "junction_pause_duration_mean": "mean_junction_pause_duration",
        "reorientation_count_mean": "mean_reorientation_count",
        "junction_deliberation_proxy_mean": "mean_junction_deliberation_proxy",
    }

    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    reward_idx: List[int] = []
    threat_idx: List[int] = []
    for cid in df["condition_id"].astype(str):
        r, t = parse_condition_id(cid)
        reward_idx.append(int(r))
        threat_idx.append(int(t))

    df["reward_idx"] = reward_idx
    df["threat_idx"] = threat_idx

    if "p_open" not in df.columns and "p_covered" in df.columns:
        df["p_open"] = 1.0 - pd.to_numeric(df["p_covered"], errors="coerce")

    if "p_covered" not in df.columns and "p_open" in df.columns:
        df["p_covered"] = 1.0 - pd.to_numeric(df["p_open"], errors="coerce")

    if "mode_at_junction_distribution" not in df.columns:
        df["mode_at_junction_distribution"] = "{}"

    return df.sort_values(["threat_idx", "reward_idx"]).reset_index(drop=True)


def parse_mode_distribution(value: Any) -> Dict[str, float]:
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}

    if pd.isna(value):
        return {}

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw.replace("'", '"'))
            if isinstance(parsed, dict):
                return {str(k): float(v) for k, v in parsed.items()}
        except Exception:
            return {}

    return {}


def add_mode_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    explore_rates: List[float] = []
    exploit_rates: List[float] = []
    exploit_safe_rates: List[float] = []
    absence_check_rates: List[float] = []
    dominant_modes: List[str] = []

    for raw in df["mode_at_junction_distribution"]:
        dist = parse_mode_distribution(raw)
        explore_rates.append(float(dist.get("explore", 0.0)))
        exploit_rates.append(float(dist.get("exploit", 0.0)))
        exploit_safe_rates.append(float(dist.get("exploit_safe", 0.0)))
        absence_check_rates.append(float(dist.get("absence_check", 0.0)))
        dominant_modes.append(max(dist, key=dist.get) if dist else "")

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
    *,
    title: str,
    output_path: Path,
    fmt: str,
    cmap: str,
) -> None:
    values = pivot_df.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    im = ax.imshow(values, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels(list(pivot_df.columns))
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(list(pivot_df.index))
    ax.set_xlabel("Reward level")
    ax.set_ylabel("Threat level")
    ax.set_title(title)

    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    midpoint = (vmin + vmax) / 2.0 if not math.isclose(vmin, vmax) else vmin

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            color = "white" if val <= midpoint else "black"
            ax.text(
                j,
                i,
                format(float(val), fmt),
                ha="center",
                va="center",
                color=color,
                fontsize=10,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Value", rotation=270, labelpad=12)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"✓ Figure saved: {output_path}")


def draw_mode_label_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    pivot = df.pivot(index="threat_idx", columns="reward_idx", values="dominant_mode")
    pivot = pivot.sort_index().sort_index(axis=1)
    pivot.index = [f"T{idx}" for idx in pivot.index]
    pivot.columns = [f"R{idx}" for idx in pivot.columns]

    mode_to_num = {
        "exploit": 0,
        "explore": 1,
        "exploit_safe": 2,
        "absence_check": 3,
        "": -1,
    }

    num = (
        pivot.apply(lambda col: col.map(lambda x: mode_to_num.get(str(x), -1)))
        .to_numpy(dtype=float)
    )

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.imshow(num, cmap="Pastel1", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    ax.set_xlabel("Reward level")
    ax.set_ylabel("Threat level")
    ax.set_title("Stage 3.1B matrix: dominant mode at junction")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(
                j,
                i,
                str(pivot.iloc[i, j]),
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"✓ Figure saved: {output_path}")


def compute_seed_cell_metrics(trials_path: Optional[Path]) -> pd.DataFrame:
    if trials_path is None or not trials_path.exists():
        return pd.DataFrame(
            [
                {
                    "note": "No all_trials CSV available; seed-level matrix metrics not computed.",
                }
            ]
        )

    trials = pd.read_csv(trials_path)
    required = {"seed", "condition_id", "path_choice"}
    missing = required - set(trials.columns)
    if missing:
        return pd.DataFrame(
            [
                {
                    "note": f"Missing required columns for seed-level metrics: {sorted(missing)}",
                }
            ]
        )

    trials = trials[trials["condition_id"].astype(str).apply(is_matrix_condition)].copy()

    rows: List[Dict[str, Any]] = []
    for (seed, condition_id), sdf in trials.groupby(["seed", "condition_id"], dropna=False):
        r, t = parse_condition_id(str(condition_id))

        row: Dict[str, Any] = {
            "seed": int(seed),
            "condition_id": str(condition_id),
            "reward_idx": int(r),
            "threat_idx": int(t),
            "n_trials": int(len(sdf)),
            "p_open": float((sdf["path_choice"].astype(str) == "open").mean()),
            "p_covered": float((sdf["path_choice"].astype(str) == "covered").mean()),
        }

        if "commit_reason" in sdf.columns:
            row["p_commit_timeout"] = float(
                (sdf["commit_reason"].astype(str) == "timeout").mean()
            )

        for col, out_col in [
            ("commit_latency", "mean_commit_latency"),
            ("junction_pause_duration", "mean_junction_pause_duration"),
            ("reorientation_count", "mean_reorientation_count"),
            ("junction_deliberation_proxy", "mean_junction_deliberation_proxy"),
        ]:
            if col in sdf.columns:
                row[out_col] = float(pd.to_numeric(sdf[col], errors="coerce").mean())

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["threat_idx", "reward_idx", "seed"])
        .reset_index(drop=True)
    )


def compute_acceptance_checks(df: pd.DataFrame) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}

    reward_monotonic: Dict[str, bool] = {}
    for t in sorted(df["threat_idx"].unique()):
        sub = df[df["threat_idx"] == t].sort_values("reward_idx")
        vals = pd.to_numeric(sub["p_open"], errors="coerce").tolist()
        reward_monotonic[f"T{t}"] = bool(vals[0] <= vals[1] <= vals[2])

    threat_monotonic: Dict[str, bool] = {}
    for r in sorted(df["reward_idx"].unique()):
        sub = df[df["reward_idx"] == r].sort_values("threat_idx")
        vals = pd.to_numeric(sub["p_open"], errors="coerce").tolist()
        threat_monotonic[f"R{r}"] = bool(vals[0] >= vals[1] >= vals[2])

    checks["reward_monotonicity_per_threat"] = reward_monotonic
    checks["threat_monotonicity_per_reward"] = threat_monotonic
    checks["reward_axis_order_ok"] = bool(all(reward_monotonic.values()))
    checks["threat_axis_order_ok"] = bool(all(threat_monotonic.values()))

    center = df[df["condition_id"].astype(str) == "R1_T2"]
    if len(center) == 1:
        center_row = center.iloc[0]
        for col, check_name in [
            ("mean_junction_pause_duration", "center_pause_peak"),
            ("mean_commit_latency", "center_latency_peak"),
            ("mean_reorientation_count", "center_reorientation_peak"),
            ("mean_junction_deliberation_proxy", "center_proxy_peak"),
        ]:
            if col in df.columns:
                checks[check_name] = bool(
                    float(center_row[col]) >= float(pd.to_numeric(df[col], errors="coerce").max())
                )
            else:
                checks[check_name] = False

        checks["balanced_conflict_peak_ok"] = bool(
            checks.get("center_pause_peak", False)
            and checks.get("center_latency_peak", False)
            and checks.get("center_reorientation_peak", False)
            and checks.get("center_proxy_peak", False)
        )
    else:
        checks["balanced_conflict_peak_ok"] = False

    checks["n_matrix_cells"] = int(len(df))
    checks["matrix_complete_3x3"] = bool(len(df) == 9)

    return checks


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✓ Table saved: {path}")


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✓ Stats saved: {path}")


def save_report(
    df: pd.DataFrame,
    checks: Dict[str, Any],
    *,
    source_path: Path,
    trials_path: Optional[Path],
    output_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Stage 3.1B Matrix Publication Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "This report covers the 3x3 reward x threat matrix layer of Stage 3.1B."
    )
    lines.append("It is separate from one-shot shock/treat event-aligned analyses.")
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    lines.append(f"- Condition summary: `{source_path}`")
    lines.append(f"- Trial table: `{trials_path}`")
    lines.append("")
    lines.append("## Matrix cell summary")
    lines.append("")
    lines.append("```")
    lines.append(df.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Acceptance checks")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(checks, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Interpretation boundary")
    lines.append("")
    lines.append(
        "Matrix heatmaps describe the conflict surface across reward and threat levels. "
        "They do not by themselves demonstrate one-shot carryover; that claim belongs "
        "to the one-shot publication tables and figures."
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ Report saved: {output_path}")


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    source_path = resolve_condition_summary(input_dir, args.ablation)
    trials_path = resolve_trials_csv(input_dir, args.ablation)

    print(f"Using matrix condition source: {source_path}")
    print(f"Using matrix trial source: {trials_path}")

    raw_df = pd.read_csv(source_path)
    cell_df = add_mode_columns(normalize_matrix_df(raw_df))
    seed_cell_df = compute_seed_cell_metrics(trials_path)
    checks = compute_acceptance_checks(cell_df)

    save_table(cell_df, output_dir / "Table_3_1B_matrix_cell_summary.csv")
    save_table(seed_cell_df, output_dir / "Table_3_1B_matrix_seed_cell_metrics.csv")

    for value_col, title, filename, fmt, cmap in MATRIX_HEATMAPS:
        if value_col not in cell_df.columns:
            print(f"WARNING: skipped {filename}; missing column {value_col!r}")
            continue
        draw_numeric_heatmap(
            build_pivot(cell_df, value_col),
            title=title,
            output_path=output_dir / filename,
            fmt=fmt,
            cmap=cmap,
        )

    if "explore_rate" in cell_df.columns:
        draw_numeric_heatmap(
            build_pivot(cell_df, "explore_rate"),
            title="Stage 3.1B matrix: mode at junction = explore",
            output_path=output_dir / "Figure_3_1B_Matrix_Mode_Explore_Rate_Heatmap.png",
            fmt=".3f",
            cmap="Blues",
        )

    if "exploit_rate" in cell_df.columns:
        draw_numeric_heatmap(
            build_pivot(cell_df, "exploit_rate"),
            title="Stage 3.1B matrix: mode at junction = exploit",
            output_path=output_dir / "Figure_3_1B_Matrix_Mode_Exploit_Rate_Heatmap.png",
            fmt=".3f",
            cmap="Oranges",
        )

    draw_mode_label_heatmap(
        cell_df,
        output_dir / "Figure_3_1B_Matrix_Mode_Dominant_Label.png",
    )

    save_json(checks, output_dir / "stage3_1b_matrix_acceptance_check.json")
    save_report(
        cell_df,
        checks,
        source_path=source_path,
        trials_path=trials_path,
        output_path=output_dir / "Stage3_1B_Matrix_Publication_Report.md",
    )

    metadata = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "ablation": args.ablation,
        "condition_source": str(source_path),
        "trials_source": str(trials_path) if trials_path else None,
        "n_cells": int(len(cell_df)),
        "n_seed_cell_rows": int(len(seed_cell_df)),
        "acceptance": checks,
    }
    save_json(metadata, output_dir / "stage3_1b_matrix_publication_analysis_meta.json")


if __name__ == "__main__":
    main()