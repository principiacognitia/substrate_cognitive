"""Stage 3.2 VTE analysis.

Reads trial-level VTE wrapper output and writes publication-facing summaries.
This module does not import stage3 internals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_VTE_METRIC_COLUMNS = (
    "run_id",
    "seed",
    "trial",
    "raw_idphi",
    "log_idphi",
    "z_idphi",
    "pause_ticks",
    "reorientation_count",
    "vte_binary",
)

OPTIONAL_GROUP_COLUMNS = (
    "protocol",
    "condition",
    "ablation",
    "pose_source",
    "committed_path",
)


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required VTE metric columns: {missing}")


def _summary_row(df: pd.DataFrame) -> dict[str, object]:
    return {
        "n_trials": int(len(df)),
        "n_seeds": int(df["seed"].nunique()) if "seed" in df.columns else None,
        "vte_rate": float(pd.to_numeric(df["vte_binary"], errors="coerce").mean()),
        "raw_idphi_mean": float(pd.to_numeric(df["raw_idphi"], errors="coerce").mean()),
        "raw_idphi_sd": float(pd.to_numeric(df["raw_idphi"], errors="coerce").std()),
        "z_idphi_mean": float(pd.to_numeric(df["z_idphi"], errors="coerce").mean()),
        "z_idphi_sd": float(pd.to_numeric(df["z_idphi"], errors="coerce").std()),
        "pause_ticks_mean": float(pd.to_numeric(df["pause_ticks"], errors="coerce").mean()),
        "reorientation_count_mean": float(
            pd.to_numeric(df["reorientation_count"], errors="coerce").mean()
        ),
    }


def summarize_overall(df: pd.DataFrame) -> pd.DataFrame:
    row = _summary_row(df)
    row["group"] = "overall"
    return pd.DataFrame([row])


def summarize_by(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if not group_cols:
        return summarize_overall(df)

    rows: list[dict[str, object]] = []

    for key, g in df.groupby(group_cols, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_tuple))
        row.update(_summary_row(g))
        rows.append(row)

    return pd.DataFrame(rows)


def analyze_vte_metrics(metrics_csv: str | Path, output_dir: str | Path) -> dict[str, object]:
    metrics_path = Path(metrics_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not metrics_path.exists():
        raise FileNotFoundError(f"VTE metrics file not found: {metrics_path}")

    df = pd.read_csv(metrics_path)
    _require_columns(df, REQUIRED_VTE_METRIC_COLUMNS)

    # Normalize binary column after CSV roundtrip.
    df["vte_binary"] = pd.to_numeric(df["vte_binary"], errors="coerce").fillna(0).astype(int)

    tables: dict[str, pd.DataFrame] = {
        "Table_3_2_VTE_Overall_Summary.csv": summarize_overall(df),
        "Table_3_2_VTE_By_Seed.csv": summarize_by(df, ["run_id", "seed"]),
    }

    if "condition" in df.columns:
        tables["Table_3_2_VTE_By_Condition.csv"] = summarize_by(
            df,
            ["condition"],
        )

    if "committed_path" in df.columns:
        tables["Table_3_2_VTE_By_Committed_Path.csv"] = summarize_by(
            df,
            ["committed_path"],
        )

    if "condition" in df.columns and "committed_path" in df.columns:
        tables["Table_3_2_VTE_By_Condition_x_Path.csv"] = summarize_by(
            df,
            ["condition", "committed_path"],
        )

    written_tables: list[str] = []
    for filename, table in tables.items():
        path = output_path / filename
        table.to_csv(path, index=False)
        written_tables.append(filename)
        print(f"✓ Table saved: {path}")

    meta = {
        "input_metrics_csv": str(metrics_path),
        "output_dir": str(output_path),
        "n_trials": int(len(df)),
        "n_seeds": int(df["seed"].nunique()),
        "run_ids": sorted(str(x) for x in df["run_id"].dropna().unique()),
        "available_group_columns": [
            col for col in OPTIONAL_GROUP_COLUMNS if col in df.columns
        ],
        "tables": written_tables,
    }

    meta_path = output_path / "stage3_2_vte_analysis_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"✓ Metadata saved: {meta_path}")

    report_path = output_path / "Stage3_2_VTE_Analysis_Report.md"
    report_path.write_text(_build_report(meta, tables), encoding="utf-8")
    print(f"✓ Report saved: {report_path}")

    return meta


def _build_report(meta: dict[str, object], tables: dict[str, pd.DataFrame]) -> str:
    overall = tables["Table_3_2_VTE_Overall_Summary.csv"].iloc[0]

    lines = [
        "# Stage 3.2 VTE Analysis Report",
        "",
        "## Scope",
        "",
        "This report summarizes trial-level VTE wrapper output.",
        "The analysis operates on externalized VTE metrics and does not import Stage 3 internals.",
        "",
        "## Input",
        "",
        f"- Metrics CSV: `{meta['input_metrics_csv']}`",
        f"- Trials: {meta['n_trials']}",
        f"- Seeds: {meta['n_seeds']}",
        f"- Run IDs: {', '.join(meta['run_ids'])}",
        "",
        "## Overall summary",
        "",
        f"- VTE rate: {overall['vte_rate']:.4f}",
        f"- Mean raw IdPhi: {overall['raw_idphi_mean']:.4f}",
        f"- Mean z IdPhi: {overall['z_idphi_mean']:.4f}",
        f"- Mean pause ticks: {overall['pause_ticks_mean']:.4f}",
        f"- Mean reorientation count: {overall['reorientation_count_mean']:.4f}",
        "",
        "## Tables",
        "",
    ]

    for table_name in meta["tables"]:
        lines.append(f"- `{table_name}`")

    lines.extend(
        [
            "",
            "## Interpretation note",
            "",
            "For Stage 3 step-log derived traces, pose is synthetic. The VTE-like IdPhi metric measures open/covered candidate switching in a pose-like schema, not biological head-sweep kinematics.",
            "",
        ]
    )

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Stage 3.2 VTE metrics.")
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    analyze_vte_metrics(args.metrics_csv, args.output_dir)


if __name__ == "__main__":
    main()