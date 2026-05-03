"""Stage 3.2 VTE analysis.

Reads trial-level VTE wrapper output and writes publication-facing summaries.
This module does not import stage3 internals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

def summarize_distribution_by_path(df: pd.DataFrame) -> pd.DataFrame:
    if "committed_path" not in df.columns:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []

    for path, g in df.groupby("committed_path", dropna=False, sort=True):
        raw = pd.to_numeric(g["raw_idphi"], errors="coerce")
        z = pd.to_numeric(g["z_idphi"], errors="coerce")

        rows.append(
            {
                "committed_path": path,
                "n_trials": int(len(g)),
                "vte_rate": float(pd.to_numeric(g["vte_binary"], errors="coerce").mean()),
                "raw_idphi_q10": float(raw.quantile(0.10)),
                "raw_idphi_q25": float(raw.quantile(0.25)),
                "raw_idphi_median": float(raw.quantile(0.50)),
                "raw_idphi_q75": float(raw.quantile(0.75)),
                "raw_idphi_q90": float(raw.quantile(0.90)),
                "z_idphi_q10": float(z.quantile(0.10)),
                "z_idphi_q25": float(z.quantile(0.25)),
                "z_idphi_median": float(z.quantile(0.50)),
                "z_idphi_q75": float(z.quantile(0.75)),
                "z_idphi_q90": float(z.quantile(0.90)),
            }
        )

    return pd.DataFrame(rows)


def _save_vte_rate_by_path(df: pd.DataFrame, output_dir: Path) -> str | None:
    if "committed_path" not in df.columns:
        return None

    summary = summarize_by(df, ["committed_path"])
    summary = summary.sort_values("committed_path")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(summary["committed_path"].astype(str), summary["vte_rate"])
    ax.set_title("Stage 3.2 VTE rate by committed path")
    ax.set_xlabel("Committed path")
    ax.set_ylabel("VTE rate")
    ax.set_ylim(0, max(0.05, float(summary["vte_rate"].max()) * 1.25))
    fig.tight_layout()

    filename = "Figure_3_2_VTE_Rate_By_Path.png"
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)
    return filename


def _save_idphi_by_path_boxplot(df: pd.DataFrame, output_dir: Path) -> str | None:
    if "committed_path" not in df.columns:
        return None

    paths = sorted(str(x) for x in df["committed_path"].dropna().unique())
    data = [
        pd.to_numeric(df.loc[df["committed_path"].astype(str) == path, "raw_idphi"], errors="coerce").dropna()
        for path in paths
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, labels=paths, showfliers=False)
    ax.set_title("Stage 3.2 raw IdPhi distribution by committed path")
    ax.set_xlabel("Committed path")
    ax.set_ylabel("Raw IdPhi")
    fig.tight_layout()

    filename = "Figure_3_2_IdPhi_By_Path_Boxplot.png"
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)
    return filename


def _save_vte_rate_by_seed(df: pd.DataFrame, output_dir: Path) -> str:
    seed_summary = summarize_by(df, ["seed"]).sort_values("seed")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(seed_summary["seed"], seed_summary["vte_rate"], marker="o", linewidth=1)
    ax.set_title("Stage 3.2 VTE rate by seed")
    ax.set_xlabel("Seed")
    ax.set_ylabel("VTE rate")
    ax.set_ylim(0, max(0.05, float(seed_summary["vte_rate"].max()) * 1.25))
    fig.tight_layout()

    filename = "Figure_3_2_VTE_Rate_By_Seed.png"
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)
    return filename


def _save_idphi_vs_pause(df: pd.DataFrame, output_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        pd.to_numeric(df["pause_ticks"], errors="coerce"),
        pd.to_numeric(df["raw_idphi"], errors="coerce"),
        s=8,
        alpha=0.35,
    )
    ax.set_title("Stage 3.2 raw IdPhi vs pause ticks")
    ax.set_xlabel("Pause ticks")
    ax.set_ylabel("Raw IdPhi")
    fig.tight_layout()

    filename = "Figure_3_2_IdPhi_vs_Pause.png"
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)
    return filename


def write_figures(df: pd.DataFrame, output_dir: Path) -> list[str]:
    figures: list[str] = []

    for maybe_name in (
        _save_vte_rate_by_path(df, output_dir),
        _save_idphi_by_path_boxplot(df, output_dir),
        _save_vte_rate_by_seed(df, output_dir),
        _save_idphi_vs_pause(df, output_dir),
    ):
        if maybe_name is not None:
            figures.append(maybe_name)
            print(f"✓ Figure saved: {output_dir / maybe_name}")

    return figures

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