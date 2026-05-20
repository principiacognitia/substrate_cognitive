from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_20B_FILES = {
    "seed_aggregates": "Table_3_2_Seed_Level_Aggregates.csv",
    "vte_contrasts": "Table_3_2_Seed_Level_VTE_Contrasts.csv",
    "ablation_contrasts": "Table_3_2_Seed_Level_Ablation_Contrasts.csv",
    "tests": "Table_3_2_Seed_Level_Statistical_Tests.csv",
    "effects": "Table_3_2_Seed_Level_Effect_Sizes.csv",
}

CONTEXT_COLUMNS = [
    "source",
    "dataset_id",
    "task_family",
    "protocol",
    "condition",
    "ablation",
    "run_id",
    "reference_ablation",
    "candidate_ablation",
]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_20b(stats_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        key: _load_optional_csv(stats_dir / filename)
        for key, filename in REQUIRED_20B_FILES.items()
    }


def _as_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _valid_value(value: object) -> bool:
    if value is None:
        return False
    text = str(value)
    return text not in {"", "nan", "NaN", "None"}


def _short(value: object, max_len: int = 28) -> str:
    text = str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _context_columns_present(df: pd.DataFrame) -> list[str]:
    return [c for c in CONTEXT_COLUMNS if c in df.columns]


def _row_context_label(row: pd.Series, max_parts: int = 4) -> str:
    parts: list[str] = []

    for col in ["protocol", "condition", "ablation", "run_id"]:
        if col in row.index and _valid_value(row.get(col)):
            parts.append(f"{col}={_short(row.get(col), 22)}")

    for col in ["reference_ablation", "candidate_ablation"]:
        if col in row.index and _valid_value(row.get(col)):
            parts.append(f"{col}={_short(row.get(col), 22)}")

    return " | ".join(parts[:max_parts])

def _compact_run_id(value: object, max_len: int = 26) -> str:
    if not _valid_value(value):
        return "run_id=NA"

    text = str(value)

    replacements = [
        ("_one_shot_full_all_steps", ""),
        ("_one_shot_novg_all_steps", ""),
        ("_one_shot_novp_all_steps", ""),
        ("_one_shot_nox_all_steps", ""),
        ("_one_shot_off_all_steps", ""),
        ("_balanced_conflict_full_all_steps", ""),
        ("_balanced_conflict_novg_all_steps", ""),
        ("_balanced_conflict_novp_all_steps", ""),
        ("_balanced_conflict_nox_all_steps", ""),
        ("_balanced_conflict_one_shot_off_all_steps", ""),
        ("_all_steps", ""),
    ]

    for old, new in replacements:
        text = text.replace(old, new)

    return _short(text, max_len)


def _common_context_note(df: pd.DataFrame, columns: list[str]) -> str:
    parts: list[str] = []

    for col in columns:
        if col not in df.columns:
            continue
        values = [str(x) for x in df[col].dropna().unique()]
        values = [x for x in values if x not in {"", "nan", "None"}]
        if len(values) == 1:
            parts.append(f"{col}={values[0]}")
        elif len(values) > 1:
            parts.append(f"{col}: mixed")

    return "\n".join(parts)


def _clean_metric_name(value: object) -> str:
    text = str(value)
    return text.replace("_mean", "")

def _fmt_num(value: object, digits: int = 6) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(x):
        return str(x)
    return f"{x:.{digits}g}"


def _prepare_tests(tests: pd.DataFrame) -> pd.DataFrame:
    if tests.empty:
        return pd.DataFrame()

    df = tests.copy()
    for col in ["p_fdr_bh", "p_raw", "effect_size", "mean_delta", "ci95_low", "ci95_high"]:
        if col in df.columns:
            df[col] = _as_number(df[col])

    if "status" in df.columns:
        df = df.loc[df["status"] == "tested"].copy()

    if df.empty:
        return pd.DataFrame()

    df["_abs_effect"] = df["effect_size"].abs() if "effect_size" in df.columns else np.nan
    return df


def _top_findings(tests: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    df = _prepare_tests(tests)
    if df.empty:
        return pd.DataFrame()

    context_cols = _context_columns_present(df)

    cols = [
        "test_family",
        "contrast",
        *context_cols,
        "metric",
        "group_a",
        "group_b",
        "n_seed_pairs",
        "mean_delta",
        "effect_size",
        "effect_size_type",
        "ci95_low",
        "ci95_high",
        "p_raw",
        "p_fdr_bh",
        "direction",
    ]

    out = (
        df.sort_values(["p_fdr_bh", "_abs_effect"], ascending=[True, False])
        .head(top_n)
        [[c for c in cols if c in df.columns]]
        .copy()
    )

    if context_cols:
        out.insert(
            min(2 + len(context_cols), len(out.columns)),
            "context_label",
            out.apply(_row_context_label, axis=1),
        )

    return out


def _family_summary(tests: pd.DataFrame) -> pd.DataFrame:
    if tests.empty or "test_family" not in tests.columns:
        return pd.DataFrame()

    df = tests.copy()
    df["p_fdr_bh"] = _as_number(df.get("p_fdr_bh"))

    rows = []
    for family, g in df.groupby("test_family", dropna=False, sort=True):
        status = g.get("status", pd.Series(dtype=str))
        rows.append(
            {
                "test_family": family,
                "n_tests": int(len(g)),
                "n_tested": int((status == "tested").sum()),
                "n_fdr_lt_05": int((g["p_fdr_bh"] < 0.05).sum()),
                "n_fdr_lt_10": int((g["p_fdr_bh"] < 0.10).sum()),
                "min_p_fdr_bh": float(g["p_fdr_bh"].min())
                if g["p_fdr_bh"].notna().any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _metric_direction_summary(tests: pd.DataFrame) -> pd.DataFrame:
    df = _prepare_tests(tests)
    if df.empty:
        return pd.DataFrame()

    rows = []
    for key, g in df.groupby(["test_family", "metric"], dropna=False, sort=True):
        test_family, metric = key
        rows.append(
            {
                "test_family": test_family,
                "metric": metric,
                "n_tests": int(len(g)),
                "n_positive": int((g.get("direction") == "positive").sum())
                if "direction" in g.columns
                else 0,
                "n_negative": int((g.get("direction") == "negative").sum())
                if "direction" in g.columns
                else 0,
                "mean_effect_size": float(_as_number(g["effect_size"]).mean()),
                "median_effect_size": float(_as_number(g["effect_size"]).median()),
                "min_p_fdr_bh": float(_as_number(g["p_fdr_bh"]).min()),
            }
        )
    return pd.DataFrame(rows)


def _save_horizontal_effect_plot(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    filename: str,
    title: str,
    xlabel: str,
    label_fn: Callable[[pd.Series], str],
    value_col: str,
    bar_label_fn: Callable[[pd.Series], str] | None = None,
    context_note: str | None = None,
) -> str | None:
    if df.empty:
        return None

    plot_df = df.copy()
    plot_df = plot_df.dropna(subset=[value_col])
    if plot_df.empty:
        return None

    labels = [label_fn(row) for _, row in plot_df.iterrows()]
    values = plot_df[value_col].to_numpy(dtype=float)

    height = max(4.8, 0.38 * len(plot_df) + 1.6)
    fig, ax = plt.subplots(figsize=(11.5, height))

    y = np.arange(len(plot_df))
    bars = ax.barh(y, values)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)

    finite_values = values[np.isfinite(values)]
    if finite_values.size:
        xmin = min(0.0, float(finite_values.min()))
        xmax = max(0.0, float(finite_values.max()))
        span = max(xmax - xmin, 1.0)
        ax.set_xlim(xmin - 0.10 * span, xmax + 0.28 * span)

    if bar_label_fn is not None:
        xmin, xmax = ax.get_xlim()
        span = max(xmax - xmin, 1.0)

        for rect, (_, row), value in zip(bars, plot_df.iterrows(), values):
            label = bar_label_fn(row)
            if not _valid_value(label):
                continue

            width = float(rect.get_width())
            y_pos = rect.get_y() + rect.get_height() / 2

            if abs(width) > 0.12 * span:
                x_pos = width / 2
                ha = "center"
            elif width >= 0:
                x_pos = width + 0.015 * span
                ha = "left"
            else:
                x_pos = width - 0.015 * span
                ha = "right"

            ax.text(
                x_pos,
                y_pos,
                str(label),
                va="center",
                ha=ha,
                fontsize=7,
                clip_on=False,
            )

    if context_note:
        ax.text(
            0.99,
            0.01,
            context_note,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round", alpha=0.12),
        )

    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return filename


def _vte_label(row: pd.Series) -> str:
    metric = _clean_metric_name(row.get("metric", "metric"))
    ablation = row.get("ablation", "")

    if _valid_value(ablation):
        return f"{metric} | {ablation}"
    return metric


def _vte_bar_label(row: pd.Series) -> str:
    return _compact_run_id(row.get("run_id", ""))


def _ablation_label(row: pd.Series) -> str:
    metric = _clean_metric_name(row.get("metric", "metric"))
    return metric


def _ablation_bar_label(row: pd.Series) -> str:
    candidate = row.get("candidate_ablation", row.get("group_b", ""))
    return str(candidate) if _valid_value(candidate) else ""


def _save_vte_delta_effects(
    tests: pd.DataFrame,
    output_dir: Path,
    top_n: int = 25,
) -> str | None:
    df = _prepare_tests(tests)
    if df.empty:
        return None

    df = df.loc[df.get("test_family") == "vte_binary_within_seed"].copy()
    if df.empty:
        return None

    df = (
        df.sort_values(["p_fdr_bh", "_abs_effect"], ascending=[True, False])
        .head(top_n)
        .copy()
    )

    context_note = _common_context_note(df, ["protocol", "condition"])

    return _save_horizontal_effect_plot(
        df,
        output_dir,
        filename="Figure_3_2_Seed_Level_VTE_Delta_Effect_Sizes.png",
        title="Patch 20D VTE minus non-VTE seed-level effects",
        xlabel="Effect size, Cohen dz",
        label_fn=_vte_label,
        value_col="effect_size",
        bar_label_fn=_vte_bar_label,
        context_note=context_note,
    )


def _save_ablation_effects(
    tests: pd.DataFrame,
    output_dir: Path,
    top_n: int = 28,
) -> str | None:
    df = _prepare_tests(tests)
    if df.empty:
        return None

    df = df.loc[df.get("test_family") == "ablation_vs_reference_within_seed"].copy()
    if df.empty:
        return None

    df = (
        df.sort_values(["p_fdr_bh", "_abs_effect"], ascending=[True, False])
        .head(top_n)
        .copy()
    )

    df["signed_log_effect"] = (
        np.sign(df["effect_size"]) * np.log10(1.0 + df["effect_size"].abs())
    )

    context_note = _common_context_note(
        df,
        ["protocol", "condition", "reference_ablation"],
    )

    return _save_horizontal_effect_plot(
        df,
        output_dir,
        filename="Figure_3_2_Seed_Level_Ablation_Effect_Sizes.png",
        title="Patch 20D ablation minus reference seed-level effects",
        xlabel="Signed log10(1 + |Cohen dz|), sign preserved",
        label_fn=_ablation_label,
        value_col="signed_log_effect",
        bar_label_fn=_ablation_bar_label,
        context_note=context_note,
    )


def _save_vte_rate_by_ablation(
    seed_aggregates: pd.DataFrame,
    output_dir: Path,
) -> str | None:
    if (
        seed_aggregates.empty
        or "ablation" not in seed_aggregates.columns
        or "vte_rate" not in seed_aggregates.columns
    ):
        return None

    df = seed_aggregates.copy()
    df["vte_rate"] = _as_number(df["vte_rate"])
    df = df.dropna(subset=["vte_rate"])
    if df.empty:
        return None

    labels = sorted(str(x) for x in df["ablation"].dropna().unique())
    data = [
        df.loc[df["ablation"].astype(str) == label, "vte_rate"].dropna().to_numpy()
        for label in labels
    ]

    fig, ax = plt.subplots(figsize=(max(6.5, len(labels) * 1.2), 4.8))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_xlabel("Ablation")
    ax.set_ylabel("Seed-level VTE rate")
    ax.set_title("Seed-level VTE rate by ablation")
    fig.tight_layout()

    filename = "Figure_3_2_Seed_Level_VTE_Rate_By_Ablation.png"
    fig.savefig(output_dir / filename, dpi=220)
    plt.close(fig)
    return filename


def _report(meta: dict[str, object], top: pd.DataFrame, family: pd.DataFrame) -> str:
    lines = [
        "# Patch 20D Seed-Level Statistics Presentation Update",
        "",
        "Patch 20D reads Patch 20B outputs through the Patch 20C analyzer path.",
        "It changes presentation only: context columns, figure readability, and production output packaging.",
        "It does not recompute or alter Patch 20B statistical tests.",
        "",
        "## Inputs",
        "",
        f"- Stats directory: `{meta['stats_dir']}`",
        "",
        "## Outputs",
        "",
    ]

    for p in meta["outputs"].values():
        lines.append(f"- `{p}`")

    lines += ["", "## Figures", ""]
    for fig in meta.get("figures", []):
        lines.append(f"- `{fig}`")

    lines += ["", "## Test-family summary", ""]
    if family.empty:
        lines.append("No test-family summary available.")
    else:
        for _, r in family.iterrows():
            lines.append(
                f"- {r['test_family']}: "
                f"n_tests={r['n_tests']}, "
                f"n_tested={r['n_tested']}, "
                f"fdr<0.05={r['n_fdr_lt_05']}"
            )

    lines += ["", "## Top findings with context", ""]
    if top.empty:
        lines.append("No tested findings available.")
    else:
        for _, r in top.head(12).iterrows():
            context = r.get("context_label", "")
            context_part = f" | {context}" if _valid_value(context) else ""
            lines.append(
                f"- {r.get('test_family')} | {r.get('contrast')}{context_part} | "
                f"{r.get('metric')}: "
                f"effect={_fmt_num(r.get('effect_size'))}, "
                f"p_fdr={_fmt_num(r.get('p_fdr_bh'))}"
            )

    lines += [
        "",
        "## Interpretation boundary",
        "",
        "Patch 20D is a presentation-layer patch. It should not be cited as a new inferential analysis.",
        "Wrapper sanity effects and model-relevant effects remain to be separated in Patch 20E.",
        "",
    ]
    return "\n".join(lines)


def analyze_seed_level_stats(
    stats_dir: str | Path,
    output_dir: str | Path,
    top_n: int = 25,
) -> dict[str, object]:
    stats_path = Path(stats_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tables = _load_20b(stats_path)
    tests = tables["tests"]
    seed_aggregates = tables["seed_aggregates"]

    top = _top_findings(tests, top_n)
    family = _family_summary(tests)
    direction = _metric_direction_summary(tests)

    outputs = {
        "top_findings": out / "Table_3_2_Seed_Level_Stats_Top_Findings.csv",
        "test_family_summary": out / "Table_3_2_Seed_Level_Stats_By_Test_Family.csv",
        "metric_direction_summary": out / "Table_3_2_Seed_Level_Stats_By_Metric_Direction.csv",
        "meta": out / "stage3_2_seed_level_stats_analysis_meta.json",
        "report": out / "Stage3_2_Seed_Level_Stats_Analysis_Report.md",
    }

    top.to_csv(outputs["top_findings"], index=False)
    family.to_csv(outputs["test_family_summary"], index=False)
    direction.to_csv(outputs["metric_direction_summary"], index=False)

    figures = [
        x
        for x in (
            _save_vte_delta_effects(tests, out, top_n),
            _save_ablation_effects(tests, out, max(top_n, 28)),
            _save_vte_rate_by_ablation(seed_aggregates, out),
        )
        if x
    ]

    meta = {
        "patch": "20D",
        "script": "vte.analysis.analyze_stage3_2_seed_level_stats",
        "created_at": _now_iso(),
        "stats_dir": str(stats_path),
        "output_dir": str(out),
        "n_tests": int(len(tests)),
        "n_top_findings": int(len(top)),
        "n_figures": int(len(figures)),
        "figures": figures,
        "presentation_changes": [
            "top findings retain available group context columns",
            "effect-size plots are horizontal",
            "ablation effect-size plot uses signed log10(1 + |Cohen dz|) display transform",
            "figure labels are compact; shared protocol/condition/reference context is moved into figure note boxes",
            "bar labels encode candidate ablation or run_id where useful",
            "statistical inputs from Patch 20B are not modified",
        ],
        "outputs": {
            k: str(v)
            for k, v in outputs.items()
            if k not in ("meta", "report")
        },
    }

    outputs["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    outputs["report"].write_text(_report(meta, top, family), encoding="utf-8")

    print(f"Top findings saved: {outputs['top_findings']}")
    print(f"Test-family summary saved: {outputs['test_family_summary']}")
    print(f"Metric-direction summary saved: {outputs['metric_direction_summary']}")
    for fig in figures:
        print(f"Figure saved: {out / fig}")
    print(f"Metadata saved: {outputs['meta']}")
    print(f"Report saved: {outputs['report']}")
    print(f"Tests: {meta['n_tests']}")
    print(f"Figures: {meta['n_figures']}")
    return meta


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Patch 20D: presentation update for Patch 20B/20C "
            "seed-level statistical outputs."
        )
    )
    p.add_argument("--stats-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--top-n", type=int, default=25)
    return p


def main() -> None:
    a = build_arg_parser().parse_args()
    analyze_seed_level_stats(a.stats_dir, a.output_dir, a.top_n)


if __name__ == "__main__":
    main()