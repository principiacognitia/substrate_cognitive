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

ROLE_TEST_TABLE_COLUMNS = [
    "test_family",
    "test_role",
    "role_note",
    "contrast",
    "source",
    "dataset_id",
    "task_family",
    "protocol",
    "condition",
    "ablation",
    "run_id",
    "reference_ablation",
    "candidate_ablation",
    "metric",
    "group_a",
    "group_b",
    "n_seed_pairs",
    "mean_delta",
    "median_delta",
    "sd_delta",
    "effect_size",
    "effect_size_type",
    "ci95_low",
    "ci95_high",
    "p_raw",
    "p_fdr_bh",
    "direction",
    "status",
]

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

WRAPPER_SANITY_METRICS = {
    "raw_idphi",
    "log_idphi",
    "z_idphi",
    "lab_idphi",
    "z_lab_idphi_by_session",
    "pause_ticks",
    "pause_time_s",
    "reorientation_count",
}

MODEL_RELEVANT_METRICS = {
    "reward",
    "total_reward",
    "cost",
    "native_cost_bin",
    "comparable_cost_bin",
    "vte_rate",
    "dwell_proxy",
    "dwell_z",
    "z_dwell_proxy_by_session",
    "deliberation_proxy",
    "deliberation_z",
    "z_pause_time_by_session",
}


def _row_candidate_ablation(row: pd.Series) -> str:
    for col in ("candidate_ablation", "group_b", "ablation"):
        if col in row.index and _valid_value(row.get(col)):
            return str(row.get(col)).lower()
    return ""


def _classify_test_role(row: pd.Series) -> str:
    family = str(row.get("test_family", ""))
    metric = _clean_metric_name(row.get("metric", ""))
    candidate = _row_candidate_ablation(row)

    if family == "ablation_vs_reference_within_seed" and candidate == "novg":
        return "degenerate_ablation_diagnostic"

    if family == "vte_binary_within_seed" and metric in WRAPPER_SANITY_METRICS:
        return "wrapper_sanity_check"

    if family == "vte_binary_within_seed" and metric in MODEL_RELEVANT_METRICS:
        return "model_relevant_test"

    if family == "ablation_vs_reference_within_seed":
        return "model_relevant_test"

    return "diagnostic"


def _role_note(row: pd.Series) -> str:
    role = _classify_test_role(row)
    metric = _clean_metric_name(row.get("metric", ""))

    if role == "wrapper_sanity_check":
        return (
            "Expected VTE-label separation on metrics used by or adjacent to "
            "the VTE measurement definition; not independent model validation."
        )

    if role == "model_relevant_test":
        return (
            "Seed-level behavioral or ablation contrast relevant to model-level "
            "interpretation."
        )

    if role == "degenerate_ablation_diagnostic":
        return (
            "Diagnostic of architectural collapse or extreme regime shift under "
            "novg; should not be treated as a clean localized effect."
        )

    return f"Diagnostic row for metric={metric}."

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

    df["test_role"] = df.apply(_classify_test_role, axis=1)
    df["role_note"] = df.apply(_role_note, axis=1)
    df["_abs_effect"] = df["effect_size"].abs() if "effect_size" in df.columns else np.nan
    return df


def _top_findings(tests: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    df = _prepare_tests(tests)
    if df.empty:
        return pd.DataFrame()

    context_cols = _context_columns_present(df)

    cols = [
        "test_family",
        "test_role",
        "role_note",
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

def _role_summary(prepared_tests: pd.DataFrame) -> pd.DataFrame:
    if prepared_tests.empty or "test_role" not in prepared_tests.columns:
        return pd.DataFrame()

    rows = []
    for role, g in prepared_tests.groupby("test_role", dropna=False, sort=True):
        p = pd.to_numeric(g.get("p_fdr_bh"), errors="coerce")
        effect = pd.to_numeric(g.get("effect_size"), errors="coerce")

        rows.append(
            {
                "test_role": role,
                "n_tests": int(len(g)),
                "n_fdr_lt_05": int((p < 0.05).sum()),
                "n_fdr_lt_10": int((p < 0.10).sum()),
                "min_p_fdr_bh": float(p.min()) if p.notna().any() else np.nan,
                "median_abs_effect_size": float(effect.abs().median())
                if effect.notna().any()
                else np.nan,
                "max_abs_effect_size": float(effect.abs().max())
                if effect.notna().any()
                else np.nan,
            }
        )

    return pd.DataFrame(rows)


def _tests_for_role(prepared_tests: pd.DataFrame, role: str) -> pd.DataFrame:
    if prepared_tests.empty or "test_role" not in prepared_tests.columns:
        return pd.DataFrame(columns=ROLE_TEST_TABLE_COLUMNS)

    df = prepared_tests.loc[prepared_tests["test_role"] == role].copy()
    if df.empty:
        return pd.DataFrame(columns=ROLE_TEST_TABLE_COLUMNS)

    df = df.sort_values(["p_fdr_bh", "_abs_effect"], ascending=[True, False])

    for col in ROLE_TEST_TABLE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    return df[ROLE_TEST_TABLE_COLUMNS]

def _save_compact_markdown_table(
    df: pd.DataFrame,
    path: Path,
    *,
    title: str,
    max_rows: int = 25,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [f"# {title}", ""]

    if df.empty:
        lines.append("No rows.")
        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    view = df.copy().head(max_rows)

    keep = [
        "test_role",
        "test_family",
        "contrast",
        "protocol",
        "condition",
        "ablation",
        "candidate_ablation",
        "metric",
        "group_a",
        "group_b",
        "n_seed_pairs",
        "mean_delta",
        "effect_size",
        "ci95_low",
        "ci95_high",
        "p_fdr_bh",
        "direction",
        "status",
    ]
    keep = [c for c in keep if c in view.columns]
    view = view[keep].copy()

    for col in ["mean_delta", "effect_size", "ci95_low", "ci95_high", "p_fdr_bh"]:
        if col in view.columns:
            view[col] = pd.to_numeric(view[col], errors="coerce").map(
                lambda x: "" if pd.isna(x) else f"{x:.6g}"
            )

    lines.append(view.to_markdown(index=False))
    lines.append("")
    lines.append(f"Rows shown: {len(view)} of {len(df)}.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

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
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[value_col])
    plot_df = plot_df.loc[np.isfinite(plot_df[value_col])].copy()
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

def _role_effect_label(row: pd.Series) -> str:
    metric = _clean_metric_name(row.get("metric", "metric"))
    family = str(row.get("test_family", ""))

    if family == "ablation_vs_reference_within_seed":
        candidate = row.get("candidate_ablation", row.get("group_b", "candidate"))
        reference = row.get("reference_ablation", row.get("group_a", "reference"))
        return f"{candidate} − {reference}: {metric}"

    ablation = row.get("ablation", "")
    if _valid_value(ablation):
        return f"{metric} | {ablation}"

    return metric


def _role_effect_bar_label(row: pd.Series) -> str:
    p = row.get("p_fdr_bh", np.nan)
    try:
        p_text = f"q={float(p):.2g}"
    except (TypeError, ValueError):
        p_text = "q=NA"

    direction = row.get("direction", "")
    if _valid_value(direction):
        return f"{p_text}, {direction}"
    return p_text


def _save_model_relevant_effects(
    model_tests: pd.DataFrame,
    output_dir: Path,
    top_n: int = 25,
) -> str | None:
    if model_tests.empty:
        return None

    df = model_tests.copy()
    df["effect_size"] = pd.to_numeric(df["effect_size"], errors="coerce")
    df["_abs_effect"] = df["effect_size"].abs()
    df = df.dropna(subset=["effect_size"])
    df = df.loc[np.isfinite(df["effect_size"])].copy()
    if df.empty:
        return None

    df = df.sort_values(["p_fdr_bh", "_abs_effect"], ascending=[True, False]).head(top_n)

    context_note = _common_context_note(df, ["protocol", "condition"])

    return _save_horizontal_effect_plot(
        df,
        output_dir,
        filename="Figure_3_2_Model_Relevant_Seed_Level_Effects.png",
        title="Patch 20E model-relevant seed-level effects",
        xlabel="Effect size, Cohen dz",
        label_fn=_role_effect_label,
        value_col="effect_size",
        bar_label_fn=_role_effect_bar_label,
        context_note=context_note,
    )


def _save_degenerate_ablation_diagnostics(
    degenerate_tests: pd.DataFrame,
    output_dir: Path,
    top_n: int = 25,
) -> str | None:
    if degenerate_tests.empty:
        return None

    df = degenerate_tests.copy()
    df["effect_size"] = pd.to_numeric(df["effect_size"], errors="coerce")
    df["_abs_effect"] = df["effect_size"].abs()
    df = df.dropna(subset=["effect_size"])
    df = df.loc[np.isfinite(df["effect_size"])].copy()
    if df.empty:
        return None

    df = df.sort_values("_abs_effect", ascending=False).head(top_n)
    df["signed_log_effect"] = (
        np.sign(df["effect_size"]) * np.log10(1.0 + df["effect_size"].abs())
    )

    context_note = _common_context_note(
        df,
        ["protocol", "condition", "reference_ablation", "candidate_ablation"],
    )

    return _save_horizontal_effect_plot(
        df,
        output_dir,
        filename="Figure_3_2_Degenerate_Ablation_Diagnostics.png",
        title="Patch 20E degenerate ablation diagnostics",
        xlabel="Signed log10(1 + |Cohen dz|), sign preserved",
        label_fn=_role_effect_label,
        value_col="signed_log_effect",
        bar_label_fn=_role_effect_bar_label,
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

def _glm_response_report(
    meta: dict[str, object],
    role_summary: pd.DataFrame,
    model_tests: pd.DataFrame,
    wrapper_tests: pd.DataFrame,
    degenerate_tests: pd.DataFrame,
) -> str:
    def _n_sig(df: pd.DataFrame) -> int:
        if df.empty or "p_fdr_bh" not in df.columns:
            return 0
        return int((pd.to_numeric(df["p_fdr_bh"], errors="coerce") < 0.05).sum())

    lines = [
        "# Stage 3.2 Response to GLM Statistical Critique",
        "",
        "This note separates the Patch 20B seed-level statistics into interpretive roles.",
        "Patch 20E does not recompute the Patch 20B tests. It classifies their inferential role and regenerates presentation tables and figures.",
        "",
        "## Main distinction",
        "",
        "Patch 20B contains two different kinds of statistically significant effects:",
        "",
        "1. **Wrapper sanity checks**: expected separation between VTE and non-VTE rows on metrics that define or closely track the VTE measurement itself, such as IdPhi, z-IdPhi, pause, and reorientation.",
        "2. **Model-relevant tests**: behavioral or ablation contrasts that can support interpretation of the model beyond the mechanical definition of the VTE label.",
        "",
        "These categories must not be conflated.",
        "",
        "## Counts",
        "",
        f"- Total Patch 20B tests read: {meta.get('n_tests', 'NA')}",
        f"- Model-relevant tested rows: {len(model_tests)}",
        f"- Wrapper-sanity tested rows: {len(wrapper_tests)}",
        f"- Degenerate-ablation diagnostic rows: {len(degenerate_tests)}",
        f"- Model-relevant q<0.05 rows: {_n_sig(model_tests)}",
        f"- Wrapper-sanity q<0.05 rows: {_n_sig(wrapper_tests)}",
        f"- Degenerate-ablation q<0.05 rows: {_n_sig(degenerate_tests)}",
        "",
        "## Interpretation",
        "",
        "The largest VTE-minus-non-VTE effects on IdPhi-like metrics are expected and should be interpreted as wrapper sanity checks, not as independent validation of the cognitive model.",
        "",
        "The model-relevant evidence should instead be read from seed-level behavioral and ablation contrasts, especially contrasts that remain meaningful after circular VTE-definition metrics are separated.",
        "",
        "The `novg` ablation is treated separately as a degenerate-ablation diagnostic. Its large effects indicate architectural collapse or extreme regime shift, not a clean localized component effect.",
        "",
        "## Files produced by Patch 20E",
        "",
        "- `Table_3_2_Model_Relevant_Seed_Level_Tests.csv`",
        "- `Table_3_2_Wrapper_Sanity_Tests.csv`",
        "- `Table_3_2_Degenerate_Ablation_Diagnostics.csv`",
        "- `Table_3_2_Seed_Level_Stats_By_Test_Role.csv`",
        "- `Figure_3_2_Model_Relevant_Seed_Level_Effects.png`",
        "- `Figure_3_2_Degenerate_Ablation_Diagnostics.png`",
        "",
        "## Boundary",
        "",
        "Patch 20E is a classification and presentation layer over Patch 20B. It should be cited as a response to statistical interpretation concerns, not as a new experiment.",
        "",
    ]

    if not role_summary.empty:
        lines += ["## Role summary", ""]
        for _, r in role_summary.iterrows():
            lines.append(
                f"- {r.get('test_role')}: n={r.get('n_tests')}, "
                f"q<0.05={r.get('n_fdr_lt_05')}, "
                f"median |dz|={_fmt_num(r.get('median_abs_effect_size'))}"
            )
        lines.append("")

    return "\n".join(lines)

def _report(meta: dict[str, object], top: pd.DataFrame, family: pd.DataFrame) -> str:
    lines = [
        "# Patch 20E Seed-Level Statistics Presentation Update",
        "",
        "Patch 20E reads Patch 20B outputs through the Patch 20C analyzer path.",
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
        "Patch 20E is a presentation-layer patch. It should not be cited as a new inferential analysis.",
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

    prepared_tests = _prepare_tests(tests)

    top = _top_findings(tests, top_n)
    family = _family_summary(tests)
    direction = _metric_direction_summary(tests)

    role_summary = _role_summary(prepared_tests)
    model_relevant = _tests_for_role(prepared_tests, "model_relevant_test")
    wrapper_sanity = _tests_for_role(prepared_tests, "wrapper_sanity_check")
    degenerate = _tests_for_role(prepared_tests, "degenerate_ablation_diagnostic")
    diagnostic = _tests_for_role(prepared_tests, "diagnostic")

    outputs = {
        "top_findings": out / "Table_3_2_Seed_Level_Stats_Top_Findings.csv",
        "test_family_summary": out / "Table_3_2_Seed_Level_Stats_By_Test_Family.csv",
        "metric_direction_summary": out / "Table_3_2_Seed_Level_Stats_By_Metric_Direction.csv",
        "meta": out / "stage3_2_seed_level_stats_analysis_meta.json",
        "report": out / "Stage3_2_Seed_Level_Stats_Analysis_Report.md",
        "role_summary": out / "Table_3_2_Seed_Level_Stats_By_Test_Role.csv",
        "model_relevant_tests": out / "Table_3_2_Model_Relevant_Seed_Level_Tests.csv",
        "wrapper_sanity_tests": out / "Table_3_2_Wrapper_Sanity_Tests.csv",
        "degenerate_ablation_diagnostics": out / "Table_3_2_Degenerate_Ablation_Diagnostics.csv",
        "diagnostic_tests": out / "Table_3_2_Diagnostic_Seed_Level_Tests.csv",
        "glm_response": out / "Stage3_2_Response_To_GLM_Stats_Critique.md",
        "model_relevant_tests_md": out / "Table_3_2_Model_Relevant_Seed_Level_Tests.md",
        "wrapper_sanity_tests_md": out / "Table_3_2_Wrapper_Sanity_Tests.md",
        "degenerate_ablation_diagnostics_md": out / "Table_3_2_Degenerate_Ablation_Diagnostics.md",
    }

    top.to_csv(outputs["top_findings"], index=False)
    family.to_csv(outputs["test_family_summary"], index=False)
    direction.to_csv(outputs["metric_direction_summary"], index=False)
    role_summary.to_csv(outputs["role_summary"], index=False)
    model_relevant.to_csv(outputs["model_relevant_tests"], index=False)
    wrapper_sanity.to_csv(outputs["wrapper_sanity_tests"], index=False)
    degenerate.to_csv(outputs["degenerate_ablation_diagnostics"], index=False)
    diagnostic.to_csv(outputs["diagnostic_tests"], index=False)

    _save_compact_markdown_table(
        model_relevant,
        outputs["model_relevant_tests_md"],
        title="Table 3.2 Model-Relevant Seed-Level Tests",
    )
    _save_compact_markdown_table(
        wrapper_sanity,
        outputs["wrapper_sanity_tests_md"],
        title="Table 3.2 Wrapper-Sanity Tests",
    )
    _save_compact_markdown_table(
        degenerate,
        outputs["degenerate_ablation_diagnostics_md"],
        title="Table 3.2 Degenerate Ablation Diagnostics",
    )

    figures = [
        x
        for x in (
            _save_vte_delta_effects(tests, out, top_n),
            _save_ablation_effects(tests, out, max(top_n, 28)),
            _save_vte_rate_by_ablation(seed_aggregates, out),
            _save_model_relevant_effects(model_relevant, out, top_n),
            _save_degenerate_ablation_diagnostics(degenerate, out, top_n),
        )
        if x
    ]

    meta = {
        "patch": "20E",
        "script": "vte.analysis.analyze_stage3_2_seed_level_stats",
        "created_at": _now_iso(),
        "stats_dir": str(stats_path),
        "output_dir": str(out),
        "n_tests": int(len(tests)),
        "n_top_findings": int(len(top)),
        "n_figures": int(len(figures)),
        "figures": figures,
        "n_model_relevant_tests": int(len(model_relevant)),
        "n_wrapper_sanity_tests": int(len(wrapper_sanity)),
        "n_degenerate_ablation_diagnostics": int(len(degenerate)),
        "n_diagnostic_tests": int(len(diagnostic)),
        "presentation_changes": [
            "top findings retain available group context columns",
            "effect-size plots are horizontal",
            "ablation effect-size plot uses signed log10(1 + |Cohen dz|) display transform",
            "figure labels are compact; shared protocol/condition/reference context is moved into figure note boxes",
            "bar labels encode candidate ablation or run_id where useful",
            "statistical inputs from Patch 20B are not modified",
            "tests are classified into wrapper_sanity_check, model_relevant_test, degenerate_ablation_diagnostic, and diagnostic roles",
            "GLM critique response note is generated from role-aware tables",
        ],
        "outputs": {
            k: str(v)
            for k, v in outputs.items()
            if k not in ("meta", "report")
        },
    }

    outputs["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    outputs["report"].write_text(_report(meta, top, family), encoding="utf-8")
    outputs["glm_response"].write_text(
        _glm_response_report(meta, role_summary, model_relevant, wrapper_sanity, degenerate),
        encoding="utf-8",
    )

    print(f"Top findings saved: {outputs['top_findings']}")
    print(f"Test-family summary saved: {outputs['test_family_summary']}")
    print(f"Metric-direction summary saved: {outputs['metric_direction_summary']}")
    print(f"Role summary saved: {outputs['role_summary']}")
    print(f"Model-relevant tests saved: {outputs['model_relevant_tests']}")
    print(f"Model-relevant markdown saved: {outputs['model_relevant_tests_md']}")
    print(f"Wrapper sanity tests saved: {outputs['wrapper_sanity_tests']}")
    print(f"Wrapper sanity markdown saved: {outputs['wrapper_sanity_tests_md']}")
    print(f"Degenerate ablation diagnostics saved: {outputs['degenerate_ablation_diagnostics']}")
    print(f"Degenerate ablation markdown saved: {outputs['degenerate_ablation_diagnostics_md']}")
    print(f"GLM response saved: {outputs['glm_response']}")
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