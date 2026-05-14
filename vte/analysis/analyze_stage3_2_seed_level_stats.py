from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

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


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_20b(stats_dir: Path) -> dict[str, pd.DataFrame]:
    return {key: _load_optional_csv(stats_dir / filename) for key, filename in REQUIRED_20B_FILES.items()}


def _top_findings(tests: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    if tests.empty:
        return pd.DataFrame()
    df = tests.copy()
    df["p_fdr_bh"] = pd.to_numeric(df.get("p_fdr_bh"), errors="coerce")
    df["effect_size"] = pd.to_numeric(df.get("effect_size"), errors="coerce")
    df["mean_delta"] = pd.to_numeric(df.get("mean_delta"), errors="coerce")
    df = df.loc[df.get("status", "tested") == "tested"].copy()
    if df.empty:
        return pd.DataFrame()
    df["_abs_effect"] = df["effect_size"].abs()
    cols = [
        "test_family", "contrast", "metric", "group_a", "group_b",
        "n_seed_pairs", "mean_delta", "effect_size", "effect_size_type",
        "ci95_low", "ci95_high", "p_raw", "p_fdr_bh", "direction",
    ]
    return df.sort_values(["p_fdr_bh", "_abs_effect"], ascending=[True, False]).head(top_n)[[c for c in cols if c in df.columns]]


def _family_summary(tests: pd.DataFrame) -> pd.DataFrame:
    if tests.empty or "test_family" not in tests.columns:
        return pd.DataFrame()
    df = tests.copy()
    df["p_fdr_bh"] = pd.to_numeric(df.get("p_fdr_bh"), errors="coerce")
    rows = []
    for family, g in df.groupby("test_family", dropna=False, sort=True):
        rows.append({
            "test_family": family,
            "n_tests": int(len(g)),
            "n_tested": int((g.get("status", pd.Series(dtype=str)) == "tested").sum()),
            "n_fdr_lt_05": int((g["p_fdr_bh"] < 0.05).sum()),
            "n_fdr_lt_10": int((g["p_fdr_bh"] < 0.10).sum()),
            "min_p_fdr_bh": float(g["p_fdr_bh"].min()) if g["p_fdr_bh"].notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def _metric_direction_summary(tests: pd.DataFrame) -> pd.DataFrame:
    if tests.empty:
        return pd.DataFrame()
    df = tests.loc[tests.get("status", "tested") == "tested"].copy()
    if df.empty:
        return pd.DataFrame()
    rows = []
    for key, g in df.groupby(["test_family", "metric"], dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({
            "test_family": key[0],
            "metric": key[1],
            "n_tests": int(len(g)),
            "n_positive": int((g.get("direction") == "positive").sum()) if "direction" in g.columns else 0,
            "n_negative": int((g.get("direction") == "negative").sum()) if "direction" in g.columns else 0,
            "mean_effect_size": float(pd.to_numeric(g["effect_size"], errors="coerce").mean()),
            "median_effect_size": float(pd.to_numeric(g["effect_size"], errors="coerce").median()),
            "min_p_fdr_bh": float(pd.to_numeric(g["p_fdr_bh"], errors="coerce").min()),
        })
    return pd.DataFrame(rows)


def _save_vte_delta_effects(tests: pd.DataFrame, output_dir: Path, top_n: int = 20) -> str | None:
    if tests.empty:
        return None
    df = tests.loc[(tests.get("test_family") == "vte_binary_within_seed") & (tests.get("status") == "tested")].copy()
    if df.empty:
        return None
    df["effect_size"] = pd.to_numeric(df["effect_size"], errors="coerce")
    df["_abs_effect"] = df["effect_size"].abs()
    df = df.sort_values("_abs_effect", ascending=False).head(top_n)
    labels = [f"{r.metric}\n{str(r.get('condition', '') or r.get('ablation', ''))}".strip() for _, r in df.iterrows()]
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.5), 4.5))
    ax.bar(range(len(df)), df["effect_size"])
    ax.axhline(0, linewidth=1)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, rotation=65, ha="right")
    ax.set_ylabel("Effect size, Cohen dz")
    ax.set_title("Patch 20B VTE minus non-VTE seed-level effects")
    fig.tight_layout()
    filename = "Figure_3_2_Seed_Level_VTE_Delta_Effect_Sizes.png"
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)
    return filename


def _save_ablation_effects(tests: pd.DataFrame, output_dir: Path, top_n: int = 20) -> str | None:
    if tests.empty:
        return None
    df = tests.loc[(tests.get("test_family") == "ablation_vs_reference_within_seed") & (tests.get("status") == "tested")].copy()
    if df.empty:
        return None
    df["effect_size"] = pd.to_numeric(df["effect_size"], errors="coerce")
    df["_abs_effect"] = df["effect_size"].abs()
    df = df.sort_values("_abs_effect", ascending=False).head(top_n)
    labels = [f"{r.get('group_b')}:{r.metric}" for _, r in df.iterrows()]
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.5), 4.5))
    ax.bar(range(len(df)), df["effect_size"])
    ax.axhline(0, linewidth=1)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, rotation=65, ha="right")
    ax.set_ylabel("Effect size, Cohen dz")
    ax.set_title("Patch 20B ablation minus reference seed-level effects")
    fig.tight_layout()
    filename = "Figure_3_2_Seed_Level_Ablation_Effect_Sizes.png"
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)
    return filename


def _save_vte_rate_by_ablation(seed_aggregates: pd.DataFrame, output_dir: Path) -> str | None:
    if seed_aggregates.empty or "ablation" not in seed_aggregates.columns or "vte_rate" not in seed_aggregates.columns:
        return None
    df = seed_aggregates.copy()
    df["vte_rate"] = pd.to_numeric(df["vte_rate"], errors="coerce")
    df = df.dropna(subset=["vte_rate"])
    if df.empty:
        return None
    labels = sorted(str(x) for x in df["ablation"].dropna().unique())
    data = [df.loc[df["ablation"].astype(str) == label, "vte_rate"].dropna().to_numpy() for label in labels]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4.5))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_xlabel("Ablation")
    ax.set_ylabel("Seed-level VTE rate")
    ax.set_title("Seed-level VTE rate by ablation")
    fig.tight_layout()
    filename = "Figure_3_2_Seed_Level_VTE_Rate_By_Ablation.png"
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)
    return filename


def _report(meta: dict[str, object], top: pd.DataFrame, family: pd.DataFrame) -> str:
    lines = ["# Patch 20C Seed-Level Statistics Analyzer", "", "Patch 20C reads Patch 20B outputs and produces compact inspection tables and figures.", "", "## Inputs", "", f"- Stats directory: `{meta['stats_dir']}`", "", "## Outputs", ""]
    for p in meta["outputs"].values():
        lines.append(f"- `{p}`")
    lines += ["", "## Test-family summary", ""]
    if family.empty:
        lines.append("No test-family summary available.")
    else:
        for _, r in family.iterrows():
            lines.append(f"- {r['test_family']}: n_tests={r['n_tests']}, n_tested={r['n_tested']}, fdr<0.05={r['n_fdr_lt_05']}")
    lines += ["", "## Top findings", ""]
    if top.empty:
        lines.append("No tested findings available.")
    else:
        for _, r in top.head(12).iterrows():
            lines.append(f"- {r.get('test_family')} | {r.get('contrast')} | {r.get('metric')}: effect={r.get('effect_size'):.6g}, p_fdr={r.get('p_fdr_bh'):.6g}")
    lines += ["", "## Interpretation boundary", "", "This analyzer is descriptive. It does not introduce new statistical tests beyond Patch 20B.", ""]
    return "\n".join(lines)


def analyze_seed_level_stats(stats_dir: str | Path, output_dir: str | Path, top_n: int = 25) -> dict[str, object]:
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
    figures = [x for x in (_save_vte_delta_effects(tests, out, top_n), _save_ablation_effects(tests, out, top_n), _save_vte_rate_by_ablation(seed_aggregates, out)) if x]
    meta = {"patch": "20C", "script": "vte.analysis.analyze_stage3_2_seed_level_stats", "created_at": _now_iso(), "stats_dir": str(stats_path), "output_dir": str(out), "n_tests": int(len(tests)), "n_top_findings": int(len(top)), "n_figures": int(len(figures)), "figures": figures, "outputs": {k: str(v) for k, v in outputs.items() if k not in ("meta", "report")}}
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
    p = argparse.ArgumentParser(description="Patch 20C: analyze Patch 20B seed-level statistical outputs.")
    p.add_argument("--stats-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--top-n", type=int, default=25)
    return p


def main() -> None:
    a = build_arg_parser().parse_args()
    analyze_seed_level_stats(a.stats_dir, a.output_dir, a.top_n)


if __name__ == "__main__":
    main()
