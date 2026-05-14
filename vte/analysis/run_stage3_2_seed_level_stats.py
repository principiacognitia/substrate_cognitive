from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_GROUP_COLUMNS = (
    "source",
    "dataset_id",
    "task_family",
    "protocol",
    "condition",
    "ablation",
    "run_id",
)

CANDIDATE_NUMERIC_COLUMNS = (
    "reward",
    "total_reward",
    "cost",
    "native_cost_bin",
    "comparable_cost_bin",
    "raw_idphi",
    "lab_idphi",
    "z_idphi",
    "z_lab_idphi_by_session",
    "pause_ticks",
    "pause_time_s",
    "dwell_proxy",
    "dwell_z",
    "z_dwell_proxy_by_session",
    "deliberation_proxy",
    "deliberation_z",
    "z_pause_time_by_session",
    "reorientation_count",
)

REQUIRED_MINIMAL_COLUMNS = ("seed", "trial", "vte_binary")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_csvs(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Input metrics CSV not found: {path}")
        df = pd.read_csv(path)
        df["source_file"] = str(path)
        frames.append(df)
    if not frames:
        raise ValueError("No input metrics CSVs were provided.")
    return pd.concat(frames, ignore_index=True)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "seed" not in out.columns:
        for c in ("subject_or_seed", "subject_id"):
            if c in out.columns:
                out["seed"] = out[c].astype(str)
                break
    missing = [c for c in REQUIRED_MINIMAL_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns for seed-level stats: {missing}")
    for c in set(CANDIDATE_NUMERIC_COLUMNS + ("vte_binary",)):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "reward" not in out.columns and "total_reward" in out.columns:
        out["reward"] = pd.to_numeric(out["total_reward"], errors="coerce")
    if "reward" not in out.columns and "outcome" in out.columns:
        mapping = {"correct": 1.0, "error": 0.0, "success": 1.0, "failure": 0.0}
        out["reward"] = out["outcome"].astype(str).str.lower().map(mapping)
    return out


def _available_group_columns(df: pd.DataFrame, requested: list[str] | None) -> list[str]:
    cols = requested if requested else list(DEFAULT_GROUP_COLUMNS)
    return [c for c in cols if c in df.columns]


def _available_metric_columns(df: pd.DataFrame) -> list[str]:
    metrics = []
    for c in CANDIDATE_NUMERIC_COLUMNS:
        if c in df.columns and c not in metrics and pd.to_numeric(df[c], errors="coerce").notna().any():
            metrics.append(c)
    return metrics


def _groupby(df: pd.DataFrame, cols: list[str]):
    if cols:
        return df.groupby(cols, dropna=False, sort=True)
    return [((), df)]


def _key_dict(cols: list[str], key) -> dict[str, object]:
    if not cols:
        return {}
    if not isinstance(key, tuple):
        key = (key,)
    return dict(zip(cols, key))


def _seed_aggregates(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    cols = group_cols + ["seed"]
    for key, g in _groupby(df, cols):
        row = _key_dict(cols, key)
        vb = pd.to_numeric(g["vte_binary"], errors="coerce")
        row.update(
            n_trials=int(len(g)),
            n_vte=int((vb == 1).sum()),
            n_nonvte=int((vb == 0).sum()),
            vte_rate=float(vb.mean()),
        )
        for m in metrics:
            v = pd.to_numeric(g[m], errors="coerce")
            row[f"{m}_mean"] = float(v.mean()) if v.notna().any() else np.nan
            row[f"{m}_median"] = float(v.median()) if v.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _seed_by_vte(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    cols = group_cols + ["seed", "vte_binary"]
    clean = df.loc[df["vte_binary"].isin([0, 1])].copy()
    for key, g in _groupby(clean, cols):
        row = _key_dict(cols, key)
        row["n_trials"] = int(len(g))
        for m in metrics:
            v = pd.to_numeric(g[m], errors="coerce")
            row[f"{m}_mean"] = float(v.mean()) if v.notna().any() else np.nan
            row[f"{m}_median"] = float(v.median()) if v.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _vte_contrasts(seed_by_vte: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    if seed_by_vte.empty:
        return pd.DataFrame()
    rows = []
    cols = group_cols + ["seed"]
    for key, g in _groupby(seed_by_vte, cols):
        if set(pd.to_numeric(g["vte_binary"], errors="coerce").dropna().astype(int)) != {0, 1}:
            continue
        row = _key_dict(cols, key)
        non = g.loc[pd.to_numeric(g["vte_binary"], errors="coerce") == 0]
        vte = g.loc[pd.to_numeric(g["vte_binary"], errors="coerce") == 1]
        row["n_nonvte_trials"] = int(non["n_trials"].sum())
        row["n_vte_trials"] = int(vte["n_trials"].sum())
        for m in metrics:
            c = f"{m}_mean"
            if c in g.columns:
                a = pd.to_numeric(non[c], errors="coerce").mean()
                b = pd.to_numeric(vte[c], errors="coerce").mean()
                row[f"{m}_vte_minus_nonvte"] = float(b - a) if pd.notna(a) and pd.notna(b) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _stable_seed(values: pd.Series, metric: str, analysis_id: str) -> int:
    text = "|".join([analysis_id, metric] + [f"{float(x):.12g}" for x in values])
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _bootstrap_ci(values: pd.Series, seed: int, n_boot: int = 4000) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float).to_numpy()
    n = len(clean)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(clean[0]), float(clean[0])
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(float(np.mean([clean[rng.randrange(n)] for _ in range(n)])))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _sign_flip(values: pd.Series, metric: str, analysis_id: str, exact_n: int = 20, n_perm: int = 20000) -> dict[str, object]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    n = int(len(clean))
    if n == 0:
        return dict(n_seed_pairs=0, mean_delta=np.nan, median_delta=np.nan, sd_delta=np.nan, test_statistic=np.nan, p_raw=np.nan, effect_size=np.nan, effect_size_type="cohen_dz", ci95_low=np.nan, ci95_high=np.nan, direction="insufficient_data")
    raw = clean.to_numpy()
    mean_delta = float(raw.mean())
    observed = abs(mean_delta)
    sd = float(raw.std(ddof=1)) if n > 1 else 0.0
    if sd > 0:
        effect = float(mean_delta / sd)
    elif mean_delta > 0:
        effect = math.inf
    elif mean_delta < 0:
        effect = -math.inf
    else:
        effect = 0.0
    if n <= exact_n:
        total = 2 ** n
        extreme = 0
        for signs in product((-1.0, 1.0), repeat=n):
            if abs(float(np.mean(raw * np.array(signs)))) >= observed - 1e-15:
                extreme += 1
        p = extreme / total
    else:
        rng = random.Random(_stable_seed(clean, metric, analysis_id))
        extreme = 0
        for _ in range(n_perm):
            signs = np.array([1.0 if rng.random() >= 0.5 else -1.0 for _ in range(n)])
            if abs(float(np.mean(raw * signs))) >= observed - 1e-15:
                extreme += 1
        p = (extreme + 1) / (n_perm + 1)
    ci_low, ci_high = _bootstrap_ci(clean, _stable_seed(clean, metric + "_ci", analysis_id))
    direction = "positive" if mean_delta > 0 else "negative" if mean_delta < 0 else "zero"
    return dict(n_seed_pairs=n, mean_delta=mean_delta, median_delta=float(np.median(raw)), sd_delta=sd, test_statistic=mean_delta, p_raw=float(p), effect_size=effect, effect_size_type="cohen_dz", ci95_low=ci_low, ci95_high=ci_high, direction=direction)


def _bh(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna()
    if valid.empty:
        return out
    order = valid.sort_values().index
    vals = valid.loc[order].to_numpy()
    m = len(vals)
    adj = np.empty(m)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        val = min(prev, vals[i] * m / (i + 1), 1.0)
        adj[i] = val
        prev = val
    out.loc[order] = adj
    return out


def _analysis_id(prefix: str, key_dict: dict[str, object], cols: list[str]) -> str:
    parts = [prefix] + [str(key_dict.get(c, "")) for c in cols if str(key_dict.get(c, "")) not in ("", "nan", "None")]
    return "::".join(parts)


def _test_vte(vte_contrasts: pd.DataFrame, group_cols: list[str], metrics: list[str], min_seed_pairs: int) -> pd.DataFrame:
    if vte_contrasts.empty:
        return pd.DataFrame()
    rows = []
    for key, g in _groupby(vte_contrasts, group_cols):
        kd = _key_dict(group_cols, key)
        aid = _analysis_id("vte_minus_nonvte", kd, group_cols)
        for m in metrics:
            dc = f"{m}_vte_minus_nonvte"
            if dc not in g.columns:
                continue
            vals = pd.to_numeric(g[dc], errors="coerce").dropna()
            status = "tested" if len(vals) >= min_seed_pairs else "insufficient_seed_pairs"
            result = _sign_flip(vals if status == "tested" else pd.Series(dtype=float), m, aid)
            rows.append({**kd, "analysis_id": aid, "test_family": "vte_binary_within_seed", "contrast": "vte_minus_nonvte", "metric": m, "delta_column": dc, "group_a": "nonvte", "group_b": "vte", "test_name": "two_sided_sign_flip_mean_delta", "status": status, **result})
    return pd.DataFrame(rows)


def _ablation_contrasts(seed_agg: pd.DataFrame, group_cols: list[str], reference: str) -> pd.DataFrame:
    if "ablation" not in seed_agg.columns:
        return pd.DataFrame()
    pair_cols = [c for c in group_cols if c not in ("ablation", "run_id")] + ["seed"]
    value_cols = [c for c in seed_agg.columns if c == "vte_rate" or c.endswith("_mean")]
    rows = []
    for key, g in _groupby(seed_agg, pair_cols):
        if reference not in set(g["ablation"].astype(str)):
            continue
        kd = _key_dict(pair_cols, key)
        ref = g.loc[g["ablation"].astype(str) == reference].iloc[0]
        for _, cand in g.iterrows():
            ab = str(cand.get("ablation", ""))
            if ab == reference:
                continue
            row = {**kd, "reference_ablation": reference, "candidate_ablation": ab}
            for c in value_cols:
                a = pd.to_numeric(pd.Series([ref.get(c)]), errors="coerce").iloc[0]
                b = pd.to_numeric(pd.Series([cand.get(c)]), errors="coerce").iloc[0]
                row[f"{c}_candidate_minus_reference"] = float(b - a) if pd.notna(a) and pd.notna(b) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _test_ablation(ab_contrasts: pd.DataFrame, group_cols: list[str], min_seed_pairs: int) -> pd.DataFrame:
    if ab_contrasts.empty:
        return pd.DataFrame()
    test_cols = [c for c in group_cols if c not in ("ablation", "run_id")] + ["reference_ablation", "candidate_ablation"]
    delta_cols = [c for c in ab_contrasts.columns if c.endswith("_candidate_minus_reference")]
    rows = []
    for key, g in _groupby(ab_contrasts, test_cols):
        kd = _key_dict(test_cols, key)
        aid = _analysis_id("ablation_vs_reference", kd, test_cols)
        for dc in delta_cols:
            metric = dc.replace("_candidate_minus_reference", "")
            vals = pd.to_numeric(g[dc], errors="coerce").dropna()
            status = "tested" if len(vals) >= min_seed_pairs else "insufficient_seed_pairs"
            result = _sign_flip(vals if status == "tested" else pd.Series(dtype=float), metric, aid)
            rows.append({**kd, "analysis_id": aid, "test_family": "ablation_vs_reference_within_seed", "contrast": f"{kd.get('candidate_ablation')} minus {kd.get('reference_ablation')}", "metric": metric, "delta_column": dc, "group_a": kd.get("reference_ablation"), "group_b": kd.get("candidate_ablation"), "test_name": "two_sided_sign_flip_mean_delta", "status": status, **result})
    return pd.DataFrame(rows)


def _effect_table(tests: pd.DataFrame) -> pd.DataFrame:
    cols = ["analysis_id", "test_family", "contrast", "metric", "group_a", "group_b", "n_seed_pairs", "mean_delta", "median_delta", "effect_size", "effect_size_type", "ci95_low", "ci95_high", "direction", "status"]
    return tests[[c for c in cols if c in tests.columns]].copy() if not tests.empty else pd.DataFrame()


def _report(meta: dict[str, object], tests: pd.DataFrame) -> str:
    lines = ["# Stage 3.2 Seed-Level Statistical Tests", "", "Patch 20B treats seed as the inferential unit. Trial rows are measurement observations, not independent inferential replicates.", "", "## Summary", "", f"- Trial rows: {meta['n_trial_rows']}", f"- Seed aggregate rows: {meta['n_seed_aggregate_rows']}", f"- VTE contrast rows: {meta['n_seed_vte_contrast_rows']}", f"- Ablation contrast rows: {meta['n_ablation_contrast_rows']}", f"- Statistical tests: {meta['n_statistical_tests']}", f"- Tested rows: {meta['n_tests_with_status_tested']}", "", "## Top tested effects", ""]
    if tests.empty or (tests.get("status") == "tested").sum() == 0:
        lines.append("No tests met the minimum seed-pair threshold.")
    else:
        top = tests.loc[tests["status"] == "tested"].copy()
        top["_abs_effect"] = pd.to_numeric(top["effect_size"], errors="coerce").abs()
        top = top.sort_values(["p_fdr_bh", "_abs_effect"], ascending=[True, False]).head(12)
        for _, r in top.iterrows():
            lines.append(f"- {r.get('test_family')} | {r.get('contrast')} | {r.get('metric')}: n={r.get('n_seed_pairs')}, mean_delta={r.get('mean_delta'):.6g}, effect={r.get('effect_size'):.6g}, p_fdr={r.get('p_fdr_bh'):.6g}")
    lines += ["", "## Interpretation boundary", "", "These tests are diagnostic seed-level summaries. They do not compare biological and synthetic action labels unless a separate namespace policy explicitly permits that comparison.", ""]
    return "\n".join(lines)


def run_seed_level_stats(metrics_csvs: list[str | Path], output_dir: str | Path, group_columns: list[str] | None = None, reference_ablation: str = "full", min_seed_pairs: int = 3) -> dict[str, object]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = _normalize_columns(_read_csvs(metrics_csvs))
    group_cols = _available_group_columns(df, group_columns)
    metrics = _available_metric_columns(df)
    if not metrics:
        raise ValueError("No usable numeric metrics were found for seed-level stats.")

    seed_agg = _seed_aggregates(df, group_cols, metrics)
    seed_vte = _seed_by_vte(df, group_cols, metrics)
    vte_con = _vte_contrasts(seed_vte, group_cols, metrics)
    ab_con = _ablation_contrasts(seed_agg, group_cols, reference_ablation)
    tests = pd.concat([x for x in (_test_vte(vte_con, group_cols, metrics, min_seed_pairs), _test_ablation(ab_con, group_cols, min_seed_pairs)) if not x.empty], ignore_index=True) if (not vte_con.empty or not ab_con.empty) else pd.DataFrame()
    if not tests.empty:
        tests["p_fdr_bh"] = _bh(tests["p_raw"])
    effects = _effect_table(tests)

    paths = {
        "seed_aggregates": out / "Table_3_2_Seed_Level_Aggregates.csv",
        "seed_by_vte": out / "Table_3_2_Seed_Level_By_VTE.csv",
        "seed_vte_contrasts": out / "Table_3_2_Seed_Level_VTE_Contrasts.csv",
        "ablation_contrasts": out / "Table_3_2_Seed_Level_Ablation_Contrasts.csv",
        "statistical_tests": out / "Table_3_2_Seed_Level_Statistical_Tests.csv",
        "effect_sizes": out / "Table_3_2_Seed_Level_Effect_Sizes.csv",
        "meta": out / "stage3_2_seed_level_stats_meta.json",
        "report": out / "Stage3_2_Seed_Level_Stats_Report.md",
    }
    seed_agg.to_csv(paths["seed_aggregates"], index=False)
    seed_vte.to_csv(paths["seed_by_vte"], index=False)
    vte_con.to_csv(paths["seed_vte_contrasts"], index=False)
    ab_con.to_csv(paths["ablation_contrasts"], index=False)
    tests.to_csv(paths["statistical_tests"], index=False)
    effects.to_csv(paths["effect_sizes"], index=False)

    meta = {
        "patch": "20B",
        "script": "vte.analysis.run_stage3_2_seed_level_stats",
        "created_at": _now_iso(),
        "input_csvs": [str(Path(p)) for p in metrics_csvs],
        "output_dir": str(out),
        "seed_is_inferential_unit": True,
        "trial_rows_are_measurement_observations": True,
        "reference_ablation": reference_ablation,
        "min_seed_pairs": int(min_seed_pairs),
        "n_trial_rows": int(len(df)),
        "n_seed_aggregate_rows": int(len(seed_agg)),
        "n_seed_by_vte_rows": int(len(seed_vte)),
        "n_seed_vte_contrast_rows": int(len(vte_con)),
        "n_ablation_contrast_rows": int(len(ab_con)),
        "n_statistical_tests": int(len(tests)),
        "n_tests_with_status_tested": int((tests.get("status", pd.Series(dtype=str)) == "tested").sum()) if not tests.empty else 0,
        "group_columns": group_cols,
        "metric_columns": metrics,
        "outputs": {k: str(v) for k, v in paths.items() if k not in ("meta", "report")},
    }
    paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    paths["report"].write_text(_report(meta, tests), encoding="utf-8")

    print(f"Seed aggregates saved: {paths['seed_aggregates']}")
    print(f"Seed VTE contrasts saved: {paths['seed_vte_contrasts']}")
    print(f"Ablation contrasts saved: {paths['ablation_contrasts']}")
    print(f"Statistical tests saved: {paths['statistical_tests']}")
    print(f"Effect sizes saved: {paths['effect_sizes']}")
    print(f"Metadata saved: {paths['meta']}")
    print(f"Report saved: {paths['report']}")
    print(f"Rows: {meta['n_trial_rows']}")
    print(f"Tests: {meta['n_statistical_tests']}")
    return meta


def _parse_group_columns(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [x.strip() for x in raw.split(",") if x.strip()] or None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Patch 20B: seed-level statistical tests for Stage 3.2 VTE metrics.")
    p.add_argument("--metrics-csv", required=True, nargs="+", help="One or more trial-level VTE metrics CSV files.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--group-columns", default=None, help="Optional comma-separated grouping columns.")
    p.add_argument("--reference-ablation", default="full")
    p.add_argument("--min-seed-pairs", default=3, type=int)
    return p


def main() -> None:
    a = build_arg_parser().parse_args()
    run_seed_level_stats(a.metrics_csv, a.output_dir, _parse_group_columns(a.group_columns), a.reference_ablation, a.min_seed_pairs)


if __name__ == "__main__":
    main()
