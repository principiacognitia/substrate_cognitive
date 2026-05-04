from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

SELECTED_EXAMPLE_COLUMNS = [
    "example_type",
    "run_id",
    "seed",
    "trial",
    "committed_path",
    "raw_idphi",
    "z_idphi",
    "pause_ticks",
    "reorientation_count",
    "vte_binary",
    "trace_csv",
    "recommended_output_name",
]


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _sanitize_label(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "run"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_suite_root(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    for key in ("suite_root", "root", "base_output_dir", "output_dir"):
        value = manifest.get(key)
        if value:
            candidate = Path(value)
            if candidate.exists():
                return candidate.resolve()
    return manifest_path.parent.resolve()


def _find_stage3_steps_csvs(suite_root: Path) -> list[Path]:
    if not suite_root.exists():
        raise FileNotFoundError(f"Suite root does not exist: {suite_root}")

    files = []
    for path in suite_root.rglob("*_all_steps.csv"):
        parts = {p.lower() for p in path.parts}
        if "analysis" in parts or "analysis_publication" in parts:
            continue
        files.append(path.resolve())

    return sorted(files)


def _infer_protocol(path: Path, suite_root: Path) -> str:
    rel_parts = [p.lower() for p in path.relative_to(suite_root).parts]
    for protocol in ("balanced", "shock", "treat"):
        if protocol in rel_parts:
            return protocol
    name = path.name.lower()
    if "shock" in name:
        return "shock"
    if "treat" in name:
        return "treat"
    if "balanced" in name:
        return "balanced"
    return "unknown"


def _infer_ablation(path: Path, suite_root: Path, protocol: str) -> str:
    rel_parts = [p for p in path.relative_to(suite_root).parts]
    rel_lower = [p.lower() for p in rel_parts]

    if protocol in rel_lower:
        idx = rel_lower.index(protocol)
        if idx + 1 < len(rel_parts):
            return rel_parts[idx + 1]

    name = path.stem
    for ablation in ("one_shot_off", "full", "novg", "novp", "nox"):
        if ablation in name:
            return ablation

    return "unknown"


def _infer_condition(path: Path) -> str:
    name = path.name.lower()
    if "balanced_conflict" in name:
        return "R1_T2"
    return ""


def _select_runs(
    files: list[Path],
    suite_root: Path,
    protocols: set[str] | None,
    ablations: set[str] | None,
    max_runs: int | None,
) -> list[dict[str, Any]]:
    runs = []

    for path in files:
        protocol = _infer_protocol(path, suite_root)
        ablation = _infer_ablation(path, suite_root, protocol)
        condition = _infer_condition(path)

        if protocols and protocol not in protocols:
            continue
        if ablations and ablation not in ablations:
            continue

        run_label = _sanitize_label(f"{protocol}_{ablation}_{path.stem}")
        runs.append(
            {
                "run_label": run_label,
                "protocol": protocol,
                "ablation": ablation,
                "condition": condition,
                "stage3_steps_csv": str(path),
            }
        )

    if max_runs is not None:
        runs = runs[:max_runs]

    return runs


def _run_command(cmd: list[str], dry_run: bool = False) -> None:
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected {label} not found: {path}")


def _concat_metrics(metric_files: list[Path], output_csv: Path) -> int:
    frames = []
    for path in metric_files:
        _require_file(path, "VTE metrics CSV")
        frames.append(pd.read_csv(path))

    if not frames:
        raise ValueError("No VTE metric files were produced.")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_csv, index=False)
    return len(combined)



def _format_scalar(value: Any) -> str:
    if pd.isna(value):
        return "na"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _relative_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(path)


def _trace_csv_by_run_id(batch_runs: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    trace_map: dict[str, str] = {}

    for run in batch_runs:
        run_id = str(run.get("run_label", ""))
        trace_csv = run.get("vte_trace_csv")
        if not run_id or not trace_csv:
            continue
        trace_map[run_id] = _relative_path(Path(trace_csv), output_dir)

    return trace_map


def _required_selected_example_input_columns() -> set[str]:
    return {
        "run_id",
        "seed",
        "trial",
        "committed_path",
        "raw_idphi",
        "z_idphi",
        "pause_ticks",
        "reorientation_count",
        "vte_binary",
    }


def _recommended_output_name(row: pd.Series, example_type: str) -> str:
    committed_path = row.get("committed_path", "unknown")
    return _sanitize_label(
        "__".join(
            [
                example_type,
                _format_scalar(row.get("run_id", "run")),
                f"s{_format_scalar(row.get('seed', 'na'))}",
                f"t{_format_scalar(row.get('trial', 'na'))}",
                _format_scalar(committed_path or "path_unknown"),
            ]
        )
    )


def _selected_example_record(row: pd.Series, example_type: str) -> dict[str, Any]:
    record = {col: row.get(col, "") for col in SELECTED_EXAMPLE_COLUMNS}
    record["example_type"] = example_type
    record["recommended_output_name"] = _recommended_output_name(row, example_type)
    return record


def _sort_top_vte(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        by=[
            "z_idphi",
            "raw_idphi",
            "reorientation_count",
            "pause_ticks",
            "run_id",
            "seed",
            "trial",
        ],
        ascending=[False, False, False, False, True, True, True],
        kind="mergesort",
    )


def _sort_clean_non_vte(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        by=[
            "z_idphi",
            "raw_idphi",
            "reorientation_count",
            "pause_ticks",
            "run_id",
            "seed",
            "trial",
        ],
        ascending=[True, True, True, True, True, True, True],
        kind="mergesort",
    )


def _best_matched_pause_control(
    top_row: pd.Series,
    non_vte: pd.DataFrame,
    used_row_ids: set[int],
) -> pd.Series | None:
    candidates = non_vte.loc[~non_vte["_example_row_id"].isin(used_row_ids)].copy()
    if candidates.empty:
        candidates = non_vte.copy()
    if candidates.empty:
        return None

    top_path = str(top_row.get("committed_path", ""))
    top_pause = float(top_row.get("pause_ticks", 0.0))

    candidates["_same_path_rank"] = (
        candidates["committed_path"].astype(str) != top_path
    ).astype(int)
    candidates["_pause_delta"] = (
        candidates["pause_ticks"].astype(float) - top_pause
    ).abs()

    candidates = candidates.sort_values(
        by=[
            "_same_path_rank",
            "_pause_delta",
            "z_idphi",
            "raw_idphi",
            "reorientation_count",
            "run_id",
            "seed",
            "trial",
        ],
        ascending=[True, True, True, True, True, True, True, True],
        kind="mergesort",
    )
    return candidates.iloc[0]


def _select_visualization_examples(
    metrics_df: pd.DataFrame,
    max_per_type: int = 10,
) -> pd.DataFrame:
    missing = _required_selected_example_input_columns() - set(metrics_df.columns)
    if missing:
        raise ValueError(
            "VTE metrics are missing columns required for selected examples: "
            f"{sorted(missing)}"
        )

    df = metrics_df.copy()
    df["_example_row_id"] = range(len(df))

    top_vte = _sort_top_vte(df.loc[df["vte_binary"].astype(int) == 1]).head(max_per_type)
    clean_non_vte = _sort_clean_non_vte(
        df.loc[df["vte_binary"].astype(int) == 0]
    ).head(max_per_type)
    all_non_vte = df.loc[df["vte_binary"].astype(int) == 0]

    records: list[dict[str, Any]] = []
    for _, row in top_vte.iterrows():
        records.append(_selected_example_record(row, "top_vte"))

    for _, row in clean_non_vte.iterrows():
        records.append(_selected_example_record(row, "clean_non_vte"))

    used_control_ids: set[int] = set()
    for _, top_row in top_vte.iterrows():
        control = _best_matched_pause_control(top_row, all_non_vte, used_control_ids)
        if control is None:
            continue
        used_control_ids.add(int(control["_example_row_id"]))
        records.append(_selected_example_record(control, "matched_pause_control"))

    selected = pd.DataFrame(records)
    for col in SELECTED_EXAMPLE_COLUMNS:
        if col not in selected.columns:
            selected[col] = []
    return selected[SELECTED_EXAMPLE_COLUMNS]


def _write_selected_examples(
    metrics_df: pd.DataFrame,
    batch_runs: list[dict[str, Any]],
    output_csv: Path,
    output_dir: Path,
    max_per_type: int = 10,
) -> int:
    df = metrics_df.copy()
    trace_map = _trace_csv_by_run_id(batch_runs, output_dir)
    df["trace_csv"] = df["run_id"].astype(str).map(trace_map).fillna("")

    selected = _select_visualization_examples(df, max_per_type=max_per_type)
    selected.to_csv(output_csv, index=False)
    return len(selected)


def run_batch(
    manifest_path: Path,
    output_dir: Path,
    protocols: set[str] | None = None,
    ablations: set[str] | None = None,
    max_runs: int | None = None,
    skip_analysis: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    output_dir = output_dir.resolve()

    manifest = _load_json(manifest_path)
    suite_root = _infer_suite_root(manifest_path, manifest)

    traces_dir = output_dir / "traces"
    metrics_dir = output_dir / "metrics"
    analysis_dir = output_dir / "analysis"

    for d in (output_dir, traces_dir, metrics_dir):
        if not dry_run:
            d.mkdir(parents=True, exist_ok=True)

    steps_files = _find_stage3_steps_csvs(suite_root)
    runs = _select_runs(
        files=steps_files,
        suite_root=suite_root,
        protocols=protocols,
        ablations=ablations,
        max_runs=max_runs,
    )

    if not runs:
        raise ValueError(
            f"No Stage 3 *_all_steps.csv files selected under suite root: {suite_root}"
        )

    metric_files: list[Path] = []
    batch_runs: list[dict[str, Any]] = []

    for run in runs:
        run_label = run["run_label"]
        steps_csv = Path(run["stage3_steps_csv"])
        trace_csv = traces_dir / f"{run_label}_vte_trace.csv"
        run_metrics_dir = metrics_dir / run_label
        metrics_csv = run_metrics_dir / "vte_trial_metrics.csv"

        if not dry_run:
            run_metrics_dir.mkdir(parents=True, exist_ok=True)

        _run_command(
            [
                sys.executable,
                "-m",
                "vte.analysis.translate_stage3_steps_to_vte_trace",
                "--input-csv",
                str(steps_csv),
                "--output-csv",
                str(trace_csv),
                "--run-id",
                run_label,
            ],
            dry_run=dry_run,
        )

        if not dry_run:
            _require_file(trace_csv, "translated VTE trace")

        _run_command(
            [
                sys.executable,
                "-m",
                "vte.analysis.run_stage3_2_vte",
                "--input-csv",
                str(trace_csv),
                "--output-dir",
                str(run_metrics_dir),
            ],
            dry_run=dry_run,
        )

        if not dry_run:
            _require_file(metrics_csv, "VTE trial metrics")
            metric_files.append(metrics_csv)

        run_record = {
            **run,
            "vte_trace_csv": str(trace_csv),
            "vte_metrics_csv": str(metrics_csv),
        }
        batch_runs.append(run_record)

    combined_metrics_csv = output_dir / "vte_trial_metrics_all.csv"
    selected_examples_csv = output_dir / "Table_3_2_VTE_Selected_Examples.csv"
    n_metric_rows = 0
    n_selected_examples = 0

    if not dry_run:
        n_metric_rows = _concat_metrics(metric_files, combined_metrics_csv)
        print(f"✓ Combined VTE metrics saved: {combined_metrics_csv}")

        combined_metrics = pd.read_csv(combined_metrics_csv)
        n_selected_examples = _write_selected_examples(
            metrics_df=combined_metrics,
            batch_runs=batch_runs,
            output_csv=selected_examples_csv,
            output_dir=output_dir,
        )
        print(f"✓ Selected VTE examples saved: {selected_examples_csv}")

    if not skip_analysis:
        if not dry_run:
            analysis_dir.mkdir(parents=True, exist_ok=True)

        _run_command(
            [
                sys.executable,
                "-m",
                "vte.analysis.analyze_stage3_2_vte",
                "--metrics-csv",
                str(combined_metrics_csv),
                "--output-dir",
                str(analysis_dir),
            ],
            dry_run=dry_run,
        )

    batch_manifest = {
        "script": "vte.analysis.run_stage3_2_vte_batch",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_manifest": str(manifest_path),
        "suite_root": str(suite_root),
        "output_dir": str(output_dir),
        "n_stage3_steps_files_found": len(steps_files),
        "n_runs_selected": len(batch_runs),
        "n_metric_rows": n_metric_rows,
        "combined_metrics_csv": str(combined_metrics_csv),
        "selected_examples_csv": str(selected_examples_csv),
        "n_selected_examples": n_selected_examples,
        "analysis_dir": str(analysis_dir) if not skip_analysis else None,
        "runs": batch_runs,
    }

    batch_manifest_path = output_dir / "stage3_2_vte_batch_manifest.json"
    if not dry_run:
        with batch_manifest_path.open("w", encoding="utf-8") as f:
            json.dump(batch_manifest, f, indent=2)
        print(f"✓ Batch manifest saved: {batch_manifest_path}")

    return batch_manifest


def _parse_csv_filter(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {v.strip() for v in value.split(",") if v.strip()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stage 3.2 VTE wrapper over all Stage 3.1B step logs in a suite."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Stage 3.1B suite manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for traces, per-run metrics, combined metrics, analysis, and batch manifest.",
    )
    parser.add_argument(
        "--protocols",
        default=None,
        help="Optional comma-separated protocol filter, e.g. balanced,shock,treat.",
    )
    parser.add_argument(
        "--ablations",
        default=None,
        help="Optional comma-separated ablation filter, e.g. full,novg,novp,nox,one_shot_off.",
    )
    parser.add_argument(
        "--max-runs",
        default=None,
        type=int,
        help="Optional limit for smoke testing.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Only translate and wrap; do not run aggregate analysis.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    protocols = _parse_csv_filter(args.protocols)
    ablations = _parse_csv_filter(args.ablations)

    meta = run_batch(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        protocols=protocols,
        ablations=ablations,
        max_runs=args.max_runs,
        skip_analysis=args.skip_analysis,
        dry_run=args.dry_run,
    )

    print()
    print("Stage 3.2 VTE batch complete")
    print(f"Runs selected: {meta['n_runs_selected']}")
    print(f"Metric rows: {meta['n_metric_rows']}")
    print(f"Selected examples: {meta['n_selected_examples']}")
    print(f"Output: {meta['output_dir']}")


if __name__ == "__main__":
    main()