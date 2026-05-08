from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DATASET_ID_DEFAULT = "redish_rrow_2022"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def discover_data_roots(root: Path) -> list[Path]:
    root = Path(root)

    candidates: list[Path] = []

    if (root / "Processed Behavior").is_dir():
        candidates.append(root)

    for data_dir in root.rglob("Data"):
        if (data_dir / "Processed Behavior").is_dir():
            candidates.append(data_dir)

    unique = sorted({p.resolve() for p in candidates})
    return unique


def _version_label_for_path(data_root: Path, global_root: Path) -> str:
    try:
        rel_parts = data_root.relative_to(global_root.resolve()).parts
    except Exception:
        rel_parts = data_root.parts

    for part in rel_parts:
        lower = part.lower()
        if lower.startswith("version "):
            return part
        if "2022-12-07" in lower:
            return "Version 002 - 2022-12-07"
        if "2022-10-10" in lower:
            return "Version 001 - 2022-10-10"

    return data_root.parent.name


def classify_redish_file(path: Path) -> str:
    name = path.name
    lower = name.lower()

    if lower == "idphi_rrow.mat":
        return "aggregate_idphi"
    if lower == "lapdata_behav_rrow.mat":
        return "aggregate_lapdata_behavior"
    if lower == "sessiondata_rrow.mat":
        return "aggregate_sessiondata"
    if lower == "datadef_rrow.mat":
        return "data_definition"
    if lower.endswith("_keys.m"):
        return "session_keys"
    if lower.endswith("-rrow.mat"):
        return "session_rrow_behavior"
    if lower.endswith("-vt.mat"):
        return "session_tracking_vt"
    if lower.endswith("t_wf.mat"):
        return "unit_waveform"
    if re.search(r"\.t$", lower):
        return "unit_spike_times"
    if lower.endswith(".mat") and "analysis" in str(path).lower():
        return "analysis_mat"
    if lower.endswith(".mat"):
        return "other_mat"
    if lower.endswith(".m"):
        return "matlab_code"
    if lower.endswith(".txt"):
        return "text_metadata"
    return "other"


def _extract_subject_session(path: Path, data_root: Path) -> tuple[str, str, str]:
    subject_id = ""
    session_id = ""
    session_date = ""

    try:
        rel = path.relative_to(data_root)
        parts = rel.parts
    except Exception:
        parts = path.parts

    for idx, part in enumerate(parts):
        if re.fullmatch(r"R\d+", part):
            subject_id = part
            if idx + 1 < len(parts) and re.fullmatch(r"R\d+-\d{4}-\d{2}-\d{2}", parts[idx + 1]):
                session_id = parts[idx + 1]
                m = re.search(r"(\d{4}-\d{2}-\d{2})", session_id)
                session_date = m.group(1) if m else ""
            break

    if not subject_id:
        m = re.search(r"(R\d+)", path.name)
        if m:
            subject_id = m.group(1)

    if not session_id:
        m = re.search(r"(R\d+-\d{4}-\d{2}-\d{2})", path.name)
        if m:
            session_id = m.group(1)
            m_date = re.search(r"(\d{4}-\d{2}-\d{2})", session_id)
            session_date = m_date.group(1) if m_date else ""

    return subject_id, session_id, session_date


def build_file_inventory(
    root: Path,
    *,
    dataset_id: str = DATASET_ID_DEFAULT,
) -> pd.DataFrame:
    root = Path(root).resolve()
    data_roots = discover_data_roots(root)

    rows: list[dict[str, Any]] = []

    for data_root in data_roots:
        version_label = _version_label_for_path(data_root, root)

        for path in sorted(data_root.rglob("*")):
            if not path.is_file():
                continue

            file_kind = classify_redish_file(path)
            subject_id, session_id, session_date = _extract_subject_session(path, data_root)

            try:
                relative_path = str(path.relative_to(data_root))
            except Exception:
                relative_path = str(path)

            rows.append(
                {
                    "dataset_id": dataset_id,
                    "version_label": version_label,
                    "data_root": str(data_root),
                    "relative_path": relative_path,
                    "file_path": str(path),
                    "file_name": path.name,
                    "suffix": path.suffix,
                    "file_kind": file_kind,
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "session_date": session_date,
                    "size_bytes": path.stat().st_size,
                    "inspect_priority": _inspect_priority(file_kind),
                }
            )

    return pd.DataFrame(rows)


def _inspect_priority(file_kind: str) -> int:
    priorities = {
        "aggregate_idphi": 1,
        "aggregate_lapdata_behavior": 1,
        "aggregate_sessiondata": 1,
        "data_definition": 1,
        "session_rrow_behavior": 2,
        "session_tracking_vt": 2,
        "session_keys": 2,
        "unit_spike_times": 4,
        "unit_waveform": 5,
        "analysis_mat": 6,
        "other_mat": 7,
    }
    return priorities.get(file_kind, 9)


def build_session_matrix(inventory: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty:
        return pd.DataFrame(
            columns=[
                "dataset_id",
                "version_label",
                "data_root",
                "subject_id",
                "session_id",
                "session_date",
                "has_rrow_behavior",
                "has_tracking_vt",
                "has_keys",
                "n_unit_spike_files",
                "n_unit_waveform_files",
            ]
        )

    session_files = inventory.loc[inventory["session_id"].astype(str) != ""].copy()
    if session_files.empty:
        return pd.DataFrame()

    grouped = []
    for keys, group in session_files.groupby(["dataset_id", "version_label", "data_root", "subject_id", "session_id", "session_date"], dropna=False):
        kinds = set(group["file_kind"].astype(str))
        grouped.append(
            {
                "dataset_id": keys[0],
                "version_label": keys[1],
                "data_root": keys[2],
                "subject_id": keys[3],
                "session_id": keys[4],
                "session_date": keys[5],
                "has_rrow_behavior": "session_rrow_behavior" in kinds,
                "has_tracking_vt": "session_tracking_vt" in kinds,
                "has_keys": "session_keys" in kinds,
                "n_unit_spike_files": int((group["file_kind"] == "unit_spike_times").sum()),
                "n_unit_waveform_files": int((group["file_kind"] == "unit_waveform").sum()),
            }
        )

    return pd.DataFrame(grouped).sort_values(["version_label", "subject_id", "session_id"]).reset_index(drop=True)


def build_aggregate_file_matrix(inventory: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty:
        return pd.DataFrame()

    key_kinds = {
        "aggregate_idphi",
        "aggregate_lapdata_behavior",
        "aggregate_sessiondata",
        "data_definition",
    }

    rows = []
    for keys, group in inventory.groupby(["dataset_id", "version_label", "data_root"], dropna=False):
        kinds = set(group["file_kind"].astype(str))
        rows.append(
            {
                "dataset_id": keys[0],
                "version_label": keys[1],
                "data_root": keys[2],
                "has_idphi_rrow": "aggregate_idphi" in kinds,
                "has_lapdata_behav_rrow": "aggregate_lapdata_behavior" in kinds,
                "has_sessiondata_rrow": "aggregate_sessiondata" in kinds,
                "has_datadef_rrow": "data_definition" in kinds,
                "n_key_aggregate_files": int(group["file_kind"].isin(key_kinds).sum()),
                "n_session_rrow_files": int((group["file_kind"] == "session_rrow_behavior").sum()),
                "n_tracking_vt_files": int((group["file_kind"] == "session_tracking_vt").sum()),
                "n_subjects": int(group.loc[group["subject_id"].astype(str) != "", "subject_id"].nunique()),
                "n_sessions": int(group.loc[group["session_id"].astype(str) != "", "session_id"].nunique()),
            }
        )

    return pd.DataFrame(rows).sort_values(["version_label", "data_root"]).reset_index(drop=True)


def write_inventory_outputs(
    root: Path,
    output_dir: Path,
    *,
    dataset_id: str = DATASET_ID_DEFAULT,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory = build_file_inventory(root, dataset_id=dataset_id)
    session_matrix = build_session_matrix(inventory)
    aggregate_matrix = build_aggregate_file_matrix(inventory)

    inventory_path = output_dir / "Table_Redish_RRow_File_Inventory.csv"
    session_matrix_path = output_dir / "Table_Redish_RRow_Session_File_Matrix.csv"
    aggregate_matrix_path = output_dir / "Table_Redish_RRow_Aggregate_File_Matrix.csv"
    report_path = output_dir / "Redish_RRow_Stage3_2C_Readiness_Report.md"
    meta_path = output_dir / "redish_rrow_inventory_meta.json"

    inventory.to_csv(inventory_path, index=False)
    session_matrix.to_csv(session_matrix_path, index=False)
    aggregate_matrix.to_csv(aggregate_matrix_path, index=False)

    kind_counts = (
        inventory["file_kind"].value_counts().rename_axis("file_kind").reset_index(name="count")
        if not inventory.empty
        else pd.DataFrame(columns=["file_kind", "count"])
    )

    meta = {
        "dataset_id": dataset_id,
        "root": str(Path(root).resolve()),
        "n_data_roots": len(discover_data_roots(Path(root))),
        "n_files": int(len(inventory)),
        "n_subjects": int(inventory.loc[inventory["subject_id"].astype(str) != "", "subject_id"].nunique()) if not inventory.empty else 0,
        "n_sessions": int(inventory.loc[inventory["session_id"].astype(str) != "", "session_id"].nunique()) if not inventory.empty else 0,
        "file_kind_counts": dict(zip(kind_counts["file_kind"], kind_counts["count"])),
        "inventory_path": str(inventory_path),
        "session_matrix_path": str(session_matrix_path),
        "aggregate_matrix_path": str(aggregate_matrix_path),
        "report_path": str(report_path),
    }

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    report = _make_readiness_report(meta, aggregate_matrix, session_matrix)
    report_path.write_text(report, encoding="utf-8")

    print(f"File inventory saved: {inventory_path}")
    print(f"Session matrix saved: {session_matrix_path}")
    print(f"Aggregate matrix saved: {aggregate_matrix_path}")
    print(f"Report saved: {report_path}")
    print(f"Metadata saved: {meta_path}")

    return meta


def _make_readiness_report(meta: dict[str, Any], aggregate_matrix: pd.DataFrame, session_matrix: pd.DataFrame) -> str:
    lines = [
        "# Redish RRow 2022 Stage 3.2C readiness report",
        "",
        "Patch 16A inventory report.",
        "",
        "## Scope",
        "",
        "This report checks whether the extracted Redish Restaurant Row dataset contains the minimum file layers needed for biological comparability:",
        "",
        "- processed trial/choice behavior;",
        "- processed IdPhi or VTE-like behavioral metric;",
        "- per-session tracking;",
        "- session/subject metadata;",
        "- optional unit-level physiology for later neural proxy work.",
        "",
        "## Summary",
        "",
        f"- Dataset ID: `{meta['dataset_id']}`",
        f"- Root: `{meta['root']}`",
        f"- Data roots found: `{meta['n_data_roots']}`",
        f"- Files indexed: `{meta['n_files']}`",
        f"- Subjects detected: `{meta['n_subjects']}`",
        f"- Sessions detected: `{meta['n_sessions']}`",
        "",
        "## Aggregate files",
        "",
    ]

    if aggregate_matrix.empty:
        lines.append("No aggregate matrix was produced.")
    else:
        lines.append(aggregate_matrix.to_markdown(index=False))

    lines.extend(
        [
            "",
            "## Session coverage",
            "",
        ]
    )

    if session_matrix.empty:
        lines.append("No per-session behavior/tracking files were detected.")
    else:
        coverage = {
            "sessions_with_rrow_behavior": int(session_matrix["has_rrow_behavior"].sum()),
            "sessions_with_tracking_vt": int(session_matrix["has_tracking_vt"].sum()),
            "sessions_with_keys": int(session_matrix["has_keys"].sum()),
        }
        for key, value in coverage.items():
            lines.append(f"- {key}: `{value}`")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Use this inventory only as a structural precheck. A positive inventory does not yet establish Stage 3.2C comparability. The next step is `probe_processed_behavior.py`, which inspects MATLAB field names and shapes to confirm whether `choice context -> choice/outcome -> IdPhi/deliberation proxy` can be extracted without changing the frozen VTE wrapper.",
            "",
        ]
    )

    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory extracted Redish RRow 2022 dataset files.")
    parser.add_argument("--root", required=True, help="Root directory containing Version 001/002 or a Data directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-id", default=DATASET_ID_DEFAULT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    write_inventory_outputs(
        root=Path(args.root),
        output_dir=Path(args.output_dir),
        dataset_id=args.dataset_id,
    )


if __name__ == "__main__":
    main()