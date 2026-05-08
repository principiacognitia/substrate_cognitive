from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import re

from .inventory_redish_rrow import (
    DATASET_ID_DEFAULT,
    build_file_inventory,
    discover_data_roots,
)
from .mat_probe import summarize_mat_file


CONCEPT_TOKEN_GROUPS: dict[str, list[tuple[str, ...]]] = {
    "idphi_or_vte_proxy": [
        ("idphi",),
        ("id", "phi"),
        ("dphi",),
        ("d", "phi"),
        ("avg", "dphi"),
        ("avg", "d", "phi"),
        ("vte",),
    ],
    "trial_or_lap": [
        ("trial",),
        ("trials",),
        ("lap",),
        ("laps",),
        ("current", "lap"),
        ("visit",),
        ("visits",),
        ("pass",),
    ],
    "zone_or_choice_point": [
        ("zone",),
        ("zones",),
        ("offer", "zone"),
        ("wait", "zone"),
        ("linger", "zone"),
        ("advance", "zone"),
        ("site",),
        ("siterank",),
        ("entering", "zone", "time"),
        ("exit", "zone", "time"),
        ("total", "site", "time"),
    ],
    "choice": [
        ("choice",),
        ("decision",),
        ("accept", "offer"),
        ("skip", "offer"),
        ("quit", "offer"),
        ("accept",),
        ("reject",),
        ("skip",),
        ("quit",),
        ("chosen",),
    ],
    "reward_or_outcome": [
        ("reward",),
        ("rewards",),
        ("outcome",),
        ("food", "received"),
        ("earn", "offer"),
        ("earned",),
        ("pellet",),
        ("pellets",),
        ("total", "pellets"),
    ],
    "delay_or_cost": [
        ("delay",),
        ("delays",),
        ("zone", "delay"),
        ("cost",),
        ("price",),
        ("value",),
        ("offer", "value"),
    ],
    "subject_or_session": [
        ("rat",),
        ("subject",),
        ("animal",),
        ("session",),
        ("sessions",),
        ("ssn",),
        ("date",),
        ("directory",),
        ("exp", "type"),
    ],
    "tracking_or_pose": [
        ("vt",),
        ("track",),
        ("tracking",),
        ("pos",),
        ("position",),
        ("x",),
        ("y",),
        ("xpos",),
        ("ypos",),
        ("xloc",),
        ("yloc",),
        ("time",),
        ("timestamp",),
        ("timestamps",),
        ("velocity",),
        ("speed",),
        ("heading",),
    ],
    "neural_optional": [
        ("spike",),
        ("spikes",),
        ("unit",),
        ("units",),
        ("cell",),
        ("cells",),
        ("peth",),
        ("theta",),
        ("rate",),
        ("lfp",),
        ("mutual", "info"),
        ("transfer", "entropy"),
    ],
}


PRIORITY_FILE_KINDS = {
    "aggregate_idphi",
    "aggregate_lapdata_behavior",
    "aggregate_sessiondata",
    "data_definition",
    "session_rrow_behavior",
    "session_tracking_vt",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _tokens(text: str) -> set[str]:
    text = _CAMEL_BOUNDARY.sub("_", text)
    raw = re.split(r"[^A-Za-z0-9]+", text.lower())
    return {token for token in raw if token}


def _concept_matches(field_path: str, file_name: str) -> list[str]:
    token_set = _tokens(f"{field_path} {file_name}")

    matches = []
    for concept, token_groups in CONCEPT_TOKEN_GROUPS.items():
        for group in token_groups:
            if set(group).issubset(token_set):
                matches.append(concept)
                break

    return matches


def _select_probe_files(
    inventory: pd.DataFrame,
    *,
    version_label_contains: str | None = None,
    max_session_files_per_kind: int = 3,
) -> list[Path]:
    if inventory.empty:
        return []

    data = inventory.copy()

    if version_label_contains:
        data = data.loc[data["version_label"].astype(str).str.contains(version_label_contains, case=False, regex=False)]

    priority = data.loc[data["file_kind"].isin(PRIORITY_FILE_KINDS)].copy()

    selected_rows = []

    aggregate = priority.loc[
        priority["file_kind"].isin(
            {
                "aggregate_idphi",
                "aggregate_lapdata_behavior",
                "aggregate_sessiondata",
                "data_definition",
            }
        )
    ].sort_values(["inspect_priority", "file_name"])

    selected_rows.extend(aggregate.to_dict("records"))

    for file_kind in ["session_rrow_behavior", "session_tracking_vt"]:
        subset = priority.loc[priority["file_kind"] == file_kind].sort_values(
            ["version_label", "subject_id", "session_id", "file_name"]
        )
        selected_rows.extend(subset.head(max_session_files_per_kind).to_dict("records"))

    files = []
    seen = set()
    for row in selected_rows:
        path = Path(row["file_path"])
        if path.exists() and path not in seen:
            files.append(path)
            seen.add(path)

    return files


def build_behavior_endpoint_precheck(structure: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if structure.empty:
        return pd.DataFrame(
            columns=[
                "concept",
                "n_matching_fields",
                "n_source_files",
                "candidate_field_paths",
                "candidate_files",
            ]
        )

    for concept in CONCEPT_TOKEN_GROUPS:
        matches = []
        files = set()

        for _, row in structure.iterrows():
            field_path = str(row.get("field_path", ""))
            file_name = Path(str(row.get("source_file", ""))).name
            concepts = _concept_matches(field_path, file_name)
            if concept in concepts:
                matches.append(field_path)
                files.add(str(row.get("source_file", "")))

        rows.append(
            {
                "concept": concept,
                "n_matching_fields": len(matches),
                "n_source_files": len(files),
                "candidate_field_paths": "; ".join(matches[:25]),
                "candidate_files": "; ".join(sorted(Path(f).name for f in files)[:25]),
            }
        )

    return pd.DataFrame(rows)


def write_processed_behavior_probe_outputs(
    root: Path,
    output_dir: Path,
    *,
    dataset_id: str = DATASET_ID_DEFAULT,
    version_label_contains: str | None = "Version 002",
    max_depth: int = 5,
    max_rows_per_file: int = 5000,
    max_object_items: int = 20,
    max_session_files_per_kind: int = 3,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory = build_file_inventory(root, dataset_id=dataset_id)
    probe_files = _select_probe_files(
        inventory,
        version_label_contains=version_label_contains,
        max_session_files_per_kind=max_session_files_per_kind,
    )

    frames: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []

    for path in probe_files:
        try:
            frame = summarize_mat_file(
                path,
                max_depth=max_depth,
                max_rows=max_rows_per_file,
                max_object_items=max_object_items,
            )
            frames.append(frame)
        except Exception as exc:
            errors.append({"source_file": str(path), "error": repr(exc)})

    if frames:
        structure = pd.concat(frames, ignore_index=True)
    else:
        structure = pd.DataFrame(
            columns=[
                "source_file",
                "field_path",
                "depth",
                "kind",
                "python_type",
                "shape",
                "dtype",
                "size",
                "n_fields",
                "numeric_min",
                "numeric_max",
                "numeric_mean",
                "n_finite",
            ]
        )

    precheck = build_behavior_endpoint_precheck(structure)

    structure_path = output_dir / "Table_Redish_RRow_Mat_Structure_Probe.csv"
    precheck_path = output_dir / "Table_Redish_RRow_Behavior_Endpoint_Precheck.csv"
    selected_files_path = output_dir / "Table_Redish_RRow_Probe_Selected_Files.csv"
    errors_path = output_dir / "Table_Redish_RRow_Probe_Errors.csv"
    meta_path = output_dir / "redish_rrow_processed_behavior_probe_meta.json"
    report_path = output_dir / "Redish_RRow_Processed_Behavior_Probe_Report.md"

    structure.to_csv(structure_path, index=False)
    precheck.to_csv(precheck_path, index=False)

    selected = pd.DataFrame({"source_file": [str(p) for p in probe_files]})
    selected["file_name"] = selected["source_file"].map(lambda x: Path(x).name)
    selected.to_csv(selected_files_path, index=False)

    if errors:
        pd.DataFrame(errors).to_csv(errors_path, index=False)
        errors_path_str = str(errors_path)
    else:
        errors_path_str = None

    readiness = _estimate_readiness(precheck)

    meta = {
        "dataset_id": dataset_id,
        "root": str(Path(root).resolve()),
        "version_label_contains": version_label_contains,
        "n_data_roots": len(discover_data_roots(Path(root))),
        "n_probe_files": len(probe_files),
        "n_successful_probe_files": len(frames),
        "n_failed_probe_files": len(errors),
        "n_structure_rows": int(len(structure)),
        "readiness": readiness,
        "structure_path": str(structure_path),
        "precheck_path": str(precheck_path),
        "selected_files_path": str(selected_files_path),
        "errors_path": errors_path_str,
        "report_path": str(report_path),
    }

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    report_path.write_text(_make_report(meta, precheck), encoding="utf-8")

    print(f"MAT structure probe saved: {structure_path}")
    print(f"Behavior endpoint precheck saved: {precheck_path}")
    print(f"Selected probe files saved: {selected_files_path}")
    if errors_path_str:
        print(f"Probe errors saved: {errors_path_str}")
    print(f"Metadata saved: {meta_path}")
    print(f"Report saved: {report_path}")

    return meta


def _estimate_readiness(precheck: pd.DataFrame) -> str:
    if precheck.empty:
        return "insufficient_no_fields"

    counts = {
        row["concept"]: int(row["n_matching_fields"])
        for _, row in precheck.iterrows()
    }

    required = [
        "idphi_or_vte_proxy",
        "trial_or_lap",
        "zone_or_choice_point",
        "choice",
        "reward_or_outcome",
        "subject_or_session",
    ]

    missing = [concept for concept in required if counts.get(concept, 0) <= 0]

    if not missing:
        return "likely_ready_for_patch_16b_adapter_design"

    if "idphi_or_vte_proxy" not in missing and "choice" not in missing:
        return "partially_ready_needs_field_mapping"

    return "not_ready_from_probe_only"


def _make_report(meta: dict[str, Any], precheck: pd.DataFrame) -> str:
    lines = [
        "# Redish RRow processed behavior probe report",
        "",
        "Patch 16A output.",
        "",
        "## Purpose",
        "",
        "This probe checks whether Redish RRow processed MATLAB files expose enough behavioral fields for Stage 3.2C comparability. It does not yet convert data into canonical VTE trace format.",
        "",
        "The target endpoint is not generic head movement similarity. The target is a comparable decision-layer record: choice context, choice/outcome, and an IdPhi or deliberation proxy around a decision zone.",
        "",
        "## Summary",
        "",
        f"- Dataset ID: `{meta['dataset_id']}`",
        f"- Root: `{meta['root']}`",
        f"- Version filter: `{meta['version_label_contains']}`",
        f"- Probe files: `{meta['n_probe_files']}`",
        f"- Successful files: `{meta['n_successful_probe_files']}`",
        f"- Failed files: `{meta['n_failed_probe_files']}`",
        f"- Structure rows: `{meta['n_structure_rows']}`",
        f"- Readiness estimate: `{meta['readiness']}`",
        "",
        "## Concept precheck",
        "",
    ]

    if precheck.empty:
        lines.append("No concept precheck rows were produced.")
    else:
        table = precheck[["concept", "n_matching_fields", "n_source_files", "candidate_files"]].copy()
        lines.append(table.to_markdown(index=False))

    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "A positive keyword precheck is not proof of usability. It only identifies candidate fields. Patch 16B should inspect the candidate field contents and build a frozen field mapping. The mapping must preserve the original Redish variables and produce a separate canonical comparison table rather than rewriting the frozen Stage 3.2 wrapper.",
            "",
        ]
    )

    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe processed behavior files in Redish RRow 2022.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-id", default=DATASET_ID_DEFAULT)
    parser.add_argument("--version-label-contains", default="Version 002")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-rows-per-file", type=int, default=5000)
    parser.add_argument("--max-object-items", type=int, default=20)
    parser.add_argument("--max-session-files-per-kind", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    write_processed_behavior_probe_outputs(
        root=Path(args.root),
        output_dir=Path(args.output_dir),
        dataset_id=args.dataset_id,
        version_label_contains=args.version_label_contains,
        max_depth=args.max_depth,
        max_rows_per_file=args.max_rows_per_file,
        max_object_items=args.max_object_items,
        max_session_files_per_kind=args.max_session_files_per_kind,
    )


if __name__ == "__main__":
    main()