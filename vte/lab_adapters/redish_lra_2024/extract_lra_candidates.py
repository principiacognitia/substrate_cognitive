from __future__ import annotations

import argparse
import csv
import json
import math
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
from scipy.io import loadmat

DATASET_ID = "redish_lra_2024"

TABLE_FILES = {
    "session": [("lra", "LRA/SessionData_LRA.mat"), ("mpfc_dreadds", "mPFC-DREADDs/SessionData_LRA_DREADDs.mat")],
    "lapdata": [("lra", "LRA/LapData_Behav_LRA.mat"), ("mpfc_dreadds", "mPFC-DREADDs/LapData_Behav_LRA_DREADDs.mat")],
    "idphi": [("lra", "LRA/IdPhiData_LRA.mat"), ("mpfc_dreadds", "mPFC-DREADDs/IdPhiData_LRA_DREADDs.mat")],
    "changepoint": [("lra", "LRA/ChangePointBehavAll_LRA.mat"), ("mpfc_dreadds", "mPFC-DREADDs/ChangePointBehav_LRA_DREADDs.mat")],
    "stereotypy": [("lra", "LRA/StereotypyData_LRA.mat"), ("mpfc_dreadds", "mPFC-DREADDs/StereotypyData_LRA_DREADDs.mat")],
    "rail_stereotypy": [("lra", "LRA/RailStereotypyData_LRA.mat")],
    "complexity": [("lra", "LRA/ComplexityVals_LRA.mat")],
}

FIELD_CLASSES = {
    "idphi": [r"idphi", r"avgdphi", r"dphi", r"vte", r"zlog"],
    "trial_lap": [r"\blap\b", r"laps", r"trial", r"currentlap", r"lapnumber", r"pass"],
    "left_right_choice": [r"choice", r"decision", r"turn", r"path", r"left", r"right", r"arm", r"goal", r"route"],
    "reward_outcome": [r"reward", r"correct", r"error", r"outcome", r"food", r"pellet", r"pump"],
    "rule_contingency": [r"rule", r"conting", r"state", r"strategy", r"alternat", r"task", r"condition", r"epoch", r"block"],
    "switch_changepoint": [r"change", r"switch", r"transition", r"changepoint", r"reversal", r"cp"],
    "subject_session": [r"subject", r"rat", r"animal", r"session", r"date", r"ssn", r"directory"],
    "stereotypy": [r"stereotyp", r"rail", r"path", r"variab", r"complex", r"entropy"],
    "treatment": [r"dreadd", r"cno", r"dcz", r"vehicle", r"veh", r"drug", r"injection", r"treatment"],
}

SESSION_CLASSES = {"subject_session", "rule_contingency", "treatment", "switch_changepoint"}


@dataclass
class CandidateValue:
    cohort: str
    source_table: str
    source_file: str
    mat_variable: str
    field_path: str
    field_class: str
    session_index: int | None
    event_index: int | None
    value: Any


def find_processed_root(root: Path) -> Path:
    for p in [root / "Processed Data" / "Processed Data", root / "Processed Data", root]:
        if (p / "LRA").exists() or (p / "mPFC-DREADDs").exists():
            return p
    raise FileNotFoundError("Could not find Processed Data/Processed Data with LRA or mPFC-DREADDs")


def load_mat(path: Path) -> tuple[dict[str, Any] | None, str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            data = loadmat(path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
        except MemoryError:
            return None, "MemoryError"
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
    msg = "; ".join(f"{type(w.message).__name__}: {w.message}" for w in caught)
    return data, msg


def clean_scalar(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return ""
    return v


def value_to_text(v: Any, max_len: int = 220) -> str:
    v = clean_scalar(v)
    if v is None:
        return ""
    if isinstance(v, np.ndarray):
        flat = v.ravel()
        if flat.size == 1:
            return value_to_text(flat[0], max_len=max_len)
        text = "[" + ", ".join(value_to_text(x, 40) for x in flat[:12])
        if flat.size > 12:
            text += ", ..."
        text += "]"
    elif isinstance(v, (list, tuple)):
        if len(v) == 1:
            return value_to_text(v[0], max_len=max_len)
        text = "[" + ", ".join(value_to_text(x, 40) for x in v[:12])
        if len(v) > 12:
            text += ", ..."
        text += "]"
    elif isinstance(v, dict):
        text = "{" + ", ".join(str(k) for k in list(v.keys())[:12]) + "}"
    elif hasattr(v, "__dict__"):
        keys = [k for k in vars(v) if not k.startswith("_")]
        text = "{" + ", ".join(keys[:12]) + "}"
    else:
        text = str(v)
    text = text.replace("\r", " ").replace("\n", " ")
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def is_scalar_like(v: Any) -> bool:
    if v is None or isinstance(v, (str, bytes, int, float, bool, np.generic)):
        return True
    if isinstance(v, np.ndarray):
        return v.ndim == 0 or v.size == 1
    return False


def as_list(v: Any) -> list[Any]:
    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            return [v.item()]
        return list(v.ravel())
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def is_leaf_vector(v: Any) -> bool:
    if is_scalar_like(v):
        return True
    if isinstance(v, np.ndarray):
        if v.dtype == object:
            flat = v.ravel()
            return all(is_scalar_like(x) for x in flat[: min(20, flat.size)])
        return v.ndim <= 2
    if isinstance(v, (list, tuple)):
        return all(is_scalar_like(x) for x in v[: min(20, len(v))])
    return False


def iter_fields(v: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(v, dict):
        for k, child in v.items():
            if not str(k).startswith("__"):
                yield str(k), child
    elif hasattr(v, "__dict__") and not isinstance(v, (str, bytes)):
        for k, child in vars(v).items():
            if not str(k).startswith("_"):
                yield str(k), child


def field_classes(path: str) -> list[str]:
    s = path.lower()
    return [cls for cls, pats in FIELD_CLASSES.items() if any(re.search(p, s) for p in pats)]


def walk_leaves(v: Any, base_path: str, max_depth: int, depth: int = 0) -> Iterator[tuple[str, Any, list[str]]]:
    classes = field_classes(base_path)
    if classes and is_leaf_vector(v):
        yield base_path, v, classes
        return
    if depth >= max_depth:
        if classes:
            yield base_path, v, classes
        return
    yielded_child = False
    for k, child in iter_fields(v):
        yielded_child = True
        yield from walk_leaves(child, f"{base_path}.{k}" if base_path else k, max_depth, depth + 1)
    if yielded_child:
        return
    if isinstance(v, np.ndarray) and v.dtype == object:
        for i, child in enumerate(v.ravel()[:20]):
            if isinstance(child, (dict, list, tuple, np.ndarray)) or hasattr(child, "__dict__"):
                yield from walk_leaves(child, f"{base_path}.[{i}]", max_depth, depth + 1)
    elif isinstance(v, (list, tuple)):
        for i, child in enumerate(v[:20]):
            if isinstance(child, (dict, list, tuple, np.ndarray)) or hasattr(child, "__dict__"):
                yield from walk_leaves(child, f"{base_path}.[{i}]", max_depth, depth + 1)


def split_session_vectors(v: Any) -> list[list[Any]]:
    if isinstance(v, np.ndarray):
        if v.dtype == object:
            flat = list(v.ravel())
            if any(not is_scalar_like(x) for x in flat):
                return [as_list(x) for x in flat]
            return [flat]
        if v.ndim == 0:
            return [[v.item()]]
        if v.ndim == 1:
            return [list(v.ravel())]
        if v.ndim == 2:
            return [list(row.ravel()) for row in v]
        return [list(v.ravel())]
    if isinstance(v, (list, tuple)):
        if any(not is_scalar_like(x) for x in v):
            return [as_list(x) for x in v]
        return [list(v)]
    return [[v]]


def iter_candidate_values(path: Path, cohort: str, source_table: str, max_depth: int) -> Iterator[CandidateValue]:
    data, warn = load_mat(path)
    if data is None:
        return
    for mat_variable, obj in data.items():
        if mat_variable.startswith("__"):
            continue
        for fp, raw, classes in walk_leaves(obj, mat_variable, max_depth):
            sessions = split_session_vectors(raw)
            for cls in classes:
                for si, vec in enumerate(sessions):
                    if len(vec) == 1:
                        yield CandidateValue(cohort, source_table, str(path), mat_variable, fp, cls, si, None, vec[0])
                    else:
                        for ei, val in enumerate(vec):
                            yield CandidateValue(cohort, source_table, str(path), mat_variable, fp, cls, si, ei, val)


def candidate_to_row(c: CandidateValue) -> dict[str, Any]:
    return {
        "dataset_id": DATASET_ID,
        "cohort": c.cohort,
        "source_table": c.source_table,
        "source_file": c.source_file,
        "mat_variable": c.mat_variable,
        "field_path": c.field_path,
        "field_class": c.field_class,
        "session_index": "" if c.session_index is None else c.session_index,
        "event_index": "" if c.event_index is None else c.event_index,
        "value": value_to_text(c.value),
    }


def write_candidate_csv(processed_root: Path, tables: set[str], out_path: Path, class_filter: set[str] | None, max_depth: int) -> tuple[int, dict[tuple[str, str, str, str, str], dict[str, Any]]]:
    fields = ["dataset_id", "cohort", "source_table", "source_file", "mat_variable", "field_path", "field_class", "session_index", "event_index", "value"]
    summary: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    n = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for table, entries in TABLE_FILES.items():
            if table not in tables:
                continue
            for cohort, rel in entries:
                path = processed_root / rel
                if not path.exists():
                    continue
                for c in iter_candidate_values(path, cohort, table, max_depth):
                    if class_filter is not None and c.field_class not in class_filter:
                        continue
                    row = candidate_to_row(c)
                    w.writerow(row)
                    n += 1
                    key = (c.cohort, c.source_table, c.mat_variable, c.field_path, c.field_class)
                    s = summary.setdefault(key, {"dataset_id": DATASET_ID, "cohort": c.cohort, "source_table": c.source_table, "mat_variable": c.mat_variable, "field_path": c.field_path, "field_class": c.field_class, "n_values": 0, "n_nonempty": 0, "sessions": set(), "min_event_index": None, "max_event_index": None, "sample_values": []})
                    s["n_values"] += 1
                    if row["value"] != "":
                        s["n_nonempty"] += 1
                        if len(s["sample_values"]) < 5:
                            s["sample_values"].append(row["value"])
                    if c.session_index is not None:
                        s["sessions"].add(c.session_index)
                    if c.event_index is not None:
                        s["min_event_index"] = c.event_index if s["min_event_index"] is None else min(s["min_event_index"], c.event_index)
                        s["max_event_index"] = c.event_index if s["max_event_index"] is None else max(s["max_event_index"], c.event_index)
    return n, summary


def summary_rows(summaries: list[dict[tuple[str, str, str, str, str], dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        for s in summary.values():
            rows.append({
                "dataset_id": DATASET_ID,
                "cohort": s["cohort"],
                "source_table": s["source_table"],
                "mat_variable": s["mat_variable"],
                "field_path": s["field_path"],
                "field_class": s["field_class"],
                "n_values": s["n_values"],
                "n_nonempty": s["n_nonempty"],
                "coverage": s["n_nonempty"] / s["n_values"] if s["n_values"] else "",
                "n_sessions": len(s["sessions"]),
                "min_event_index": "" if s["min_event_index"] is None else s["min_event_index"],
                "max_event_index": "" if s["max_event_index"] is None else s["max_event_index"],
                "sample_values": "; ".join(s["sample_values"]),
            })
    return sorted(rows, key=lambda r: (r["cohort"], r["source_table"], r["field_class"], r["field_path"]))


def choose_best_field(df: pd.DataFrame, regexes: list[str]) -> str | None:
    if df.empty:
        return None
    best = None
    for fp, g in df.groupby("field_path", dropna=False):
        nonempty = (g["value"].astype(str) != "").sum()
        n_sessions = g["session_index"].nunique()
        events = (g["event_index"].astype(str) != "").sum()
        score = int(nonempty + events + n_sessions * 10)
        low = str(fp).lower()
        for i, rx in enumerate(regexes):
            if re.search(rx, low):
                score += (len(regexes) - i) * 100000
        if best is None or score > best[0]:
            best = (score, str(fp))
    return None if best is None else best[1]


def pivot_field(df: pd.DataFrame, field_path: str | None, out_name: str) -> pd.DataFrame:
    if df.empty or field_path is None:
        return pd.DataFrame(columns=["cohort", "session_index", "event_index", out_name])
    sub = df[df["field_path"] == field_path][["cohort", "session_index", "event_index", "value"]].copy()
    sub = sub[sub["event_index"].astype(str) != ""]
    sub = sub.drop_duplicates(["cohort", "session_index", "event_index"], keep="first")
    return sub.rename(columns={"value": out_name})


def build_endpoint_draft(idphi_path: Path, lap_path: Path, out_path: Path) -> tuple[int, dict[str, str | None]]:
    idphi = pd.read_csv(idphi_path, dtype=str, keep_default_na=False) if idphi_path.exists() else pd.DataFrame()
    lap = pd.read_csv(lap_path, dtype=str, keep_default_na=False) if lap_path.exists() else pd.DataFrame()
    chosen = {
        "idphi": choose_best_field(idphi[idphi["field_class"] == "idphi"], [r"\bidphi\b", r"avgdphi", r"dphi", r"vte"]) if not idphi.empty else None,
        "lap": choose_best_field(lap[lap["field_class"] == "trial_lap"], [r"\blap\b", r"currentlap", r"trial"]) if not lap.empty else None,
        "choice": choose_best_field(lap[lap["field_class"] == "left_right_choice"], [r"choice", r"decision", r"turn", r"path", r"goal"]) if not lap.empty else None,
        "reward": choose_best_field(lap[lap["field_class"] == "reward_outcome"], [r"reward", r"correct", r"outcome", r"error"]) if not lap.empty else None,
        "rule": choose_best_field(lap[lap["field_class"] == "rule_contingency"], [r"conting", r"rule", r"state", r"task", r"alternat"]) if not lap.empty else None,
        "stereotypy": choose_best_field(lap[lap["field_class"] == "stereotypy"], [r"stereotyp", r"rail", r"variab"]) if not lap.empty else None,
    }
    frames = [
        pivot_field(idphi, chosen["idphi"], "lab_idphi_or_dphi"),
        pivot_field(lap, chosen["lap"], "lap_or_trial_raw"),
        pivot_field(lap, chosen["choice"], "choice_raw"),
        pivot_field(lap, chosen["reward"], "reward_raw"),
        pivot_field(lap, chosen["rule"], "rule_raw"),
        pivot_field(lap, chosen["stereotypy"], "stereotypy_raw"),
    ]
    frames = [x for x in frames if not x.empty]
    if not frames:
        pd.DataFrame().to_csv(out_path, index=False)
        return 0, chosen
    frames.sort(key=len, reverse=True)
    endpoint = frames[0]
    for frame in frames[1:]:
        endpoint = endpoint.merge(frame, on=["cohort", "session_index", "event_index"], how="left")
    endpoint.insert(0, "dataset_id", DATASET_ID)
    endpoint["decision_stage"] = "choice_point"
    endpoint["chosen_action"] = endpoint.get("choice_raw", "")
    endpoint["reward"] = endpoint.get("reward_raw", "")
    endpoint["source_note"] = "draft_alignment_by_session_event_index_not_canonical"
    endpoint.to_csv(out_path, index=False)
    return len(endpoint), chosen


def extract_redish_lra_candidates(root: Path, output_dir: Path, max_depth: int = 6) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_root = find_processed_root(root)

    session_path = output_dir / "Table_Redish_LRA_Session_Metadata_Candidates.csv"
    idphi_path = output_dir / "Table_Redish_LRA_IdPhi_Candidates_Long.csv"
    lap_path = output_dir / "Table_Redish_LRA_LapData_Candidates_Long.csv"
    cp_path = output_dir / "Table_Redish_LRA_ChangePoint_Candidates_Long.csv"
    summary_path = output_dir / "Table_Redish_LRA_Field_Alignment_Summary.csv"
    endpoint_path = output_dir / "Table_Redish_LRA_Endpoint_Draft.csv"
    meta_path = output_dir / "redish_lra_2024_candidate_extraction_meta.json"
    report_path = output_dir / "Redish_LRA_Candidate_Extraction_Report.md"

    n_session, s1 = write_candidate_csv(processed_root, {"session"}, session_path, SESSION_CLASSES, max_depth)
    n_idphi, s2 = write_candidate_csv(processed_root, {"idphi"}, idphi_path, {"idphi", "trial_lap", "subject_session", "treatment"}, max_depth)
    n_lap, s3 = write_candidate_csv(processed_root, {"lapdata", "stereotypy", "rail_stereotypy", "complexity"}, lap_path, {"trial_lap", "left_right_choice", "reward_outcome", "rule_contingency", "stereotypy", "treatment", "subject_session"}, max_depth)
    n_cp, s4 = write_candidate_csv(processed_root, {"changepoint"}, cp_path, {"switch_changepoint", "rule_contingency", "trial_lap", "subject_session", "treatment"}, max_depth)
    pd.DataFrame(summary_rows([s1, s2, s3, s4])).to_csv(summary_path, index=False)
    n_endpoint, chosen = build_endpoint_draft(idphi_path, lap_path, endpoint_path)

    meta = {
        "dataset_id": DATASET_ID,
        "root": str(root),
        "processed_root": str(processed_root),
        "max_depth": max_depth,
        "n_session_candidate_rows": n_session,
        "n_idphi_candidate_rows": n_idphi,
        "n_lap_candidate_rows": n_lap,
        "n_changepoint_candidate_rows": n_cp,
        "n_endpoint_draft_rows": n_endpoint,
        "endpoint_draft_chosen_fields": chosen,
        "outputs": {"session_candidates": str(session_path), "idphi_candidates": str(idphi_path), "lapdata_candidates": str(lap_path), "changepoint_candidates": str(cp_path), "field_alignment_summary": str(summary_path), "endpoint_draft": str(endpoint_path), "report": str(report_path)},
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text("# Redish LRA 2024 candidate extraction report\n\n" + json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Session metadata candidates saved: {session_path}")
    print(f"IdPhi candidates saved: {idphi_path}")
    print(f"LapData candidates saved: {lap_path}")
    print(f"ChangePoint candidates saved: {cp_path}")
    print(f"Field alignment summary saved: {summary_path}")
    print(f"Endpoint draft saved: {endpoint_path}")
    print(f"Metadata saved: {meta_path}")
    print(f"Report saved: {report_path}")
    print(f"Endpoint draft rows: {n_endpoint}")
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--max-depth", type=int, default=6)
    args = p.parse_args()
    extract_redish_lra_candidates(args.root, args.output_dir, args.max_depth)


if __name__ == "__main__":
    main()
