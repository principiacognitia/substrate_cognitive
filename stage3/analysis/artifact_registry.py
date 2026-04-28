#!/usr/bin/env python3
"""
Artifact registry helpers for Stage 3.1 closure packages.

This module is deliberately analysis-layer only. It does not run experiments,
does not change agent/environment behavior, and does not introduce new claims.
It records what curated artifacts exist, where they came from, and what
semantic role they play in the paper package.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REGISTRY_FILENAMES = {
    "artifact_registry.json",
    "ARTIFACT_REGISTRY.md",
}


def artifact_type_for_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
        return "figure"
    if suffix in {".csv", ".tsv"}:
        return "table"
    if suffix in {".json", ".jsonl"}:
        return "stats"
    if suffix in {".md", ".txt"}:
        return "report"
    return "misc"


def stable_artifact_id(stage: str, rel_path: str) -> str:
    raw = f"{stage}_{rel_path}"
    raw = raw.replace("\\", "/")
    raw = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").lower()
    return raw


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    lower = text.lower()
    return any(n.lower() in lower for n in needles)


def infer_condition_protocol_and_semantics(stage: str, rel_path: str) -> Dict[str, Any]:
    """
    Best-effort inference from curated filenames.

    This is not meant to replace analyzer-emitted metadata. It gives the
    closure package a useful registry immediately, while future analyzers can
    emit richer records.
    """
    name = Path(rel_path).name
    lower = rel_path.lower()

    condition = "baseline"
    protocol = "stage3_1a_baseline" if stage == "3.1A" else "stage3_1b_closure"
    branch_semantics = "none"

    if stage == "3.1B":
        if "balanced" in lower:
            condition = "balanced"
            protocol = "balanced_conflict_matrix"
            branch_semantics = "mixed"
        elif "shock" in lower:
            condition = "shock"
            protocol = "one_shot_shock"
            branch_semantics = "negative"
        elif "treat" in lower:
            condition = "treat"
            protocol = "one_shot_treat"
            branch_semantics = "positive"
        elif "one_shot" in lower:
            condition = "one_shot"
            protocol = "one_shot"
            branch_semantics = "mixed"
        elif "ablation" in lower:
            condition = "ablation_suite"
            protocol = "ablation_suite"
            branch_semantics = "mixed"

    primary_variables: List[str] = []

    variable_rules = [
        ("p_open", ["p_open", "open"]),
        ("p_covered", ["p_covered", "covered"]),
        ("path_choice", ["pathchoice", "path_choice", "path-choice"]),
        ("commit_reason", ["commitreason", "commit_reason"]),
        ("commit_latency", ["commit_latency", "latency"]),
        ("timeout", ["timeout"]),
        ("junction_pause_duration", ["junction_pause", "pause"]),
        ("reorientation_count", ["reorientation"]),
        ("mode_at_junction", ["mode", "junction_mode"]),
        ("h_risk", ["h_risk", "risk"]),
        ("q_neg", ["q_neg", "negative"]),
        ("h_opp", ["h_opp", "opportunity", "opp"]),
        ("q_pos", ["q_pos", "positive"]),
        ("target_prob", ["target_prob", "first_target_prob"]),
        ("target_choice", ["target_choice", "first_target_choice"]),
        ("local_bonus", ["local_bonus", "target_lb", "first_target_lb"]),
        ("delta_post_minus_pre", ["delta", "post_minus_pre", "post-pre"]),
        ("seed", ["seed"]),
        ("bootstrap_ci", ["ci", "bootstrap"]),
    ]

    for variable, markers in variable_rules:
        if _contains_any(lower, markers):
            primary_variables.append(variable)

    primary_variables = sorted(set(primary_variables))

    paper_role = "Curated Stage 3.1 artifact"
    interpretation = "Registered curated output for the Stage 3.1 closure package."

    if stage == "3.1A":
        paper_role = "Stage 3.1A compatibility / baseline artifact"
        interpretation = (
            "Documents the calibrated open/covered baseline preserved under "
            "the Stage 3.1B closure branch."
        )
    elif stage == "3.1B" and condition == "balanced":
        paper_role = "Stage 3.1B balanced-conflict / matrix artifact"
        interpretation = (
            "Documents tradeoff-sensitive path choice and ablation effects "
            "under balanced conflict."
        )
    elif stage == "3.1B" and condition == "shock":
        paper_role = "Stage 3.1B negative one-shot shock artifact"
        interpretation = (
            "Documents the negative branch of one-shot carryover: shock, "
            "risk trace, avoidance shift, or post-shock persistence."
        )
    elif stage == "3.1B" and condition == "treat":
        paper_role = "Stage 3.1B positive one-shot treat artifact"
        interpretation = (
            "Documents the positive branch of one-shot carryover: treatment, "
            "opportunity trace, approach shift, or post-treat persistence."
        )
    elif stage == "3.1B" and condition == "ablation_suite":
        paper_role = "Stage 3.1B ablation-localization artifact"
        interpretation = (
            "Documents whether the observed effects localize to the intended "
            "carrier or ablation family rather than to generic policy noise."
        )

    return {
        "condition": condition,
        "protocol": protocol,
        "branch_semantics": branch_semantics,
        "primary_variables": primary_variables,
        "paper_role": paper_role,
        "interpretation": interpretation,
    }


def iter_curated_files(curated_stage_root: Path) -> List[Path]:
    files: List[Path] = []
    for path in sorted(curated_stage_root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in REGISTRY_FILENAMES:
            continue
        if path.name.startswith("."):
            continue
        files.append(path)
    return files


def build_artifact_registry(
    *,
    curated_stage_root: Path,
    stage: str,
    package_id: str,
    git_info: Dict[str, Any],
    profile: Dict[str, Any],
    source: Dict[str, Any],
    schema_version: str = "stage3_1_closure_artifact_registry_v1",
) -> Dict[str, Any]:
    curated_stage_root = Path(curated_stage_root)

    artifacts: List[Dict[str, Any]] = []
    for path in iter_curated_files(curated_stage_root):
        rel_path = path.relative_to(curated_stage_root).as_posix()
        inferred = infer_condition_protocol_and_semantics(stage, rel_path)
        artifacts.append(
            {
                "id": stable_artifact_id(stage, rel_path),
                "file": rel_path,
                "type": artifact_type_for_file(path),
                "stage": stage,
                "condition": inferred["condition"],
                "protocol": inferred["protocol"],
                "branch_semantics": inferred["branch_semantics"],
                "primary_variables": inferred["primary_variables"],
                "source": source,
                "paper_role": inferred["paper_role"],
                "interpretation": inferred["interpretation"],
                "status": "publication_candidate",
            }
        )

    return {
        "package_id": package_id,
        "schema_version": schema_version,
        "stage": stage,
        "created_at": datetime.now().isoformat(),
        "curated_stage_root": str(curated_stage_root),
        "git": git_info,
        "profile": profile,
        "source": source,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def render_registry_markdown(registry: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# Artifact Registry: {registry.get('stage', 'unknown stage')}")
    lines.append("")
    lines.append(f"- Package ID: `{registry.get('package_id', '')}`")
    lines.append(f"- Schema: `{registry.get('schema_version', '')}`")
    lines.append(f"- Created: `{registry.get('created_at', '')}`")
    lines.append(f"- Artifact count: `{registry.get('artifact_count', 0)}`")
    lines.append("")

    git = registry.get("git", {}) or {}
    if git:
        lines.append("## Git")
        lines.append("")
        lines.append(f"- Branch: `{git.get('branch', '')}`")
        lines.append(f"- Commit: `{git.get('commit', '')}`")
        lines.append(f"- Working tree clean: `{git.get('working_tree_clean', '')}`")
        lines.append("")

    source = registry.get("source", {}) or {}
    if source:
        lines.append("## Source")
        lines.append("")
        for key, value in source.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append(
        "| File | Type | Condition | Protocol | Branch semantics | Primary variables | Paper role |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---|---|"
    )

    for artifact in registry.get("artifacts", []):
        primary = ", ".join(artifact.get("primary_variables", []))
        lines.append(
            "| "
            f"`{artifact.get('file', '')}` | "
            f"{artifact.get('type', '')} | "
            f"{artifact.get('condition', '')} | "
            f"{artifact.get('protocol', '')} | "
            f"{artifact.get('branch_semantics', '')} | "
            f"{primary} | "
            f"{artifact.get('paper_role', '')} |"
        )

    lines.append("")
    return "\n".join(lines)


def write_registry_files(registry: Dict[str, Any], curated_stage_root: Path) -> Dict[str, str]:
    curated_stage_root = Path(curated_stage_root)
    json_path = curated_stage_root / "artifact_registry.json"
    md_path = curated_stage_root / "ARTIFACT_REGISTRY.md"

    json_path.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_path.write_text(render_registry_markdown(registry), encoding="utf-8")

    return {
        "json": str(json_path),
        "markdown": str(md_path),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a Stage 3.1 artifact registry")
    ap.add_argument("--curated-stage-root", required=True)
    ap.add_argument("--stage", required=True, choices=["3.1A", "3.1B"])
    ap.add_argument("--package-id", default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.curated_stage_root)
    package_id = args.package_id or f"{args.stage.replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    registry = build_artifact_registry(
        curated_stage_root=root,
        stage=args.stage,
        package_id=package_id,
        git_info={},
        profile={},
        source={"manual_registry_build": True},
    )
    paths = write_registry_files(registry, root)
    print(f"Registry JSON: {paths['json']}")
    print(f"Registry MD: {paths['markdown']}")


if __name__ == "__main__":
    main()