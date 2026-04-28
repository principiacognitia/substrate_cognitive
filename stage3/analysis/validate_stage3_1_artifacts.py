#!/usr/bin/env python3
"""
Validate Stage 3.1 curated artifact packages.

This is a packaging guard, not a model test. It checks that:
- each curated package has a registry;
- every curated artifact is registered;
- registered files exist;
- Stage 3.1A figures/tables use 3_1A names;
- Stage 3.1B figures/tables use 3_1B names;
- legacy ambiguous Figure/Table_3_1C/D/E/... names do not leak back in.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


REGISTRY_FILENAMES = {
    "artifact_registry.json",
    "ARTIFACT_REGISTRY.md",
}


def load_registry(stage_dir: Path) -> Dict[str, Any]:
    path = stage_dir / "artifact_registry.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing registry: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def iter_package_files(stage_dir: Path) -> List[Path]:
    files: List[Path] = []
    for path in sorted(stage_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name in REGISTRY_FILENAMES:
            continue
        if path.name.startswith("."):
            continue
        files.append(path)
    return files


def validate_one_package(stage_dir: Path, expected_stage: str) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    try:
        registry = load_registry(stage_dir)
    except FileNotFoundError as exc:
        return [str(exc)], warnings

    if registry.get("stage") != expected_stage:
        errors.append(
            f"{stage_dir}: registry stage is {registry.get('stage')!r}, expected {expected_stage!r}"
        )

    artifacts = registry.get("artifacts", [])
    registered_files = {str(a.get("file", "")) for a in artifacts}

    actual_files = {
        p.relative_to(stage_dir).as_posix()
        for p in iter_package_files(stage_dir)
    }

    missing_from_registry = sorted(actual_files - registered_files)
    missing_on_disk = sorted(registered_files - actual_files)

    for rel in missing_from_registry:
        errors.append(f"{stage_dir}: file exists but is not registered: {rel}")

    for rel in missing_on_disk:
        errors.append(f"{stage_dir}: registry lists missing file: {rel}")

    expected_token = "3_1A" if expected_stage == "3.1A" else "3_1B"

    for rel in sorted(actual_files):
        name = Path(rel).name

        if name.startswith("Figure_") and not name.startswith(f"Figure_{expected_token}_"):
            errors.append(
                f"{stage_dir}: figure has wrong stage token for {expected_stage}: {rel}"
            )

        if name.startswith("Table_") and not name.startswith(f"Table_{expected_token}_"):
            errors.append(
                f"{stage_dir}: table has wrong stage token for {expected_stage}: {rel}"
            )

        if re.match(r"^(Figure|Table)_3_1[C-Z]_", name):
            errors.append(
                f"{stage_dir}: legacy ambiguous Stage 3.1 suffix leaked into curated package: {rel}"
            )

    # Soft semantic warning: treat should not be explained as a negative-only branch.
    for artifact in artifacts:
        condition = artifact.get("condition")
        primary = set(artifact.get("primary_variables", []))
        rel = artifact.get("file", "")

        if expected_stage == "3.1B" and condition == "treat":
            has_positive_marker = bool(
                primary
                & {
                    "h_opp",
                    "q_pos",
                    "target_prob",
                    "target_choice",
                    "local_bonus",
                    "delta_post_minus_pre",
                    "timeout",
                    "commit_latency",
                }
            )
            negative_only = bool(primary & {"h_risk", "q_neg"}) and not has_positive_marker
            if negative_only:
                warnings.append(
                    f"{stage_dir}: treat artifact appears negative-branch-only; check semantics: {rel}"
                )

    return errors, warnings


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate Stage 3.1 curated artifacts")
    ap.add_argument(
        "--root",
        default="docs/results",
        help="Curated results root containing stage3_1a and stage3_1b_closure",
    )
    ap.add_argument(
        "--stage",
        choices=["all", "3.1A", "3.1B"],
        default="all",
        help="Package to validate",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    packages: List[Tuple[Path, str]] = []
    if args.stage in {"all", "3.1A"}:
        packages.append((root / "stage3_1a", "3.1A"))
    if args.stage in {"all", "3.1B"}:
        packages.append((root / "stage3_1b_closure", "3.1B"))

    all_errors: List[str] = []
    all_warnings: List[str] = []

    for stage_dir, expected_stage in packages:
        errors, warnings = validate_one_package(stage_dir, expected_stage)
        all_errors.extend(errors)
        all_warnings.extend(warnings)

    for warning in all_warnings:
        print(f"WARNING: {warning}")

    if all_errors:
        for error in all_errors:
            print(f"ERROR: {error}")
        print(f"\nArtifact validation failed: {len(all_errors)} error(s), {len(all_warnings)} warning(s)")
        sys.exit(1)

    print(f"Artifact validation passed: {len(packages)} package(s), {len(all_warnings)} warning(s)")


if __name__ == "__main__":
    main()