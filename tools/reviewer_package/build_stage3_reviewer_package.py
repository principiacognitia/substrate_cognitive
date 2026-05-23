#!/usr/bin/env python3
"""
Build compact Stage 3 reviewer packages from committed docs/results artifacts.

Patch 20F default target:
  preset: stage3_2_seed_level_stats

Default output is LLM-facing:
  - Markdown report
  - Markdown key tables
  - 1 or 2 PNG figure pages
  - optional JSON registry

The builder does not rerun experiments and does not import project modules.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


STAGE3_2_SOURCE_SUBDIR = Path("vte") / "stage3_2_seed_level_stats_analysis"

STAGE3_2_FIGURES_STATS: List[Tuple[str, str]] = [
    (
        "Figure_3_2_Model_Relevant_Seed_Level_Effects.png",
        "Model-relevant seed-level effects",
    ),
    (
        "Figure_3_2_Seed_Level_VTE_Rate_By_Ablation.png",
        "Seed-level VTE rate by ablation",
    ),
    (
        "Figure_3_2_Degenerate_Ablation_Diagnostics.png",
        "Degenerate ablation diagnostics",
    ),
]

STAGE3_2_FIGURES_DIAGNOSTICS: List[Tuple[str, str]] = [
    (
        "Figure_3_2_Seed_Level_Ablation_Effect_Sizes.png",
        "Ablation minus reference effects",
    ),
    (
        "Figure_3_2_Seed_Level_VTE_Delta_Effect_Sizes.png",
        "VTE minus non-VTE effects",
    ),
]

STAGE3_2_MARKDOWN_TABLES: List[Tuple[str, str, str]] = [
    (
        "Table_3_2_Seed_Level_Stats_By_Test_Role.csv",
        "Test-role summary",
        "Role-level counts and effect-size summaries.",
    ),
    (
        "Table_3_2_Model_Relevant_Seed_Level_Tests.md",
        "Model-relevant seed-level tests",
        "Behavioral or ablation contrasts that can support model-level interpretation.",
    ),
    (
        "Table_3_2_Degenerate_Ablation_Diagnostics.md",
        "Degenerate ablation diagnostics",
        "Rows separated from clean localized model effects because they indicate architectural collapse or extreme regime shift.",
    ),
    (
        "Table_3_2_Wrapper_Sanity_Tests.md",
        "Wrapper sanity checks",
        "Expected VTE-label separation on IdPhi-like, pause, or reorientation metrics.",
    ),
]

STAGE3_2_REQUIRED_FILES: List[str] = [
    "Stage3_2_Response_To_GLM_Stats_Critique.md",
    "stage3_2_seed_level_stats_analysis_meta.json",
    "Table_3_2_Seed_Level_Stats_By_Test_Role.csv",
    "Table_3_2_Model_Relevant_Seed_Level_Tests.csv",
    "Table_3_2_Wrapper_Sanity_Tests.csv",
    "Table_3_2_Degenerate_Ablation_Diagnostics.csv",
    *[name for name, _title in STAGE3_2_FIGURES_STATS],
    *[name for name, _title in STAGE3_2_FIGURES_DIAGNOSTICS],
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build compact Stage 3 reviewer packages from docs/results."
    )
    ap.add_argument(
        "--preset",
        default="stage3_2_seed_level_stats",
        choices=["stage3_2_seed_level_stats"],
    )
    ap.add_argument(
        "--profile",
        default="llm5",
        choices=["llm3", "llm5"],
        help="llm3 emits 3 files; llm5 emits 5 files.",
    )
    ap.add_argument(
        "--results-root",
        default="docs/results",
        help="Path to committed results root. Relative paths are allowed.",
    )
    ap.add_argument(
        "--output-dir",
        default="docs/reviewer_packages/stage3_2_seed_level_stats",
        help="Reviewer package output directory. Relative paths are allowed.",
    )
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--source-commit", default=None)
    return ap.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def reset_output_dir(output_dir: Path, clean: bool) -> None:
    if output_dir.exists() and clean:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if any(output_dir.iterdir()) and not clean:
        raise SystemExit(
            f"ERROR: output dir is not empty: {output_dir}\n"
            "Pass --clean to rebuild it."
        )


def ensure_required_files(source_dir: Path) -> None:
    missing = [name for name in STAGE3_2_REQUIRED_FILES if not (source_dir / name).is_file()]
    if missing:
        raise SystemExit(
            "ERROR: missing required Stage 3.2 reviewer-package sources:\n- "
            + "\n- ".join(missing)
        )


def truncate_cell(value: object, max_len: int = 72) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("|", "/").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def csv_to_markdown(path: Path, max_rows: int = 25) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return "No rows.\n"

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not fieldnames:
        return "No columns.\n"

    shown = rows[:max_rows]
    header = "| " + " | ".join(fieldnames) + " |"
    sep = "| " + " | ".join(["---"] * len(fieldnames)) + " |"
    body = [
        "| " + " | ".join(truncate_cell(row.get(col, "")) for col in fieldnames) + " |"
        for row in shown
    ]

    lines = [header, sep, *body, ""]
    lines.append(f"Rows shown: {len(shown)} of {len(rows)}.")
    lines.append("")
    return "\n".join(lines)


def read_markdown_or_csv(source_dir: Path, filename: str, max_rows: int = 25) -> str:
    path = source_dir / filename
    if path.suffix.lower() == ".md" and path.exists():
        return path.read_text(encoding="utf-8").strip() + "\n"

    if path.exists() and path.suffix.lower() == ".csv":
        return csv_to_markdown(path, max_rows=max_rows)

    csv_fallback = path.with_suffix(".csv")
    if csv_fallback.exists():
        return csv_to_markdown(csv_fallback, max_rows=max_rows)

    return f"Missing table source: `{filename}`.\n"


def strip_first_heading(markdown: str) -> str:
    lines = markdown.splitlines()
    if lines and lines[0].startswith("# "):
        return "\n".join(lines[1:]).strip()
    return markdown.strip()


def write_report(
    source_dir: Path,
    output_dir: Path,
    results_root: Path,
    source_commit: str | None,
) -> Path:
    meta = read_json_if_exists(source_dir / "stage3_2_seed_level_stats_analysis_meta.json")
    glm = (source_dir / "Stage3_2_Response_To_GLM_Stats_Critique.md").read_text(
        encoding="utf-8"
    )

    report_path = output_dir / "Stage3_2_Reviewer_Report.md"

    lines = [
        "# Stage 3.2 LLM Reviewer Package",
        "",
        "This package is a compact reviewer-facing export built from committed Stage 3.2 results.",
        "It does not rerun experiments and does not alter the statistical outputs.",
        "",
        "## Source",
        "",
        f"- Results root: `{results_root}`",
        f"- Source directory: `{source_dir}`",
        f"- Source commit: `{source_commit or 'not provided'}`",
        f"- Patch recorded in meta: `{meta.get('patch', 'not recorded')}`",
        f"- Tests read: `{meta.get('n_tests', 'not recorded')}`",
        f"- Model-relevant tests: `{meta.get('n_model_relevant_tests', 'not recorded')}`",
        f"- Wrapper-sanity tests: `{meta.get('n_wrapper_sanity_tests', 'not recorded')}`",
        f"- Degenerate-ablation diagnostics: `{meta.get('n_degenerate_ablation_diagnostics', 'not recorded')}`",
        "",
        "## Included files",
        "",
        "- `Stage3_2_Reviewer_Report.md`",
        "- `Stage3_2_Key_Tables.md`",
        "- `Figure_Stage3_2_Reviewer_Page_Stats.png`",
    ]

    if (output_dir / "Figure_Stage3_2_Reviewer_Page_Diagnostics.png").exists():
        lines.append("- `Figure_Stage3_2_Reviewer_Page_Diagnostics.png`")
    if (output_dir / "reviewer_package_registry.json").exists():
        lines.append("- `reviewer_package_registry.json`")

    lines += [
        "",
        "## Interpretation boundary",
        "",
        "The package supports review of Stage 3.2 seed-level VTE statistics.",
        "It separates wrapper-sanity effects from model-relevant effects and degenerate-ablation diagnostics.",
        "It does not claim rodent-level VTE equivalence, allocentric spatial cognition, absence inference, or self-model-based visibility reasoning.",
        "",
        "## Statistical critique response",
        "",
        strip_first_heading(glm),
        "",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_key_tables(source_dir: Path, output_dir: Path) -> Path:
    out = output_dir / "Stage3_2_Key_Tables.md"
    lines = [
        "# Stage 3.2 Key Tables",
        "",
        "Compact Markdown aggregation for LLM reviewers that cannot reliably read CSV/XLSX files.",
        "",
    ]

    for filename, title, description in STAGE3_2_MARKDOWN_TABLES:
        lines += [f"## {title}", "", description, ""]
        lines.append(read_markdown_or_csv(source_dir, filename).strip())
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def render_figure_page(
    source_dir: Path,
    output_path: Path,
    specs: Sequence[Tuple[str, str]],
    title: str,
) -> Path:
    if not specs:
        raise ValueError("No figure specs supplied.")

    n = len(specs)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols

    fig_width = 15 if ncols == 2 else 10
    fig_height = max(7, nrows * 5.2)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))
    if not isinstance(axes, (list, tuple)):
        try:
            axes_flat = list(axes.ravel())
        except AttributeError:
            axes_flat = [axes]
    else:
        axes_flat = list(axes)

    fig.suptitle(title, fontsize=16, y=0.995)

    for ax, (filename, panel_title) in zip(axes_flat, specs):
        img = mpimg.imread(source_dir / filename)
        ax.imshow(img)
        ax.set_title(panel_title, fontsize=11)
        ax.axis("off")

    for ax in axes_flat[len(specs):]:
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def collect_registry_records(
    output_dir: Path,
    source_dir: Path,
    results_root: Path,
    source_commit: str | None,
) -> Dict[str, Any]:
    files = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name == "reviewer_package_registry.json":
            continue
        files.append(
            {
                "name": path.name,
                "kind": path.suffix.lower().lstrip("."),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )

    source_files = []
    for name in STAGE3_2_REQUIRED_FILES:
        src = source_dir / name
        if src.exists():
            source_files.append(
                {
                    "name": name,
                    "sha256": sha256_file(src),
                    "size_bytes": src.stat().st_size,
                }
            )

    return {
        "package": "stage3_2_llm_reviewer_package",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "results_root": str(results_root),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "source_commit": source_commit,
        "files": files,
        "source_files": source_files,
    }


def write_registry(
    output_dir: Path,
    source_dir: Path,
    results_root: Path,
    source_commit: str | None,
) -> Path:
    registry = collect_registry_records(output_dir, source_dir, results_root, source_commit)
    path = output_dir / "reviewer_package_registry.json"
    path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def validate_output(output_dir: Path, profile: str) -> None:
    expected = {
        "llm3": {
            "Stage3_2_Reviewer_Report.md",
            "Stage3_2_Key_Tables.md",
            "Figure_Stage3_2_Reviewer_Page_Stats.png",
        },
        "llm5": {
            "Stage3_2_Reviewer_Report.md",
            "Stage3_2_Key_Tables.md",
            "Figure_Stage3_2_Reviewer_Page_Stats.png",
            "Figure_Stage3_2_Reviewer_Page_Diagnostics.png",
            "reviewer_package_registry.json",
        },
    }[profile]

    produced = {p.name for p in output_dir.iterdir() if p.is_file()}
    missing = expected - produced
    extra_dirs = [p.name for p in output_dir.iterdir() if p.is_dir()]

    problems = []
    if missing:
        problems.append("missing files: " + ", ".join(sorted(missing)))
    if extra_dirs:
        problems.append("unexpected subdirectories: " + ", ".join(sorted(extra_dirs)))

    if profile == "llm3" and len(produced) != 3:
        problems.append(f"profile llm3 expected exactly 3 files, found {len(produced)}")
    if profile == "llm5" and len(produced) != 5:
        problems.append(f"profile llm5 expected exactly 5 files, found {len(produced)}")

    if problems:
        raise SystemExit("ERROR: reviewer package validation failed:\n- " + "\n- ".join(problems))


def build_stage3_2_package(args: argparse.Namespace) -> None:
    results_root = Path(args.results_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    source_dir = results_root / STAGE3_2_SOURCE_SUBDIR

    if not results_root.is_dir():
        raise SystemExit(f"ERROR: results root does not exist: {results_root}")
    if not source_dir.is_dir():
        raise SystemExit(f"ERROR: Stage 3.2 source directory does not exist: {source_dir}")

    reset_output_dir(output_dir, clean=args.clean)
    ensure_required_files(source_dir)

    write_key_tables(source_dir, output_dir)

    render_figure_page(
        source_dir,
        output_dir / "Figure_Stage3_2_Reviewer_Page_Stats.png",
        STAGE3_2_FIGURES_STATS,
        "Stage 3.2 Seed-Level Statistics: Main Reviewer Page",
    )

    if args.profile == "llm5":
        render_figure_page(
            source_dir,
            output_dir / "Figure_Stage3_2_Reviewer_Page_Diagnostics.png",
            STAGE3_2_FIGURES_DIAGNOSTICS,
            "Stage 3.2 Seed-Level Statistics: Diagnostics Page",
        )

    if args.profile == "llm5":
        # Write report before registry so the registry can hash the report.
        write_report(source_dir, output_dir, results_root, args.source_commit)
        write_registry(output_dir, source_dir, results_root, args.source_commit)
    else:
        write_report(source_dir, output_dir, results_root, args.source_commit)

    validate_output(output_dir, args.profile)

    produced = sorted(p.name for p in output_dir.iterdir() if p.is_file())
    print("Stage 3.2 reviewer package built successfully")
    print(f"Profile: {args.profile}")
    print(f"Output: {output_dir}")
    print(f"Files: {len(produced)}")
    for name in produced:
        print(f"  {name}")


def main() -> None:
    args = parse_args()
    if args.preset == "stage3_2_seed_level_stats":
        build_stage3_2_package(args)
        return
    raise SystemExit(f"Unsupported preset: {args.preset}")


if __name__ == "__main__":
    main()