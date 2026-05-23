#!/usr/bin/env python3
"""
Build compact consolidated Stage 3 reviewer packages from committed docs/results artifacts.

Presets:
  stage3_1_closure        Stage 3.1A/B closure package.
  stage3_1_3_2            Stage 3.1A/B + Stage 3.2 seed-level statistics.
  stage3_visualization    Stage 3 visualization package, when visualization artifacts exist.

Profiles:
  llm3  -> 3 files: report, key tables, main figure page.
  llm5  -> 5 files: report, key tables, main figure page, diagnostics figure page, registry.

The builder does not rerun experiments and does not import project modules.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class FigureSpec:
    relpath: str
    title: str
    required: bool = True


@dataclass(frozen=True)
class TableSpec:
    relpath: str
    title: str
    description: str
    required: bool = True
    max_rows: int = 25


STAGE31_MAIN_FIGURES: List[FigureSpec] = [
    FigureSpec(
        "stage3_1a/figures/Figure_3_1A_Block_Covered_Rate.png",
        "Stage 3.1A block covered-rate dynamics",
    ),
    FigureSpec(
        "stage3_1b_closure/figures/Figure_3_1B_Matrix_P_Open_Heatmap.png",
        "Stage 3.1B matrix P(open)",
    ),
    FigureSpec(
        "stage3_1b_closure/figures/Figure_3_1B_Shock_Target_Choice_SEM_Zoom.png",
        "Shock target-choice carryover",
    ),
    FigureSpec(
        "stage3_1b_closure/figures/Figure_3_1B_Treat_Target_Choice_SEM_Zoom.png",
        "Treat target-choice carryover",
    ),
]

STAGE31_DIAGNOSTIC_FIGURES: List[FigureSpec] = [
    FigureSpec(
        "stage3_1b_closure/figures/Figure_3_1B_Matrix_P_Timeout_Heatmap.png",
        "Stage 3.1B matrix P(timeout)",
    ),
    FigureSpec(
        "stage3_1b_closure/figures/Figure_3_1B_Matrix_Commit_Latency_Heatmap.png",
        "Stage 3.1B matrix commit latency",
    ),
    FigureSpec(
        "stage3_1b_closure/figures/Figure_3_1B_Shock_QNeg_SEM_Zoom.png",
        "Shock q_neg carrier",
    ),
    FigureSpec(
        "stage3_1b_closure/figures/Figure_3_1B_Shock_HRisk_SEM_Zoom.png",
        "Shock h_risk trace",
    ),
    FigureSpec(
        "stage3_1b_closure/figures/Figure_3_1B_Treat_QPos_SEM_Zoom.png",
        "Treat q_pos carrier",
    ),
    FigureSpec(
        "stage3_1b_closure/figures/Figure_3_1B_OneShot_Effect_By_Ablation.png",
        "One-shot effect by ablation",
    ),
]

STAGE32_MAIN_FIGURES: List[FigureSpec] = [
    FigureSpec(
        "vte/stage3_2_seed_level_stats_analysis/Figure_3_2_Model_Relevant_Seed_Level_Effects.png",
        "Stage 3.2 model-relevant seed-level effects",
    ),
    FigureSpec(
        "vte/stage3_2_seed_level_stats_analysis/Figure_3_2_Seed_Level_VTE_Rate_By_Ablation.png",
        "Stage 3.2 VTE rate by ablation",
    ),
]

STAGE32_DIAGNOSTIC_FIGURES: List[FigureSpec] = [
    FigureSpec(
        "vte/stage3_2_seed_level_stats_analysis/Figure_3_2_Degenerate_Ablation_Diagnostics.png",
        "Stage 3.2 degenerate ablation diagnostics",
    ),
    FigureSpec(
        "vte/stage3_2_seed_level_stats_analysis/Figure_3_2_Seed_Level_Ablation_Effect_Sizes.png",
        "Stage 3.2 ablation effect sizes",
    ),
    FigureSpec(
        "vte/stage3_2_seed_level_stats_analysis/Figure_3_2_Seed_Level_VTE_Delta_Effect_Sizes.png",
        "Stage 3.2 VTE delta effect sizes",
    ),
]

STAGE31_TABLES: List[TableSpec] = [
    TableSpec(
        "stage3_1a/tables/Table_3_1A_Seed_Summary.csv",
        "Stage 3.1A seed summary",
        "Seed-level compatibility summary for the Stage 3.1A baseline.",
    ),
    TableSpec(
        "stage3_1a/tables/Table_3_1A_Block_Dynamics.csv",
        "Stage 3.1A block dynamics",
        "Block-level baseline dynamics.",
    ),
    TableSpec(
        "stage3_1b_closure/tables/Table_3_1B_matrix_cell_summary.csv",
        "Stage 3.1B matrix cell summary",
        "Balanced-conflict matrix summary by cell.",
    ),
    TableSpec(
        "stage3_1b_closure/tables/Table_3_1B_ablation_summary_wide.csv",
        "Stage 3.1B ablation summary",
        "Balanced and one-shot ablation summary.",
    ),
    TableSpec(
        "stage3_1b_closure/tables/Table_3_1B_one_shot_effect_stats.csv",
        "Stage 3.1B one-shot effect statistics",
        "Target-choice one-shot effect statistics.",
    ),
    TableSpec(
        "stage3_1b_closure/tables/Table_3_1B_one_shot_carrier_effect_stats.csv",
        "Stage 3.1B carrier effect statistics",
        "Carrier-trace one-shot effect statistics.",
    ),
    TableSpec(
        "stage3_1b_closure/tables/Table_3_1B_one_shot_ablation_localization.csv",
        "Stage 3.1B ablation localization",
        "One-shot ablation-localization diagnostics.",
    ),
    TableSpec(
        "stage3_1b_closure/tables/Table_3_1B_one_shot_acceptance_summary.csv",
        "Stage 3.1B acceptance summary",
        "Closure acceptance summary.",
    ),
]

STAGE32_TABLES: List[TableSpec] = [
    TableSpec(
        "vte/stage3_2_seed_level_stats_analysis/Table_3_2_Seed_Level_Stats_By_Test_Role.csv",
        "Stage 3.2 test-role summary",
        "Role-level counts separating model-relevant, wrapper-sanity, and degenerate-ablation rows.",
    ),
    TableSpec(
        "vte/stage3_2_seed_level_stats_analysis/Table_3_2_Model_Relevant_Seed_Level_Tests.md",
        "Stage 3.2 model-relevant tests",
        "Seed-level behavioral or ablation contrasts relevant to model interpretation.",
    ),
    TableSpec(
        "vte/stage3_2_seed_level_stats_analysis/Table_3_2_Degenerate_Ablation_Diagnostics.md",
        "Stage 3.2 degenerate ablation diagnostics",
        "Rows separated from clean localized model effects.",
    ),
    TableSpec(
        "vte/stage3_2_seed_level_stats_analysis/Table_3_2_Wrapper_Sanity_Tests.md",
        "Stage 3.2 wrapper-sanity tests",
        "Expected VTE-label separation on IdPhi-like, pause, and reorientation metrics.",
    ),
]

STAGE32_OPTIONAL_REPORTS: List[str] = [
    "vte/stage3_2_seed_level_stats_analysis/Stage3_2_Response_To_GLM_Stats_Critique.md",
    "vte/stage3_2_seed_level_stats_analysis/stage3_2_seed_level_stats_analysis_meta.json",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build consolidated Stage 3 reviewer packages from docs/results."
    )
    ap.add_argument(
        "--preset",
        required=True,
        choices=["stage3_1_closure", "stage3_1_3_2", "stage3_visualization"],
    )
    ap.add_argument(
        "--profile",
        default="llm5",
        choices=["llm3", "llm5"],
    )
    ap.add_argument(
        "--results-root",
        default="docs/results",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Default: docs/reviewer_packages/<preset>",
    )
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--source-commit", default=None)
    ap.add_argument(
        "--visualization-dir",
        default="vte/visualization",
        help="Relative to --results-root. Used only for preset=stage3_visualization.",
    )
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
    lines = [header, sep, *body, "", f"Rows shown: {len(shown)} of {len(rows)}.", ""]
    return "\n".join(lines)


def strip_first_heading(markdown: str) -> str:
    lines = markdown.splitlines()
    if lines and lines[0].startswith("# "):
        return "\n".join(lines[1:]).strip()
    return markdown.strip()


def read_table_as_markdown(results_root: Path, spec: TableSpec) -> str:
    path = results_root / spec.relpath
    if not path.exists():
        if spec.required:
            raise SystemExit(f"ERROR: required table source is missing: {path}")
        return f"Missing optional table source: `{spec.relpath}`.\n"

    if path.suffix.lower() == ".md":
        return strip_first_heading(path.read_text(encoding="utf-8")) + "\n"

    if path.suffix.lower() == ".csv":
        return csv_to_markdown(path, max_rows=spec.max_rows)

    return f"Unsupported table format: `{spec.relpath}`.\n"


def figure_specs_for_visualization(results_root: Path, rel_dir: str) -> tuple[List[FigureSpec], List[FigureSpec], List[TableSpec]]:
    source_dir = results_root / rel_dir
    if not source_dir.is_dir():
        raise SystemExit(f"ERROR: visualization source directory does not exist: {source_dir}")

    pngs = sorted(source_dir.glob("*.png"))
    if not pngs:
        raise SystemExit(f"ERROR: no PNG visualization figures found in: {source_dir}")

    specs = [
        FigureSpec(str(path.relative_to(results_root)), path.stem.replace("_", " "))
        for path in pngs
    ]

    main = specs[:4]
    diagnostics = specs[4:10] or specs[:1]

    tables: List[TableSpec] = []
    for candidate in [
        "selected_examples.csv",
        "Selected_Examples.csv",
        "Table_3_2_Selected_Visualization_Examples.csv",
    ]:
        if (source_dir / candidate).exists():
            tables.append(
                TableSpec(
                    str((source_dir / candidate).relative_to(results_root)),
                    "Stage 3 visualization selected examples",
                    "Selected examples used by the visualization layer.",
                    required=False,
                )
            )
            break

    return main, diagnostics, tables


def get_preset_content(
    preset: str,
    results_root: Path,
    visualization_dir: str,
) -> tuple[str, List[FigureSpec], List[FigureSpec], List[TableSpec], str]:
    if preset == "stage3_1_closure":
        return (
            "Stage 3.1A/B Closure Reviewer Package",
            STAGE31_MAIN_FIGURES,
            STAGE31_DIAGNOSTIC_FIGURES,
            STAGE31_TABLES,
            (
                "This package supports review of Stage 3.1A baseline compatibility "
                "and Stage 3.1B valence/exposure kernel closure."
            ),
        )

    if preset == "stage3_1_3_2":
        return (
            "Stage 3.1/3.2 Consolidated Reviewer Package",
            STAGE31_MAIN_FIGURES + STAGE32_MAIN_FIGURES,
            STAGE31_DIAGNOSTIC_FIGURES + STAGE32_DIAGNOSTIC_FIGURES,
            STAGE31_TABLES + STAGE32_TABLES,
            (
                "This package combines Stage 3.1A/B closure artifacts with Stage 3.2 "
                "seed-level VTE statistics."
            ),
        )

    if preset == "stage3_visualization":
        main, diagnostics, tables = figure_specs_for_visualization(results_root, visualization_dir)
        return (
            "Stage 3 Visualization Reviewer Package",
            main,
            diagnostics,
            tables,
            (
                "This package supports review of the Stage 3 visualization layer. "
                "Visualization outputs are presentation artifacts and are not metric sources."
            ),
        )

    raise SystemExit(f"Unsupported preset: {preset}")


def ensure_figure_sources(results_root: Path, specs: Sequence[FigureSpec]) -> None:
    missing = [
        spec.relpath
        for spec in specs
        if spec.required and not (results_root / spec.relpath).is_file()
    ]
    if missing:
        raise SystemExit(
            "ERROR: missing required figure sources:\n- " + "\n- ".join(missing)
        )


def render_text_page(output_path: Path, title: str, message: str) -> Path:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    ax.text(0.5, 0.65, title, ha="center", va="center", fontsize=18, weight="bold")
    ax.text(0.5, 0.45, message, ha="center", va="center", fontsize=12, wrap=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_figure_page(
    results_root: Path,
    output_path: Path,
    specs: Sequence[FigureSpec],
    title: str,
) -> Path:
    if not specs:
        return render_text_page(output_path, title, "No figure sources were selected for this page.")

    n = len(specs)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols

    fig_width = 16 if ncols == 2 else 10
    fig_height = max(7, nrows * 5.0)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))
    try:
        axes_flat = list(axes.ravel())
    except AttributeError:
        axes_flat = [axes]

    fig.suptitle(title, fontsize=16, y=0.995)

    for ax, spec in zip(axes_flat, specs):
        path = results_root / spec.relpath
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(spec.title, fontsize=10)
        ax.axis("off")

    for ax in axes_flat[len(specs):]:
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_key_tables(
    results_root: Path,
    output_dir: Path,
    package_title: str,
    table_specs: Sequence[TableSpec],
) -> Path:
    out = output_dir / "Stage3_Key_Tables.md"
    lines = [
        f"# {package_title}: Key Tables",
        "",
        "Compact Markdown aggregation for reviewers that cannot reliably read CSV/XLSX files.",
        "",
    ]

    if not table_specs:
        lines += ["No table sources were selected for this preset.", ""]
    else:
        for spec in table_specs:
            lines += [f"## {spec.title}", "", spec.description, ""]
            lines.append(read_table_as_markdown(results_root, spec).strip())
            lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def read_optional_text(results_root: Path, relpath: str, max_chars: int = 8000) -> str:
    path = results_root / relpath
    if not path.exists() or path.suffix.lower() != ".md":
        return ""
    text = path.read_text(encoding="utf-8").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[truncated]\n"


def write_report(
    results_root: Path,
    output_dir: Path,
    *,
    preset: str,
    profile: str,
    package_title: str,
    package_description: str,
    source_commit: str | None,
) -> Path:
    out = output_dir / "Stage3_Reviewer_Report.md"

    manifest = read_json_if_exists(results_root / "stage3_1_closure_manifest.json")
    stage32_meta = read_json_if_exists(
        results_root
        / "vte"
        / "stage3_2_seed_level_stats_analysis"
        / "stage3_2_seed_level_stats_analysis_meta.json"
    )

    lines = [
        f"# {package_title}",
        "",
        package_description,
        "",
        "This is a compact reviewer-facing transport package.",
        "It does not rerun experiments, does not recompute statistics, and does not modify source artifacts.",
        "",
        "## Source",
        "",
        f"- Preset: `{preset}`",
        f"- Profile: `{profile}`",
        f"- Results root: `{results_root}`",
        f"- Source commit: `{source_commit or 'not provided'}`",
        f"- Stage 3.1 manifest mode: `{manifest.get('mode', 'not recorded')}`",
        f"- Stage 3.1 manifest run id: `{manifest.get('run_id', 'not recorded')}`",
        f"- Stage 3.2 patch: `{stage32_meta.get('patch', 'not recorded')}`",
        f"- Stage 3.2 tests read: `{stage32_meta.get('n_tests', 'not recorded')}`",
        "",
        "## Included files",
        "",
        "- `Stage3_Reviewer_Report.md`",
        "- `Stage3_Key_Tables.md`",
        "- `Figure_Stage3_Reviewer_Page_Main.png`",
    ]

    if profile == "llm5":
        lines += [
            "- `Figure_Stage3_Reviewer_Page_Diagnostics.png`",
            "- `reviewer_package_registry.json`",
        ]

    lines += [
        "",
        "## Interpretation boundary",
        "",
    ]

    if preset == "stage3_1_closure":
        lines += [
            "This package supports Stage 3.1A baseline compatibility and Stage 3.1B valence/exposure kernel closure.",
            "It does not claim Stage 3.2 VTE validation, absence inference, allocentric spatial cognition, or self-model-based visibility reasoning.",
            "",
        ]
    elif preset == "stage3_1_3_2":
        lines += [
            "This package combines Stage 3.1A/B closure with Stage 3.2 seed-level VTE statistics.",
            "Stage 3.2 statistics are role-aware: wrapper-sanity effects are separated from model-relevant effects and degenerate-ablation diagnostics.",
            "It does not claim rodent-level VTE equivalence, absence inference, allocentric spatial cognition, or self-model-based visibility reasoning.",
            "",
        ]
        glm = read_optional_text(
            results_root,
            "vte/stage3_2_seed_level_stats_analysis/Stage3_2_Response_To_GLM_Stats_Critique.md",
        )
        if glm:
            lines += ["## Stage 3.2 statistical critique response", "", strip_first_heading(glm), ""]
    elif preset == "stage3_visualization":
        lines += [
            "Visualization artifacts are read-only presentation artifacts.",
            "They should not be used as metric sources and should not be cited as additional inferential evidence.",
            "",
        ]

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def collect_source_files(
    results_root: Path,
    figure_specs: Sequence[FigureSpec],
    table_specs: Sequence[TableSpec],
    preset: str,
) -> List[Dict[str, Any]]:
    rels = {spec.relpath for spec in figure_specs}
    rels.update(spec.relpath for spec in table_specs)

    if preset in {"stage3_1_closure", "stage3_1_3_2"}:
        rels.add("stage3_1_closure_manifest.json")

    if preset == "stage3_1_3_2":
        rels.update(STAGE32_OPTIONAL_REPORTS)

    records: List[Dict[str, Any]] = []
    for rel in sorted(rels):
        path = results_root / rel
        if path.exists() and path.is_file():
            records.append(
                {
                    "relpath": rel,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return records


def write_registry(
    results_root: Path,
    output_dir: Path,
    *,
    preset: str,
    profile: str,
    package_title: str,
    source_commit: str | None,
    figure_specs: Sequence[FigureSpec],
    table_specs: Sequence[TableSpec],
) -> Path:
    out = output_dir / "reviewer_package_registry.json"

    output_files = []
    for path in sorted(output_dir.iterdir()):
        if path.name == out.name or not path.is_file():
            continue
        output_files.append(
            {
                "name": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )

    registry = {
        "package": "stage3_consolidated_reviewer_package",
        "preset": preset,
        "profile": profile,
        "title": package_title,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "source_commit": source_commit,
        "output_files": output_files,
        "source_files": collect_source_files(results_root, figure_specs, table_specs, preset),
    }

    out.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def validate_output(output_dir: Path, profile: str) -> None:
    expected = {
        "llm3": {
            "Stage3_Reviewer_Report.md",
            "Stage3_Key_Tables.md",
            "Figure_Stage3_Reviewer_Page_Main.png",
        },
        "llm5": {
            "Stage3_Reviewer_Report.md",
            "Stage3_Key_Tables.md",
            "Figure_Stage3_Reviewer_Page_Main.png",
            "Figure_Stage3_Reviewer_Page_Diagnostics.png",
            "reviewer_package_registry.json",
        },
    }[profile]

    produced = {p.name for p in output_dir.iterdir() if p.is_file()}
    dirs = [p.name for p in output_dir.iterdir() if p.is_dir()]

    problems = []
    missing = expected - produced
    extra = produced - expected

    if missing:
        problems.append("missing files: " + ", ".join(sorted(missing)))
    if extra:
        problems.append("unexpected files: " + ", ".join(sorted(extra)))
    if dirs:
        problems.append("unexpected subdirectories: " + ", ".join(sorted(dirs)))

    if problems:
        raise SystemExit("ERROR: reviewer package validation failed:\n- " + "\n- ".join(problems))


def default_output_dir_for_preset(preset: str) -> Path:
    return Path("docs") / "reviewer_packages" / preset


def build_package(args: argparse.Namespace) -> None:
    results_root = Path(args.results_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_output_dir_for_preset(args.preset).resolve()
    )

    if not results_root.is_dir():
        raise SystemExit(f"ERROR: results root does not exist: {results_root}")

    package_title, main_figures, diagnostic_figures, table_specs, package_description = (
        get_preset_content(args.preset, results_root, args.visualization_dir)
    )

    figure_specs = main_figures + ([] if args.profile == "llm3" else diagnostic_figures)
    ensure_figure_sources(results_root, figure_specs)

    reset_output_dir(output_dir, clean=args.clean)

    write_key_tables(results_root, output_dir, package_title, table_specs)

    render_figure_page(
        results_root,
        output_dir / "Figure_Stage3_Reviewer_Page_Main.png",
        main_figures,
        f"{package_title}: Main Page",
    )

    if args.profile == "llm5":
        render_figure_page(
            results_root,
            output_dir / "Figure_Stage3_Reviewer_Page_Diagnostics.png",
            diagnostic_figures,
            f"{package_title}: Diagnostics Page",
        )

    write_report(
        results_root,
        output_dir,
        preset=args.preset,
        profile=args.profile,
        package_title=package_title,
        package_description=package_description,
        source_commit=args.source_commit,
    )

    if args.profile == "llm5":
        write_registry(
            results_root,
            output_dir,
            preset=args.preset,
            profile=args.profile,
            package_title=package_title,
            source_commit=args.source_commit,
            figure_specs=figure_specs,
            table_specs=table_specs,
        )

    validate_output(output_dir, args.profile)

    produced = sorted(p.name for p in output_dir.iterdir() if p.is_file())
    print("Stage 3 reviewer package built successfully")
    print(f"Preset: {args.preset}")
    print(f"Profile: {args.profile}")
    print(f"Output: {output_dir}")
    print(f"Files: {len(produced)}")
    for name in produced:
        print(f"  {name}")


def main() -> None:
    args = parse_args()
    build_package(args)


if __name__ == "__main__":
    main()