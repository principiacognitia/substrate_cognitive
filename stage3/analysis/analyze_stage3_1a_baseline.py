"""
Stage 3.1A: Full Baseline Analysis

Анализирует один выбранный run Stage 3.1A и строит:
- seed-level distribution plots
- bound vs timeout analysis
- path-choice vs deliberation metrics
- block dynamics across trials
- compact markdown report

Usage:
    python -m stage3.analysis.analyze_stage3_1a_baseline ^
        --run-dir logs/stage3/stage3_1a/run_YYYYMMDD_HHMMSS ^
        --output-dir logs/figures/stage3/stage3_1a_baseline
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# LOADERS
# =============================================================================

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_run_data(run_dir: Path):
    metadata_path = run_dir / "metadata.json"
    run_summary_path = run_dir / "run_summary.json"
    seed_summary_path = run_dir / "seed_summary.csv"
    combined_path = run_dir / "all_trials_combined.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json: {metadata_path}")
    if not run_summary_path.exists():
        raise FileNotFoundError(f"Missing run_summary.json: {run_summary_path}")
    if not seed_summary_path.exists():
        raise FileNotFoundError(f"Missing seed_summary.csv: {seed_summary_path}")

    metadata = load_json(metadata_path)
    run_summary = load_json(run_summary_path)
    seed_summary = pd.read_csv(seed_summary_path)

    if combined_path.exists():
        all_trials = pd.read_csv(combined_path)
    else:
        csv_files = sorted(run_dir.glob("stage3_1a_seed*_trials.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"Missing all_trials_combined.csv and no seed trial CSV files found in {run_dir}"
            )
        dfs = []
        for f in csv_files:
            df = pd.read_csv(f)
            df["source_file"] = f.name
            dfs.append(df)
        all_trials = pd.concat(dfs, ignore_index=True)

    if "commit_reason" not in all_trials.columns:
        all_trials["commit_reason"] = "unknown"

    return metadata, run_summary, seed_summary, all_trials


# =============================================================================
# PREP
# =============================================================================

def add_block_labels(all_trials: pd.DataFrame, block_size: int = 25) -> pd.DataFrame:
    df = all_trials.copy()
    df["trial_block_idx"] = (df["trial"] // block_size) + 1

    def make_label(idx: int) -> str:
        start = (idx - 1) * block_size + 1
        end = idx * block_size
        return f"{start}-{end}"

    df["trial_block_label"] = df["trial_block_idx"].map(make_label)
    return df


# =============================================================================
# TABLES
# =============================================================================

def build_commit_reason_path_table(all_trials: pd.DataFrame) -> pd.DataFrame:
    table = (
        all_trials
        .groupby(["commit_reason", "path_choice"], dropna=False)
        .size()
        .reset_index(name="n_trials")
        .sort_values(["commit_reason", "path_choice"])
    )
    return table


def build_block_dynamics_table(all_trials: pd.DataFrame) -> pd.DataFrame:
    df = add_block_labels(all_trials, block_size=25)

    table = (
        df
        .groupby("trial_block_label", dropna=False)
        .agg(
            n_trials=("trial", "size"),
            p_covered=("path_choice", lambda s: (s == "covered").mean()),
            mean_commit_latency=("commit_latency", "mean"),
            mean_junction_pause_duration=("junction_pause_duration", "mean"),
            mean_reorientation_count=("reorientation_count", "mean"),
            p_commit_bound=("commit_reason", lambda s: (s == "bound").mean()),
            p_commit_timeout=("commit_reason", lambda s: (s == "timeout").mean()),
        )
        .reset_index()
    )
    return table


def build_path_vs_metrics_table(all_trials: pd.DataFrame) -> pd.DataFrame:
    table = (
        all_trials
        .groupby("path_choice", dropna=False)
        .agg(
            n_trials=("trial", "size"),
            mean_commit_latency=("commit_latency", "mean"),
            median_commit_latency=("commit_latency", "median"),
            mean_junction_pause_duration=("junction_pause_duration", "mean"),
            median_junction_pause_duration=("junction_pause_duration", "median"),
            mean_reorientation_count=("reorientation_count", "mean"),
            p_commit_bound=("commit_reason", lambda s: (s == "bound").mean()),
            p_commit_timeout=("commit_reason", lambda s: (s == "timeout").mean()),
        )
        .reset_index()
    )
    return table


# =============================================================================
# PLOTS
# =============================================================================

def save_plot(fig, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"✓ Figure saved: {output_path}")


def plot_seed_covered_rate(seed_summary: pd.DataFrame, output_dir: Path):
    df = seed_summary.sort_values("p_covered").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(df)), df["p_covered"])
    ax.set_title("Stage 3.1A: Covered Rate by Seed")
    ax.set_xlabel("Seeds (sorted)")
    ax.set_ylabel("P(covered)")
    ax.set_ylim(0, 1)
    save_plot(fig, output_dir / "Figure_3_1B_Seed_CoveredRate.png")


def plot_commit_reason_overall(all_trials: pd.DataFrame, output_dir: Path):
    counts = all_trials["commit_reason"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title("Stage 3.1A: Commit Reason Overall")
    ax.set_xlabel("commit_reason")
    ax.set_ylabel("count")
    save_plot(fig, output_dir / "Figure_3_1C_CommitReason_Overall.png")


def plot_path_by_commit_reason(all_trials: pd.DataFrame, output_dir: Path):
    ctab = pd.crosstab(all_trials["commit_reason"], all_trials["path_choice"], normalize="index")
    ctab = ctab.reindex(index=sorted(ctab.index))

    fig, ax = plt.subplots(figsize=(7, 5))
    bottom = None

    for col in ctab.columns:
        vals = ctab[col].values
        ax.bar(ctab.index.astype(str), vals, bottom=bottom, label=col)
        bottom = vals if bottom is None else bottom + vals

    ax.set_title("Stage 3.1A: Path Choice by Commit Reason")
    ax.set_xlabel("commit_reason")
    ax.set_ylabel("proportion")
    ax.legend()
    save_plot(fig, output_dir / "Figure_3_1D_PathChoice_by_CommitReason.png")


def plot_latency_by_commit_reason(all_trials: pd.DataFrame, output_dir: Path):
    df = (
        all_trials
        .groupby("commit_reason", dropna=False)["commit_latency"]
        .mean()
        .reset_index()
        .sort_values("commit_reason")
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(df["commit_reason"].astype(str), df["commit_latency"])
    ax.set_title("Stage 3.1A: Mean Commit Latency by Commit Reason")
    ax.set_xlabel("commit_reason")
    ax.set_ylabel("mean commit latency")
    save_plot(fig, output_dir / "Figure_3_1E_Latency_by_CommitReason.png")


def plot_reorientation_by_path(all_trials: pd.DataFrame, output_dir: Path):
    df = (
        all_trials
        .groupby("path_choice", dropna=False)["reorientation_count"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(df["path_choice"].astype(str), df["reorientation_count"])
    ax.set_title("Stage 3.1A: Mean Reorientation by Path Choice")
    ax.set_xlabel("path_choice")
    ax.set_ylabel("mean reorientation_count")
    save_plot(fig, output_dir / "Figure_3_1F_Reorientation_by_PathChoice.png")


def plot_block_dynamics(block_table: pd.DataFrame, output_dir: Path):
    x = block_table["trial_block_label"].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, block_table["p_covered"], marker="o")
    ax.set_title("Stage 3.1A: Covered Preference by Trial Block")
    ax.set_xlabel("trial block")
    ax.set_ylabel("P(covered)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    save_plot(fig, output_dir / "Figure_3_1G_Block_CoveredRate.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, block_table["mean_commit_latency"], marker="o")
    ax.set_title("Stage 3.1A: Commit Latency by Trial Block")
    ax.set_xlabel("trial block")
    ax.set_ylabel("mean commit latency")
    ax.grid(True, alpha=0.3)
    save_plot(fig, output_dir / "Figure_3_1H_Block_CommitLatency.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, block_table["mean_reorientation_count"], marker="o")
    ax.set_title("Stage 3.1A: Reorientation Count by Trial Block")
    ax.set_xlabel("trial block")
    ax.set_ylabel("mean reorientation_count")
    ax.grid(True, alpha=0.3)
    save_plot(fig, output_dir / "Figure_3_1I_Block_Reorientation.png")


# =============================================================================
# REPORT
# =============================================================================

import importlib.util

def df_to_report_table(df: pd.DataFrame) -> str:
    if importlib.util.find_spec("tabulate") is not None:
        return df.to_markdown(index=False)
    return "```\\n" + df.to_string(index=False) + "\\n```"

def write_markdown_report(
    metadata: dict,
    run_summary: dict,
    seed_summary: pd.DataFrame,
    commit_reason_table: pd.DataFrame,
    block_table: pd.DataFrame,
    path_metrics_table: pd.DataFrame,
    output_dir: Path,
):
    report_path = output_dir / "Stage3_1A_Baseline_Report.md"

    best_seed = seed_summary.sort_values("p_covered", ascending=False).iloc[0]
    worst_seed = seed_summary.sort_values("p_covered", ascending=True).iloc[0]

    lines = []
    lines.append("# Stage 3.1A Baseline Report")
    lines.append("")
    lines.append("## Run metadata")
    lines.append("")
    lines.append(f"- Stage: {metadata.get('stage', 'unknown')}")
    lines.append(f"- Seeds: {metadata.get('n_seeds', 'unknown')}")
    lines.append(f"- Trials per seed: {metadata.get('n_trials', 'unknown')}")
    lines.append(f"- Total trials: {run_summary['n_trials_total']}")
    lines.append("")

    lines.append("## Core summary")
    lines.append("")
    lines.append(f"- P(covered): {run_summary['p_covered_total']:.4f}")
    lines.append(f"- Mean commit latency: {run_summary['mean_commit_latency']:.4f}")
    lines.append(f"- Mean junction pause duration: {run_summary['mean_junction_pause_duration']:.4f}")
    lines.append(f"- Mean reorientation count: {run_summary['mean_reorientation_count']:.4f}")
    lines.append(f"- P(commit by bound): {run_summary['p_commit_bound_total']:.4f}")
    lines.append(f"- P(commit by timeout): {run_summary['p_commit_timeout_total']:.4f}")
    lines.append(f"- P(explore at junction): {run_summary['p_explore_at_junction']:.4f}")
    lines.append("")

    lines.append("## Seed-level spread")
    lines.append("")
    lines.append(f"- Mean covered-rate across seeds: {run_summary['covered_rate_mean_across_seeds']:.4f}")
    lines.append(f"- Std covered-rate across seeds: {run_summary['covered_rate_std_across_seeds']:.4f}")
    lines.append(
        f"- Best seed: {int(best_seed['seed'])} with P(covered)={best_seed['p_covered']:.4f}"
    )
    lines.append(
        f"- Worst seed: {int(worst_seed['seed'])} with P(covered)={worst_seed['p_covered']:.4f}"
    )
    lines.append("")

    lines.append("## Commit reason × path choice")
    lines.append("")
    lines.append(df_to_report_table(commit_reason_table))
    lines.append("")

    lines.append("## Path choice × deliberation metrics")
    lines.append("")
    lines.append(df_to_report_table(path_metrics_table))
    lines.append("")

    lines.append("## Block dynamics")
    lines.append("")
    lines.append(df_to_report_table(block_table))
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✓ Report saved: {report_path}")


# =============================================================================
# SAVE TABLES
# =============================================================================

def save_tables(
    seed_summary: pd.DataFrame,
    commit_reason_table: pd.DataFrame,
    block_table: pd.DataFrame,
    path_metrics_table: pd.DataFrame,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_summary.to_csv(output_dir / "Table_3_1A_SeedSummary.csv", index=False)
    commit_reason_table.to_csv(output_dir / "Table_3_1B_CommitReason_x_PathChoice.csv", index=False)
    block_table.to_csv(output_dir / "Table_3_1C_BlockDynamics.csv", index=False)
    path_metrics_table.to_csv(output_dir / "Table_3_1D_PathChoice_x_Metrics.csv", index=False)

    print(f"✓ Table saved: {output_dir / 'Table_3_1A_SeedSummary.csv'}")
    print(f"✓ Table saved: {output_dir / 'Table_3_1B_CommitReason_x_PathChoice.csv'}")
    print(f"✓ Table saved: {output_dir / 'Table_3_1C_BlockDynamics.csv'}")
    print(f"✓ Table saved: {output_dir / 'Table_3_1D_PathChoice_x_Metrics.csv'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze one Stage 3.1A baseline run")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)

    metadata, run_summary, seed_summary, all_trials = load_run_data(run_dir)

    commit_reason_table = build_commit_reason_path_table(all_trials)
    block_table = build_block_dynamics_table(all_trials)
    path_metrics_table = build_path_vs_metrics_table(all_trials)

    save_tables(seed_summary, commit_reason_table, block_table, path_metrics_table, output_dir)

    plot_seed_covered_rate(seed_summary, output_dir)
    plot_commit_reason_overall(all_trials, output_dir)
    plot_path_by_commit_reason(all_trials, output_dir)
    plot_latency_by_commit_reason(all_trials, output_dir)
    plot_reorientation_by_path(all_trials, output_dir)
    plot_block_dynamics(block_table, output_dir)

    write_markdown_report(
        metadata=metadata,
        run_summary=run_summary,
        seed_summary=seed_summary,
        commit_reason_table=commit_reason_table,
        block_table=block_table,
        path_metrics_table=path_metrics_table,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()