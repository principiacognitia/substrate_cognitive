"""
Stage 3.1A: Bias Sweep Analysis

Читает несколько run_summary.json / seed_summary.csv для разных значений
exposure_q_bias и строит сводную таблицу и базовые calibration plots.

Usage example:
    python -m stage3.analysis.analyze_bias_sweep ^
        --run "0.1=logs/stage3/stage3_1a/bias_01" ^
        --run "0.3=logs/stage3/stage3_1a/bias_03" ^
        --run "0.5=logs/stage3/stage3_1a/bias_05" ^
        --output-dir logs/figures/stage3/bias_sweep
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def load_run_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run_summary.json: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def load_seed_summary(run_dir: Path) -> pd.DataFrame:
    seed_path = run_dir / "seed_summary.csv"
    if not seed_path.exists():
        raise FileNotFoundError(f"Missing seed_summary.csv: {seed_path}")

    return pd.read_csv(seed_path)


def build_bias_sweep_table(run_specs):
    rows = []

    for bias_value, run_dir in run_specs:
        run_dir = Path(run_dir)
        run_summary = load_run_summary(run_dir)
        seed_summary = load_seed_summary(run_dir)

        row = {
            "exposure_q_bias": float(bias_value),
            "run_dir": str(run_dir),

            "n_seeds": run_summary["n_seeds"],
            "n_trials_total": run_summary["n_trials_total"],

            "p_open_total": run_summary["p_open_total"],
            "p_covered_total": run_summary["p_covered_total"],

            "mean_reward_total": run_summary["mean_reward_total"],
            "mean_commit_latency": run_summary["mean_commit_latency"],
            "mean_junction_pause_duration": run_summary["mean_junction_pause_duration"],
            "mean_reorientation_count": run_summary["mean_reorientation_count"],

            "p_commit_bound_total": run_summary["p_commit_bound_total"],
            "p_commit_timeout_total": run_summary["p_commit_timeout_total"],

            "covered_rate_mean_across_seeds": run_summary["covered_rate_mean_across_seeds"],
            "covered_rate_std_across_seeds": run_summary["covered_rate_std_across_seeds"],

            # дополнительные seed-level агрегаты
            "seed_mean_commit_latency_mean": float(seed_summary["mean_commit_latency"].mean()),
            "seed_mean_commit_latency_std": float(seed_summary["mean_commit_latency"].std(ddof=0)),
            "seed_mean_pause_mean": float(seed_summary["mean_junction_pause_duration"].mean()),
            "seed_mean_pause_std": float(seed_summary["mean_junction_pause_duration"].std(ddof=0)),
            "seed_mean_reorientation_mean": float(seed_summary["mean_reorientation_count"].mean()),
            "seed_mean_reorientation_std": float(seed_summary["mean_reorientation_count"].std(ddof=0)),
        }

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("exposure_q_bias").reset_index(drop=True)
    return df


def save_bias_sweep_table(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "Table_3_1C_BiasSweep.csv"
    df.to_csv(out_csv, index=False)
    print(f"✓ Bias sweep table saved: {out_csv}")


def plot_metric(df: pd.DataFrame, x_col: str, y_col: str, output_path: Path, title: str, ylabel: str):
    plt.figure(figsize=(7, 5))
    plt.plot(df[x_col], df[y_col], marker="o")
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ Figure saved: {output_path}")


def plot_bias_sweep(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_metric(
        df=df,
        x_col="exposure_q_bias",
        y_col="p_covered_total",
        output_path=output_dir / "Figure_3_1F_BiasSweep_CoveredRate.png",
        title="Stage 3.1A: Covered Preference vs exposure_q_bias",
        ylabel="P(covered)"
    )

    plot_metric(
        df=df,
        x_col="exposure_q_bias",
        y_col="mean_commit_latency",
        output_path=output_dir / "Figure_3_1G_BiasSweep_CommitLatency.png",
        title="Stage 3.1A: Commit Latency vs exposure_q_bias",
        ylabel="Mean commit latency"
    )

    plot_metric(
        df=df,
        x_col="exposure_q_bias",
        y_col="p_commit_timeout_total",
        output_path=output_dir / "Figure_3_1H_BiasSweep_TimeoutRate.png",
        title="Stage 3.1A: Timeout Rate vs exposure_q_bias",
        ylabel="P(timeout)"
    )


def print_bias_sweep_summary(df: pd.DataFrame):
    print("\n=== Bias Sweep Summary ===")
    print(df[
        [
            "exposure_q_bias",
            "p_covered_total",
            "mean_commit_latency",
            "mean_junction_pause_duration",
            "mean_reorientation_count",
            "p_commit_timeout_total",
        ]
    ].to_string(index=False))

    # Простая эвристика "лучшего компромисса":
    # bias должен давать P(covered) > 0.5, но не схлопывать deliberation слишком сильно.
    candidates = df[df["p_covered_total"] > 0.5].copy()
    if len(candidates) > 0:
        candidates["score"] = (
            candidates["p_covered_total"]
            - 0.25 * candidates["p_commit_timeout_total"]
            - 0.10 * (1.0 / (candidates["mean_reorientation_count"] + 1e-9))
        )
        best_row = candidates.sort_values("score", ascending=False).iloc[0]

        print("\n=== Heuristic Best Bias ===")
        print(
            f"exposure_q_bias={best_row['exposure_q_bias']:.3f}, "
            f"P(covered)={best_row['p_covered_total']:.3f}, "
            f"mean_commit_latency={best_row['mean_commit_latency']:.3f}, "
            f"timeout_rate={best_row['p_commit_timeout_total']:.3f}, "
            f"mean_reorientation={best_row['mean_reorientation_count']:.3f}"
        )


def parse_run_arg(run_arg: str):
    if "=" not in run_arg:
        raise ValueError(f"Invalid --run format: {run_arg}. Expected '<bias>=<run_dir>'")
    bias_str, run_dir = run_arg.split("=", 1)
    return float(bias_str), run_dir


def main():
    parser = argparse.ArgumentParser(description="Analyze Stage 3.1A bias sweep")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Format: <bias>=<run_dir> ; can be passed multiple times"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for figures and summary table"
    )

    args = parser.parse_args()

    run_specs = [parse_run_arg(x) for x in args.run]
    output_dir = Path(args.output_dir)

    df = build_bias_sweep_table(run_specs)
    save_bias_sweep_table(df, output_dir)
    plot_bias_sweep(df, output_dir)
    print_bias_sweep_summary(df)


if __name__ == "__main__":
    main()