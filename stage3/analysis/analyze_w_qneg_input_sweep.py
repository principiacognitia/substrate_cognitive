"""
Stage 3.1B: w_qneg_input sweep analysis.

Читает manifest из run_w_qneg_input_sweep.py и строит:
- summary table
- control plots for balanced_conflict
- carryover plots for one-shot
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_condition_summary(run_dir: Path) -> dict:
    matches = list(run_dir.glob("*_condition_summary.csv"))
    if not matches:
        raise FileNotFoundError(f"Missing condition_summary.csv in {run_dir}")
    return pd.read_csv(matches[0]).iloc[0].to_dict()


def load_trials(run_dir: Path) -> pd.DataFrame:
    matches = list(run_dir.glob("*_all_trials.csv"))
    if not matches:
        raise FileNotFoundError(f"Missing all_trials.csv in {run_dir}")
    return pd.read_csv(matches[0])


def summarize_one_shot(run_dir: Path) -> dict:
    df = load_trials(run_dir)

    shock_trial = 30
    pre = df[df["trial"] < shock_trial].copy()
    post = df[df["trial"] > shock_trial].copy()

    pre_p_open = float((pre["path_choice"] == "open").mean()) if len(pre) else float("nan")
    post_p_open = float((post["path_choice"] == "open").mean()) if len(post) else float("nan")

    return {
        "pre_p_open": pre_p_open,
        "post_p_open": post_p_open,
        "delta_post_minus_pre": post_p_open - pre_p_open,
        "post_timeout": float((post["commit_reason"] == "timeout").mean()) if len(post) else float("nan"),
        "post_latency": float(post["commit_latency"].mean()) if len(post) else float("nan"),
        "post_pause": float(post["junction_pause_duration"].mean()) if len(post) else float("nan"),
    }


def build_table(manifest_rows):
    rows = []

    for spec in manifest_rows:
        row = {
            "w_qneg_input": float(spec["w_qneg_input"]),
        }

        balanced_dir = spec.get("balanced_run_dir", "")
        if balanced_dir:
            balanced = load_condition_summary(Path(balanced_dir))
            row.update({
                "balanced_p_open": float(balanced["p_open"]),
                "balanced_p_timeout": float(balanced["p_commit_timeout"]),
                "balanced_latency": float(balanced["mean_commit_latency"]),
                "balanced_pause": float(balanced["mean_junction_pause_duration"]),
            })

        one_shot_dir = spec.get("one_shot_run_dir", "")
        if one_shot_dir:
            one_shot = summarize_one_shot(Path(one_shot_dir))
            row.update(one_shot)

        rows.append(row)

    return pd.DataFrame(rows).sort_values("w_qneg_input").reset_index(drop=True)


def plot_series(df, x, y, out, title, ylabel):
    plt.figure(figsize=(7, 5))
    plt.plot(df[x], df[y], marker="o")
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Stage 3.1B w_qneg_input sweep")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest_rows = json.load(f)

    df = build_table(manifest_rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "Table_3_1B_w_qneg_input_Sweep.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    if "balanced_p_open" in df.columns:
        plot_series(
            df, "w_qneg_input", "balanced_p_open",
            out_dir / "Figure_3_1B_wqni_balanced_p_open.png",
            "Stage 3.1B: balanced_conflict P(open) vs w_qneg_input",
            "P(open)"
        )
        plot_series(
            df, "w_qneg_input", "balanced_p_timeout",
            out_dir / "Figure_3_1B_wqni_balanced_timeout.png",
            "Stage 3.1B: balanced_conflict timeout vs w_qneg_input",
            "P(timeout)"
        )

    if "delta_post_minus_pre" in df.columns:
        plot_series(
            df, "w_qneg_input", "delta_post_minus_pre",
            out_dir / "Figure_3_1B_wqni_one_shot_delta_post_minus_pre.png",
            "Stage 3.1B: one-shot carryover vs w_qneg_input",
            "Post minus pre P(open)"
        )
        plot_series(
            df, "w_qneg_input", "post_timeout",
            out_dir / "Figure_3_1B_wqni_one_shot_post_timeout.png",
            "Stage 3.1B: one-shot post timeout vs w_qneg_input",
            "Post P(timeout)"
        )

    print("\n=== Sweep summary ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()