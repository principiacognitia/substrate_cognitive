"""
Stage 3.1B: w_qpos_input sweep runner.

Запускает:
1. balanced_conflict control
2. one-shot treat protocol

для нескольких значений w_qpos_input при фиксированном k_pos
и сохраняет manifest.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def fmt_tag(x: float) -> str:
    return f"{int(round(float(x) * 100)):03d}"


def newest_run(base_dir: Path, prefix: str) -> Path:
    matches = sorted(base_dir.glob(f"{prefix}_*"))
    if not matches:
        raise FileNotFoundError(f"No run directory found for prefix: {prefix}")
    return matches[-1]


def run_cmd(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run Stage 3.1B w_qpos_input sweep")
    parser.add_argument(
        "--values",
        nargs="+",
        type=float,
        required=True,
        help="Sweep values, e.g. 0.0 0.2 0.4 0.6 0.8 1.0"
    )
    parser.add_argument("--k-pos-fixed", type=float, default=0.7)
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--n-trials-balanced", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="logs/stage3/stage3_1b")
    parser.add_argument("--include-balanced", action="store_true")
    parser.add_argument("--include-one-shot", action="store_true")
    parser.add_argument("--one-shot-kind", type=str, default="treat", choices=["shock", "treat"])
    parser.add_argument("--force-one-shot", action="store_true")
    parser.add_argument("--one-shot-path", type=str, default="open", choices=["open", "covered"])
    parser.add_argument("--one-shot-reward", type=float, default=5.0)
    parser.add_argument("--one-shot-salience", type=float, default=0.9)
    parser.add_argument("--one-shot-stakes", type=float, default=10.0)

    args = parser.parse_args()

    if not args.include_balanced and not args.include_one_shot:
        parser.error("Set at least one of --include-balanced or --include-one-shot")

    base_dir = Path(args.output_dir)
    manifest_rows = []

    for w in args.values:
        w_tag = fmt_tag(w)
        k_tag = fmt_tag(args.k_pos_fixed)

        param_suffix = f"_wqpi_{w_tag}_kpos_{k_tag}"

        row = {
            "w_qpos_input": float(w),
            "k_pos_fixed": float(args.k_pos_fixed),
            "balanced_run_dir": "",
            "one_shot_run_dir": "",
        }

        if args.include_balanced:
            run_cmd([
                sys.executable, "-m", "stage3.analysis.run_stage3_1b",
                "--condition", "balanced_conflict",
                "--n-seeds", str(args.n_seeds),
                "--n-trials", str(args.n_trials_balanced),
                "--ablation", "full",
                "--w-qpos-input-override", str(w),
                "--k-pos-override", str(args.k_pos_fixed),
            ])
            row["balanced_run_dir"] = str(
                newest_run(base_dir, f"balanced_conflict_full{param_suffix}")
            )

        if args.include_one_shot:
            cmd = [
                sys.executable, "-m", "stage3.analysis.run_stage3_1b",
                "--one-shot",
                "--n-seeds", str(args.n_seeds),
                "--ablation", "full",
                "--w-qpos-input-override", str(w),
                "--k-pos-override", str(args.k_pos_fixed),
                "--one-shot-kind", args.one_shot_kind,
                "--one-shot-path-override", args.one_shot_path,
                "--one-shot-reward-override", str(args.one_shot_reward),
                "--one-shot-salience-override", str(args.one_shot_salience),
                "--one-shot-stakes-override", str(args.one_shot_stakes),
            ]

            if args.force_one_shot:
                if args.one_shot_kind == "treat":
                    cmd.append("--diagnostic-forced-treat")
                else:
                    cmd.append("--diagnostic-forced-shock")

            run_cmd(cmd)

            forced_suffix = "_forced" if args.force_one_shot else ""
            row["one_shot_run_dir"] = str(
                newest_run(base_dir, f"one_shot_{args.one_shot_kind}_full{forced_suffix}{param_suffix}")
            )

        manifest_rows.append(row)

    manifest_dir = base_dir / "sweeps"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_json = manifest_dir / f"w_qpos_input_sweep_{args.one_shot_kind}.json"
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest_rows, f, indent=2, ensure_ascii=False)

    print(f"\nSaved manifest: {manifest_json}")


if __name__ == "__main__":
    main()