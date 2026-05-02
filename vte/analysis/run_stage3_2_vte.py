"""CLI entry point for Stage 3.2 VTE wrapper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vte.core.metrics import VTEThresholdConfig
from vte.core.wrapper import run_vte_wrapper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stage 3.2 VTE wrapper on externalized trace CSV."
    )

    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to trace CSV produced by an external model run.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where VTE wrapper outputs should be written.",
    )
    parser.add_argument(
        "--output-name",
        default="vte_trial_metrics.csv",
        help="Output CSV filename.",
    )
    parser.add_argument(
        "--z-idphi-threshold",
        type=float,
        default=1.0,
        help="Threshold for z-scored log IdPhi.",
    )
    parser.add_argument(
        "--min-pause-ticks",
        type=int,
        default=2,
        help="Minimum choice-point pause duration for binary VTE label.",
    )
    parser.add_argument(
        "--min-reorientation-count",
        type=int,
        default=1,
        help="Minimum reorientation count for binary VTE label.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    cfg = VTEThresholdConfig(
        z_idphi_threshold=args.z_idphi_threshold,
        min_pause_ticks=args.min_pause_ticks,
        min_reorientation_count=args.min_reorientation_count,
    )

    result = run_vte_wrapper(
        input_path=args.input_csv,
        output_dir=args.output_dir,
        output_name=args.output_name,
        threshold_config=cfg,
    )

    meta = {
        "input_path": str(result.input_path),
        "output_path": str(result.output_path),
        "n_trace_rows": result.n_trace_rows,
        "n_trial_rows": result.n_trial_rows,
        "threshold_config": {
            "z_idphi_threshold": cfg.z_idphi_threshold,
            "min_pause_ticks": cfg.min_pause_ticks,
            "min_reorientation_count": cfg.min_reorientation_count,
        },
    }

    meta_path = Path(args.output_dir) / "vte_wrapper_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"VTE metrics saved: {result.output_path}")
    print(f"Metadata saved: {meta_path}")
    print(f"Trace rows: {result.n_trace_rows}")
    print(f"Trial rows: {result.n_trial_rows}")


if __name__ == "__main__":
    main()