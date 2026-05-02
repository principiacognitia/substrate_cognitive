"""CLI adapter from Stage 3 all_steps CSV to VTE raw trace CSV."""

from __future__ import annotations

import argparse

from vte.adapters.stage3_steps import convert_stage3_steps_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate Stage 3 all_steps.csv into VTE raw trace schema."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--pose-source",
        default="synthetic_from_stage3_steps",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    output_path = convert_stage3_steps_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        run_id=args.run_id,
        pose_source=args.pose_source,
    )

    print(f"VTE trace saved: {output_path}")


if __name__ == "__main__":
    main()