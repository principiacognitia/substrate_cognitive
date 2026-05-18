#!/usr/bin/env python
"""Patch 21F: CLI for single-trial VTE visualization."""
from __future__ import annotations
import argparse
import pandas as pd
from pathlib import Path
import sys

from vte.visualization.config_regime_selection import load_regime_config
from vte.visualization.regime_selector import select_exploit_traces, select_explore_traces
from vte.visualization.render_trial import render_static_trial
from vte.visualization.animate_trial import render_animation


def parse_args():
    parser = argparse.ArgumentParser(description="Render VTE trial visualization")
    parser.add_argument("--trace-csv", required=True, help="Path to VTE trace CSV")
    parser.add_argument("--metrics-csv", required=True, help="Path to trial metrics CSV")
    parser.add_argument("--seed", type=int, required=True, help="Seed value")
    parser.add_argument("--trial", type=int, required=True, help="Trial index")
    parser.add_argument("--regime", choices=["exploit", "explore"], default="exploit")
    parser.add_argument("--output", required=True, help="Output path (PNG or GIF)")
    parser.add_argument("--animate", action="store_true", help="Render as animation")
    parser.add_argument("--config", default="vte/visualization/config_regime_selection.yaml")
    parser.add_argument("--pose-source", default="synthetic_from_stage3_steps",
                       choices=["synthetic_from_stage3_steps", "lab_tracking"])
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_regime_config(args.config)
    
    trace_df = pd.read_csv(args.trace_csv)
    metrics_df = pd.read_csv(args.metrics_csv)
    
    # Filter to requested seed/trial
    mask = (metrics_df["seed"] == args.seed) & (metrics_df["trial"] == args.trial)
    if not mask.any():
        print(f"Error: seed={args.seed}, trial={args.trial} not found in metrics", file=sys.stderr)
        sys.exit(1)
    
    metrics_row = metrics_df[mask].iloc[0]
    trial_trace = trace_df[(trace_df["seed"] == args.seed) & 
                          (trace_df["trial"] == args.trial)].sort_values("tick")
    
    output_path = Path(args.output)
    
    if args.animate:
        render_animation(trial_trace, metrics_row, output_path, config, 
                        pose_source=args.pose_source)
    else:
        regime_label = "EXPLOIT" if args.regime == "exploit" else "EXPLORE"
        title = f"{regime_label} | seed={args.seed} trial={args.trial}"
        render_static_trial(trial_trace, metrics_row, output_path, config,
                           pose_source=args.pose_source, title=title)
    
    print(f"Output written to {output_path}")


if __name__ == "__main__":
    main()