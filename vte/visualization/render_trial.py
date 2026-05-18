"""Patch 21C: Static single-trial VTE trace renderer."""
from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def plot_maze_schematic(ax, geometry_id: str = "fork_open_covered_v1"):
    """Render canonical fork maze skeleton."""
    # Schematic coordinates (arbitrary units)
    start = np.array([0.0, -1.0])
    junction = np.array([0.0, 0.0])
    open_end = np.array([-1.0, 1.0])
    covered_end = np.array([1.0, 1.0])
    
    # Draw paths
    ax.plot([start[0], junction[0]], [start[1], junction[1]], 'k-', lw=2, label='approach')
    ax.plot([junction[0], open_end[0]], [junction[1], open_end[1]], 
            'b--', lw=1.5, alpha=0.7, label='open path')
    ax.plot([junction[0], covered_end[0]], [junction[1], covered_end[1]], 
            'g--', lw=1.5, alpha=0.7, label='covered path')
    
    # Choice point marker
    circle = plt.Circle(junction, 0.12, color='red', fill=False, lw=2, label='choice point')
    ax.add_patch(circle)
    
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')


def plot_trajectory(ax, trace_df: pd.DataFrame, trail_length: int = 8):
    """Plot trajectory with heading arrows and fading tail."""
    if trace_df.empty:
        return
    
    x, y = trace_df["x"].values, trace_df["y"].values
    heading = trace_df["heading"].values
    
    # Fading trail
    for i in range(max(0, len(trace_df) - trail_length), len(trace_df)):
        alpha = (i - (len(trace_df) - trail_length)) / trail_length
        if i > 0:
            ax.plot([x[i-1], x[i]], [y[i-1], y[i]], 
                   color='navy', alpha=alpha*0.8, lw=2)
    
    # Heading arrows at choice point
    cp_mask = trace_df["at_choice_point"].astype(bool)
    if cp_mask.any():
        cp_idx = trace_df[cp_mask].index
        for idx in cp_idx:
            i = trace_df.index.get_loc(idx)
            dx, dy = 0.15 * np.cos(heading[i]), 0.15 * np.sin(heading[i])
            ax.arrow(x[i], y[i], dx, dy, 
                    head_width=0.08, head_length=0.1, 
                    fc='orange', ec='orange', alpha=0.9)


def add_metrics_panel(ax, metrics_row: pd.Series, pose_source: str):
    """Add inset panel with trial-level VTE metrics."""
    ax_metrics = ax.inset_axes([0.65, 0.05, 0.33, 0.25])
    ax_metrics.axis('off')
    
    lines = [
        f"run_id: {metrics_row.get('run_id', 'N/A')[:12]}...",
        f"seed: {metrics_row.get('seed', 'N/A')}, trial: {metrics_row.get('trial', 'N/A')}",
        f"committed: {metrics_row.get('committed_path', 'N/A')}",
        f"vte_binary: {metrics_row.get('vte_binary', 'N/A')}",
        f"raw_idphi: {metrics_row.get('raw_idphi', np.nan):.2f}",
        f"z_idphi: {metrics_row.get('z_idphi', np.nan):.2f}",
        f"pause_ticks: {metrics_row.get('pause_ticks', np.nan)}",
        f"reorient: {metrics_row.get('reorientation_count', np.nan)}",
        f"pose_source: {pose_source}"
    ]
    ax_metrics.text(0.05, 0.95, "\n".join(lines), 
                   fontsize=8, va='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))


def render_static_trial(
    trace_df: pd.DataFrame,
    metrics_row: pd.Series,
    output_path: Path,
    config: dict,
    pose_source: str = "synthetic_from_stage3_steps",
    title: Optional[str] = None
) -> None:
    """Render single-trial static visualization."""
    fig, ax = plt.subplots(1, 1, figsize=config["output"]["figsize_static"], 
                          dpi=config["output"]["dpi"])
    
    plot_maze_schematic(ax)
    plot_trajectory(ax, trace_df, config["rendering"]["trail_length"])
    
    if config["rendering"]["overlay_z_idphi"] and "z_idphi" in metrics_row:
        ax.text(0.02, 0.98, f"z_idphi: {metrics_row['z_idphi']:.2f}", 
               transform=ax.transAxes, fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    add_metrics_panel(ax, metrics_row, pose_source)
    
    if title:
        ax.set_title(title, fontsize=12, pad=20)
    
    # Methodological caption
    caption = "Synthetic trajectory reconstruction from Stage 3 step logs. " \
              "Not biological pose tracking." if pose_source.startswith("synthetic") else ""
    if caption:
        fig.text(0.5, 0.01, caption, ha='center', fontsize=7, style='italic', color='gray')
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=config["output"]["dpi"])
    plt.close(fig)