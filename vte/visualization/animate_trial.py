"""Patch 21D: Single-trial VTE animation renderer."""
from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from typing import Optional


def init_frame(ax, geometry_id: str = "fork_open_covered_v1"):
    """Initialize animation frame with maze skeleton."""
    ax.clear()
    plot_maze_schematic(ax)  # Reuse from render_trial
    ax.set_title("Frame: 0", fontsize=9)
    return []


def animate_frame(frame_idx, ax, trace_df, config):
    """Update animation frame with trajectory progress."""
    ax.clear()
    plot_maze_schematic(ax)
    
    # Plot trajectory up to current frame
    sub_trace = trace_df.iloc[:frame_idx+1]
    if not sub_trace.empty:
        plot_trajectory(ax, sub_trace, config["rendering"]["trail_length"])
        
        # Draw rat body marker
        x, y = sub_trace["x"].iloc[-1], sub_trace["y"].iloc[-1]
        heading = sub_trace["heading"].iloc[-1]
        circle = plt.Circle((x, y), 0.08, color='brown', fill=True, zorder=5)
        ax.add_patch(circle)
        
        # Heading arrow
        dx, dy = 0.12 * np.cos(heading), 0.12 * np.sin(heading)
        ax.arrow(x, y, dx, dy, head_width=0.07, head_length=0.09, 
                fc='orange', ec='orange', zorder=6)
    
    ax.set_title(f"Tick: {frame_idx}", fontsize=9)
    return []


def render_animation(
    trace_df: pd.DataFrame,
    metrics_row: pd.Series,
    output_path: Path,
    config: dict,
    pose_source: str = "synthetic_from_stage3_steps",
    title: Optional[str] = None
) -> None:
    """Render single-trial animation as GIF."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
    
    fps = config["rendering"]["fps"]
    frames = len(trace_df)
    
    anim = animation.FuncAnimation(
        fig, 
        lambda i: animate_frame(i, ax, trace_df, config),
        init_func=lambda: init_frame(ax),
        frames=frames,
        interval=1000/fps,
        blit=False,
        repeat=True
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close(fig)
    
    # Write metadata sidecar
    meta_path = output_path.with_suffix(".json")
    import json
    metadata = {
        "run_id": metrics_row.get("run_id"),
        "seed": int(metrics_row.get("seed", -1)),
        "trial": int(metrics_row.get("trial", -1)),
        "pose_source": pose_source,
        "frames": frames,
        "fps": fps,
        "committed_path": metrics_row.get("committed_path"),
        "vte_binary": int(metrics_row.get("vte_binary", -1)),
        "z_idphi": float(metrics_row.get("z_idphi", np.nan))
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)