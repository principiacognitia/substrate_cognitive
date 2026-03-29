"""
Stage 3.1A: Analyze Path Choice.

Generates Figure 3.1A: Path preference under exposure conflict.

Usage:
    python -m stage3.analysis.analyze_path_choice --input-dir logs/stage3/stage3_1a/
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def load_trial_data(input_dir: str) -> pd.DataFrame:
    """Загружает все trial summaries."""
    input_path = Path(input_dir)
    
    all_trials = []
    for log_file in input_path.glob('*_trials.csv'):
        df = pd.read_csv(log_file)
        all_trials.append(df)
    
    return pd.concat(all_trials, ignore_index=True)


def plot_path_choice(df: pd.DataFrame, output_path: str):
    """
    Figure 3.1A: Path choice distribution.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Count path choices
    path_counts = df['path_choice'].value_counts()
    total = len(df)
    
    # Bar plot
    bars = ax.bar(
        ['Open Path', 'Covered Path'],
        [path_counts.get('open', 0), path_counts.get('covered', 0)],
        color=['#ff7f0e', '#2ca02c'],
        edgecolor='black',
        linewidth=1.5
    )
    
    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        percentage = (height / total) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + total * 0.01,
            f'{percentage:.1f}%',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )
    
    ax.set_ylabel('Number of Trials', fontsize=14)
    ax.set_title(
        'Figure 3.1A: Path Choice Under Exposure Conflict\n(Stage 3.1A, 30 seeds)',
        fontsize=16,
        fontweight='bold'
    )
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Figure 3.1A saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Stage 3.1A Path Choice')
    parser.add_argument('--input-dir', type=str, default='logs/stage3/stage3_1a/', help='Input directory')
    parser.add_argument('--output-dir', type=str, default='logs/figures/stage3/', help='Output directory')
    
    args = parser.parse_args()
    
    # Load data
    df = load_trial_data(args.input_dir)
    print(f"Loaded {len(df)} trials from {args.input_dir}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate figure
    plot_path_choice(df, str(output_path / 'Figure_3_1A_Path_Choice.png'))
    
    # Print summary statistics
    print("\n=== Path Choice Summary ===")
    print(df['path_choice'].value_counts())
    print(f"\nTotal trials: {len(df)}")
    print(f"Covered preference: {(df['path_choice'] == 'covered').mean() * 100:.1f}%")


if __name__ == "__main__":
    main()