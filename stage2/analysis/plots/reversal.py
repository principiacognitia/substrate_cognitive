"""
Figure 4: Reversal Learning Curves.
А-ля Le et al. 2023.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from substrate_analysis.style import setup_publication_style


def generate_figure_4(data: Dict[str, pd.DataFrame],
                      reversal_trial: int = 1000,
                      output_dir: str = 'logs/figures/',
                      dpi: int = 300) -> str:
    """
    Отрисовка Figure 4: Reversal Learning Curves.
    """
    setup_publication_style()
    
    if 'Full' not in data or 'NoVG' not in data:
        raise ValueError("Нет данных для агентов 'Full' и/или 'NoVG'")
    
    # Вычисляем P(Correct) для каждого агента
    def compute_accuracy(df, reversal_trial):
        df = df.copy()
        df['correct'] = df.apply(
            lambda row: 1 if (row['a1'] == 0 and row['trial'] <= reversal_trial) or
                         (row['a1'] == 1 and row['trial'] > reversal_trial) else 0,
            axis=1
        )
        return df
    
    full_df = compute_accuracy(data['Full'], reversal_trial)
    novg_df = compute_accuracy(data['NoVG'], reversal_trial)
    
    # Агрегируем по trial
    full_agg = full_df.groupby('trial')['correct'].agg(['mean', 'sem']).reset_index()
    novg_agg = novg_df.groupby('trial')['correct'].agg(['mean', 'sem']).reset_index()
    
    # Сглаживание
    window = 10
    full_agg['smooth'] = full_agg['mean'].rolling(window=window, min_periods=1).mean()
    novg_agg['smooth'] = novg_agg['mean'].rolling(window=window, min_periods=1).mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Full Agent
    ax.plot(full_agg['trial'], full_agg['smooth'],
            label='Full Agent (with V_G)', color='#1f77b4', linewidth=2.5, alpha=0.9)
    ax.fill_between(full_agg['trial'],
                    full_agg['smooth'] - full_agg['sem'],
                    full_agg['smooth'] + full_agg['sem'],
                    color='#1f77b4', alpha=0.2)
    
    # NoVG Agent
    ax.plot(novg_agg['trial'], novg_agg['smooth'],
            label='NoVG Agent (Ablation)', color='#d62728',
            linestyle='--', linewidth=2.5, alpha=0.9)
    ax.fill_between(novg_agg['trial'],
                    novg_agg['smooth'] - novg_agg['sem'],
                    novg_agg['smooth'] + novg_agg['sem'],
                    color='#d62728', alpha=0.2)
    
    ax.axvline(x=reversal_trial, color='black', linestyle=':', linewidth=2,
               label='Reversal Point')
    ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, label='Chance Level')
    ax.axhline(y=0.8, color='green', linestyle='-.', alpha=0.5, label='Criterion (80%)')
    
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Trial', fontsize=14)
    ax.set_ylabel('P(Correct Choice)', fontsize=14)
    ax.set_title('Figure 4: Reversal Learning & Perseveration',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'Figure_4_Reversal.png')
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return filepath