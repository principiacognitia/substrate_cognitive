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
from stage2.core.args import print_always 



def generate_figure_4(data: Dict[str, pd.DataFrame],
                      reversal_trial: int = 1000,
                      output_dir: str = 'logs/figures/',
                      dpi: int = 300) -> str:
    """
    Отрисовка Figure 4: Reversal Learning Curves.
    """
    setup_publication_style()

    print_always("Генерация Figure 4: Reversal Learning...")
    
    if 'Full' not in data or 'NoVG' not in data or 'NoVp' not in data:
        raise ValueError("Нет данных для агентов 'Full' и/или 'NoVG' и/или 'NoVp'")
    
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
    novp_df = compute_accuracy(data['NoVp'], reversal_trial)
    # Агрегируем по trial
    full_agg = full_df.groupby('trial')['correct'].agg(['mean', 'sem']).reset_index()
    novg_agg = novg_df.groupby('trial')['correct'].agg(['mean', 'sem']).reset_index()
    novp_agg = novp_df.groupby('trial')['correct'].agg(['mean', 'sem']).reset_index()

    # Сглаживание
    window = 10
    full_agg['smooth'] = full_agg['mean'].rolling(window=window, min_periods=1).mean()
    novg_agg['smooth'] = novg_agg['mean'].rolling(window=window, min_periods=1).mean()
    novp_agg['smooth'] = novp_agg['mean'].rolling(window=window, min_periods=1).mean()

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
    
    # NoVp Agent
    ax.plot(novp_agg['trial'], novp_agg['smooth'],
            label='NoVp Agent (Ablation)', color='#ca982a',
            linestyle='-.', linewidth=2.5, alpha=0.9)
    ax.fill_between(novp_agg['trial'],
                    novp_agg['smooth'] - novp_agg['sem'],
                    novp_agg['smooth'] + novp_agg['sem'],
                    color="#ca982a", alpha=0.2)

    ax.axvline(x=reversal_trial, color='black', linestyle=':', linewidth=2,
               label='Reversal Point')
    ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, label='Chance Level')
    ax.axhline(y=0.8, color='green', linestyle='-.', alpha=0.5, label='Criterion (80%)')
    
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Trial', fontsize=14)
    ax.set_ylabel('P(Correct Choice)', fontsize=14)
    ax.set_title('Figure 4A: Reversal Learning & Perseveration',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'Figure_4_Reversal.png')
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()

    return filepath

def generate_figure_4b(data: Dict[str, pd.DataFrame],
                       reversal_trial: int = 1000,
                       window_before: int = 50,
                       window_after: int = 100,
                       output_dir: str = 'logs/figures/',
                       dpi: int = 300) -> str:
    """
    Figure 4B: Reversal Learning — Zoomed view around changepoint.
    """
    from substrate_analysis.style import setup_publication_style
    setup_publication_style()
    
    print_always("Генерация Figure 4B: Reversal Learning (Zoomed)...")

    # КОНВЕРТАЦИЯ: гарантируем что reversal_trial это int, а window_before/after тоже int
    reversal_trial = int(reversal_trial)
    window_before = int(window_before)
    window_after = int(window_after)
    
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
    novp_df = compute_accuracy(data['NoVp'], reversal_trial) if 'NoVp' in data else None
    
    # Агрегируем по trial
    full_agg = full_df.groupby('trial')['correct'].agg(['mean', 'sem']).reset_index()
    novg_agg = novg_df.groupby('trial')['correct'].agg(['mean', 'sem']).reset_index()
    novp_agg = novp_df.groupby('trial')['correct'].agg(['mean', 'sem']).reset_index() if novp_df is not None else None
    
    # Окно вокруг реверсала
    window_start = reversal_trial - window_before
    window_end = reversal_trial + window_after
    
    full_window = full_agg[(full_agg['trial'] >= window_start) & (full_agg['trial'] <= window_end)]
    novg_window = novg_agg[(novg_agg['trial'] >= window_start) & (novg_agg['trial'] <= window_end)]
    novp_window = novp_agg[(novp_agg['trial'] >= window_start) & (novp_agg['trial'] <= window_end)] if novp_agg is not None else None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Full Agent
    ax.plot(full_window['trial'], full_window['mean'],
            label='Full Agent (with V_G)', color='#1f77b4', linewidth=2.5, alpha=0.9)
    ax.fill_between(full_window['trial'],
                    full_window['mean'] - full_window['sem'],
                    full_window['mean'] + full_window['sem'],
                    color='#1f77b4', alpha=0.2)
    
    # NoVG Agent
    ax.plot(novg_window['trial'], novg_window['mean'],
            label='NoVG Agent (V_G=0)', color='#d62728',
            linestyle='--', linewidth=2.5, alpha=0.9)
    ax.fill_between(novg_window['trial'],
                    novg_window['mean'] - novg_window['sem'],
                    novg_window['mean'] + novg_window['sem'],
                    color='#d62728', alpha=0.2)
    
    # NoVp Agent (если есть)
    if novp_window is not None:
        ax.plot(novp_window['trial'], novp_window['mean'],
                label='NoVp Agent (V_p=0)', color='#2ca02c',
                linestyle='-.', linewidth=2.5, alpha=0.9)
        ax.fill_between(novp_window['trial'],
                        novp_window['mean'] - novp_window['sem'],
                        novp_window['mean'] + novp_window['sem'],
                        color='#2ca02c', alpha=0.2)
    
    ax.axvline(x=reversal_trial, color='black', linestyle=':', linewidth=2,
               label='Reversal Point')
    ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, label='Chance Level')
    ax.axhline(y=0.8, color='green', linestyle='-.', alpha=0.5, label='Criterion (80%)')
    
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Trial (relative to reversal)', fontsize=14)
    ax.set_ylabel('P(Correct Choice)', fontsize=14)
    ax.set_title('Figure 4B: Reversal Learning — Dynamics Around Changepoint',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'Figure_4B_Reversal_Zoomed.png')
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print_always(f"Сохранено: {filepath}")
    plt.close()
    
    return filepath