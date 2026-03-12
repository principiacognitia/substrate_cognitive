"""
Figure 3: V_G Dynamics and Hysteresis.
Концепт Gate-Rheology.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from substrate_analysis.style import setup_publication_style


def generate_figure_3(data: Dict[str, pd.DataFrame],
                      changepoint: int = 1000,
                      output_dir: str = 'logs/figures/',
                      dpi: int = 300) -> str:
    """
    Отрисовка Figure 3: Динамика V_G с агрегацией по seeds.
    """
    setup_publication_style()
    
    if 'Full' not in data:
        raise ValueError("Нет данных для агента 'Full'")
    
    df = data['Full']
    
    # Агрегируем по trial и seed
    agg = df.groupby('trial').agg({
        'V_G': ['mean', 'sem'],
        'mode': ['mean']
    }).reset_index()
    
    agg.columns = ['trial', 'V_G_mean', 'V_G_sem', 'EXPLORE_prob']
    
    # Окно вокруг смены правил
    window_start = max(0, changepoint - 200)
    window_end = min(agg['trial'].max(), changepoint + 400)
    window_mask = (agg['trial'] >= window_start) & (agg['trial'] <= window_end)
    window = agg[window_mask]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # V_G с доверительным интервалом
    ax1.plot(window['trial'], window['V_G_mean'], color='#1f77b4',
             label='V_G (Control Inertia)', linewidth=2.5)
    ax1.fill_between(window['trial'],
                     window['V_G_mean'] - window['V_G_sem'],
                     window['V_G_mean'] + window['V_G_sem'],
                     color='#1f77b4', alpha=0.3, label='± SEM')
    ax1.axvline(x=changepoint, color='black', linestyle='--', linewidth=2,
                label='Changepoint (Rule Reversal)')
    
    ax1.set_xlabel('Trial', fontsize=14)
    ax1.set_ylabel('Viscosity (V_G)', color='#1f77b4', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.grid(True, alpha=0.3)
    
    # EXPLORE probability
    ax2 = ax1.twinx()
    ax2.plot(window['trial'], window['EXPLORE_prob'], color='#ff7f0e',
             label='P(EXPLORE)', linewidth=2.5, alpha=0.8)
    ax2.set_ylabel('Probability of EXPLORE Mode', color='#ff7f0e', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    ax2.grid(False)
    
    # Объединённый legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12)
    
    plt.title('Figure 3: Gate Rheology Melting and Hysteresis',
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'Figure_3_VG_Dynamics.png')
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return filepath