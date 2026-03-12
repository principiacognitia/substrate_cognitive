"""
Figure 2: MB/MF Signatures (Stay Probabilities).
А-ля Daw et al. 2011.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from substrate_analysis.style import setup_publication_style


def generate_figure_2(data: Dict[str, pd.DataFrame],
                      output_dir: str = 'logs/figures/',
                      dpi: int = 300) -> str:
    """
    Отрисовка Figure 2: MB/MF Signatures.
    """
    setup_publication_style()
    
    # Определяем какой агент использовать для графика
    if 'RheologicalAgent' in data:
        df = data['RheologicalAgent']
        title = 'Rheological Agent (Our Model)'
    elif 'Full' in data:
        df = data['Full']
        title = 'Full Agent (V_G + V_p)'
    elif 'MB' in data:
        df = data['MB']
        title = 'MB Agent (Model-Based)'
    else:
        raise ValueError("Нет данных для агента (нужен RheologicalAgent/Full/MB)")
    
    # Вычисляем stay probabilities по 4 условиям
    conditions = []
    for reward in [1.0, 0.0]:
        for trans in [1, -1]:
            subset = df[(df['reward'] == reward) & (df['trans_factor'] == trans)]
            if len(subset) > 0:
                prob = subset['stay'].mean()
                sem = subset['stay'].sem()
            else:
                prob, sem = 0, 0
            conditions.append({'prob': prob, 'sem': sem})
    
    # Строим график
    labels = ['Common', 'Rare']
    x = np.arange(len(labels))
    width = 0.35
    
    rewarded_means = [conditions[0]['prob'], conditions[1]['prob']]
    rewarded_sems = [conditions[0]['sem'], conditions[1]['sem']]
    unrewarded_means = [conditions[2]['prob'], conditions[3]['prob']]
    unrewarded_sems = [conditions[2]['sem'], conditions[3]['sem']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - width/2, rewarded_means, width, yerr=rewarded_sems,
           label='Rewarded', color='#2ca02c', capsize=5, alpha=0.8)
    ax.bar(x + width/2, unrewarded_means, width, yerr=unrewarded_sems,
           label='Unrewarded', color='#d62728', capsize=5, alpha=0.8)
    
    ax.set_ylabel('Stay Probability', fontsize=14)
    ax.set_xlabel('Transition Type', fontsize=14)
    ax.set_title(f'Figure 2: MB/MF Signatures\n{title}', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower left', fontsize=12)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'Figure_2_Signatures.png')
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return filepath