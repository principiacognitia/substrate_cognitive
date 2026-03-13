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
from stage2.core.args import print_always

def plot_stay_probabilities(ax, data, title):
    """
    Отрисовка классического графика Stay Probability.
    
    Args:
        ax: Matplotlib axis
         DataFrame с колонками reward, trans_factor, stay
        title: Заголовок графика
    """
    conditions = []
    for reward in [1.0, 0.0]:
        for trans in [1, -1]:
            subset = data[(data['reward'] == reward) & (data['trans_factor'] == trans)]
            if len(subset) > 0:
                prob = subset['stay'].mean()
                err = subset['stay'].sem()
            else:
                prob, err = 0, 0
            conditions.append((reward, trans, prob, err))
    
    labels = ['Common', 'Rare']
    rewarded_means = [conditions[0][2], conditions[1][2]]
    rewarded_errs = [conditions[0][3], conditions[1][3]]
    unrewarded_means = [conditions[2][2], conditions[3][2]]
    unrewarded_errs = [conditions[2][3], conditions[3][3]]
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, rewarded_means, width, yerr=rewarded_errs, 
           label='Rewarded', color='#2ca02c', capsize=5, alpha=0.8)
    ax.bar(x + width/2, unrewarded_means, width, yerr=unrewarded_errs, 
           label='Unrewarded', color='#d62728', capsize=5, alpha=0.8)
    
    ax.set_ylabel('Stay Probability')
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower left')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

# ============================================================================
# FIGURE 2: MB/MF SIGNATURES (Daw et al. 2011 style)
# ============================================================================

def generate_figure_2(output_dir: str = 'logs/figures/',
                      dpi: int = 300) -> str:
    """
    Отрисовка Figure 2: MB/MF Signatures.
    Запускает агентов для генерации данных (как в оригинале).
    """
    from substrate_analysis.style import setup_publication_style
    from stage2.twostep.env_twostep import TwoStepEnv
    from stage2.core.baselines import MFAgent, MBAgent, RheologicalAgent
    
    setup_publication_style()
    
    print_always("Генерация Figure 2: MB/MF Signatures...")
    
    # Запускаем агентов для генерации данных
    env = TwoStepEnv(seed=42, n_trials=5000, with_changepoint=False)
    
    mf_data = run_agent_for_figures(env, MFAgent(beta=4.0, seed=42))
    env.reset()
    mb_data = run_agent_for_figures(env, MBAgent(beta=4.0, seed=42))
    env.reset()
    rheo_data = run_agent_for_figures(env, RheologicalAgent(beta=4.0, theta_mb=0.30, seed=42))
    
    # Строим 3 панели
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    plot_stay_probabilities(axes[0], mf_data, 'MF Agent (Model-Free)')
    plot_stay_probabilities(axes[1], mb_data, 'MB Agent (Model-Based)')
    plot_stay_probabilities(axes[2], rheo_data, 'Rheological Agent (Our Model)')
    
    plt.suptitle('Figure 2: MB/MF Signatures in Two-Step Task', 
                 fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'Figure_2_Signatures.png')
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print_always(f"Сохранено: {filepath}")
    plt.close()
    
    return filepath


def run_agent_for_figures(env, agent, n_trials=5000):
    """Вспомогательная функция для запуска агентов (как в старом figures.py)."""
    data = []
    prev_a1 = None
    prev_reward = None
    prev_trans_factor = None
    u_t_prev = np.zeros(4)
    
    for trial in range(n_trials):
        s1 = env.reset()
        
        if hasattr(agent, 'get_mode'):
            a1 = agent.select_action_stage1(s1, u_t_prev)
        else:
            a1 = agent.select_action_stage1(s1)
            
        s2, trans_type = env.step_stage1(a1)
        a2 = agent.select_action_stage2(s2)
        reward, done, info = env.step_stage2(a2)
        
        if hasattr(agent, 'get_mode'):
            u_t_prev = agent.update(a1, a2, reward, s2, trans_type, s1)
        else:
            agent.update(a1, a2, reward, s2, trans_type, s1)
            
        trans_factor = 1 if trans_type == 'common' else -1
        
        if prev_a1 is not None:
            stay = 1 if (a1 == prev_a1) else 0
            data.append({
                'trial': trial,
                'reward': prev_reward,
                'trans_factor': prev_trans_factor,
                'stay': stay
            })
        
        prev_a1 = a1
        prev_reward = reward
        prev_trans_factor = trans_factor
    
    df = pd.DataFrame(data)
    return df