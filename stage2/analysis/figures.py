"""
Генерация публикационных графиков (Publication-ready Figures) для статьи.
Отрисовывает:
1. MB/MF Signatures (Stay Probabilities) - а-ля Daw et al. 2011
2. V_G Dynamics (Расплавление) - концепт Gate-Rheology
3. Reversal Learning Curves - а-ля Le et al. 2023
"""

import sys
sys.path.insert(0, 'E:/CRS-1/substrate_cognitive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from stage2.twostep.env_twostep import TwoStepEnv
from stage2.reversal.env_reversal import ReversalEnv
from stage2.reversal.run_reversal import run_reversal_single
from stage2.core.baselines import RheologicalAgent, MFAgent, MBAgent

# Настройки стиля для научных публикаций
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

def run_agent_for_figures(env, agent, n_trials=2000):
    """Универсальный раннер, поддерживающий как простых, так и реологических агентов."""
    data =[]
    prev_a1 = None
    prev_reward = None
    prev_trans_factor = None
    u_t_prev = np.zeros(4)
    
    for trial in range(n_trials):
        s1 = env.reset()
        
        # Адаптивный вызов в зависимости от типа агента
        if hasattr(agent, 'get_mode'):
            a1 = agent.select_action_stage1(s1, u_t_prev)
        else:
            a1 = agent.select_action_stage1(s1)
            
        s2, trans_type = env.step_stage1(a1)
        a2 = agent.select_action_stage2(s2)
        reward, done, info = env.step_stage2(a2)
        
        # Адаптивное обновление
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

def plot_stay_probabilities(ax, data, title):
    """Отрисовка классического графика Stay Probability (Daw 2011)."""
    conditions = []
    for reward in[1.0, 0.0]:
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
    
    ax.bar(x - width/2, rewarded_means, width, yerr=rewarded_errs, label='Rewarded', color='#2ca02c', capsize=5)
    ax.bar(x + width/2, unrewarded_means, width, yerr=unrewarded_errs, label='Unrewarded', color='#d62728', capsize=5)
    
    ax.set_ylabel('Stay Probability')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower left')

def generate_figure_2_signatures():
    """Сбор данных и отрисовка Figure 2: MB/MF Signatures."""
    print("Генерация Figure 2: Two-Step Signatures...")
    env = TwoStepEnv(seed=42, n_trials=5000, with_changepoint=False)
    
    mf_data = run_agent_for_figures(env, MFAgent(beta=4.0, seed=42), n_trials=5000)
    env.reset()
    mb_data = run_agent_for_figures(env, MBAgent(beta=4.0, seed=42), n_trials=5000)
    env.reset()
    rheo_data = run_agent_for_figures(env, RheologicalAgent(beta=4.0, theta_mb=0.30, seed=42), n_trials=5000)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_stay_probabilities(axes[0], mf_data, 'MF Agent (Simulated)')
    plot_stay_probabilities(axes[1], mb_data, 'MB Agent (Simulated)')
    plot_stay_probabilities(axes[2], rheo_data, 'Rheological Agent (Our Model)')
    
    plt.tight_layout()
    plt.savefig('logs/twostep/Figure_2_Signatures.png', dpi=300)
    print("Сохранено: logs/twostep/Figure_2_Signatures.png")

def generate_figure_3_vg_dynamics():
    """Сбор данных и отрисовка Figure 3: Динамика расплавления V_G."""
    print("Генерация Figure 3: V_G Dynamics and Hysteresis...")
    n_trials = 2000
    changepoint = 1000
    env = TwoStepEnv(n_trials=n_trials, changepoint_trial=changepoint, seed=42, with_changepoint=True)
    agent = RheologicalAgent(theta_mb=0.30, seed=42)
    
    v_g_history = []
    mode_history =[]
    
    u_t_prev = np.zeros(4)
    for trial in range(1, n_trials + 1):
        s1 = env.reset()
        a1 = agent.select_action_stage1(s1, u_t_prev)
        mode = agent.get_mode()
        
        s2, trans_type = env.step_stage1(a1)
        a2 = agent.select_action_stage2(s2)
        reward, done, info = env.step_stage2(a2)
        u_t_prev = agent.update(a1, a2, reward, s2, trans_type, s1)
        
        v_g_history.append(agent.V_G)
        mode_history.append(1 if mode == 'EXPLORE' else 0)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    window = range(800, 1200) # Показываем окно вокруг смены правил
    ax1.plot(window, np.array(v_g_history)[window], color='blue', label='V_G (Control Inertia)', linewidth=2)
    ax1.axvline(x=changepoint, color='black', linestyle='--', label='Changepoint')
    
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Viscosity (V_G)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()
    explore_smooth = pd.Series(mode_history).rolling(window=10).mean()
    ax2.plot(window, explore_smooth[window], color='orange', label='P(EXPLORE)', linewidth=2, alpha=0.7)
    ax2.set_ylabel('Probability of EXPLORE Mode', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    plt.title('Figure 3: Gate Rheology Melting and Hysteresis')
    fig.tight_layout()
    plt.savefig('logs/twostep/Figure_3_VG_Dynamics.png', dpi=300)
    print("Сохранено: logs/twostep/Figure_3_VG_Dynamics.png")

def generate_figure_4_reversal():
    """Сбор данных и отрисовка Figure 4: Reversal Learning Curves."""
    print("Генерация Figure 4: Reversal Learning Curves...")
    
    n_trials = 300
    rev_trial = 150
    n_seeds = 30
    
    accuracy_full = np.zeros(n_trials)
    accuracy_novg = np.zeros(n_trials)
    
    for seed in range(n_seeds):
        # Full Agent
        df_full = run_reversal_single(RheologicalAgent, n_trials=n_trials, reversal_trial=rev_trial, seed=seed)
        correct_action_full = df_full.apply(lambda row: 1 if (row['a1'] == 0 and row['trial'] <= rev_trial) or (row['a1'] == 1 and row['trial'] > rev_trial) else 0, axis=1)
        accuracy_full += correct_action_full.values
        
        # NoVG Agent
        from stage2.core.baselines import RheologicalAgent_NoVG
        df_novg = run_reversal_single(RheologicalAgent_NoVG, n_trials=n_trials, reversal_trial=rev_trial, seed=seed)
        correct_action_novg = df_novg.apply(lambda row: 1 if (row['a1'] == 0 and row['trial'] <= rev_trial) or (row['a1'] == 1 and row['trial'] > rev_trial) else 0, axis=1)
        accuracy_novg += correct_action_novg.values

    accuracy_full /= n_seeds
    accuracy_novg /= n_seeds
    
    # Сглаживание
    acc_full_smooth = pd.Series(accuracy_full).rolling(window=10, min_periods=1).mean()
    acc_novg_smooth = pd.Series(accuracy_novg).rolling(window=10, min_periods=1).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(acc_full_smooth, label='Full Agent (with V_G)', color='blue', linewidth=2)
    plt.plot(acc_novg_smooth, label='NoVG Agent (Ablation)', color='red', linestyle='--', linewidth=2)
    
    plt.axvline(x=rev_trial, color='black', linestyle=':', label='Reversal Point')
    plt.axhline(y=0.5, color='gray', linestyle='-', alpha=0.5)
    
    plt.ylim(0, 1.05)
    plt.xlabel('Trial')
    plt.ylabel('P(Correct Choice)')
    plt.title('Figure 4: Reversal Learning & Perseveration')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('logs/twostep/Figure_4_Reversal.png', dpi=300)
    print("Сохранено: logs/twostep/Figure_4_Reversal.png")

if __name__ == "__main__":
    import os
    if not os.path.exists('logs/twostep'):
        os.makedirs('logs/twostep')
        
    generate_figure_2_signatures()
    generate_figure_3_vg_dynamics()
    generate_figure_4_reversal()
    print("\nВсё готово. Проверьте графики в папке logs/twostep/!")