"""
Генерация публикационных графиков (Publication-ready Figures) для статьи.

Отрисовывает:
1. MB/MF Signatures (Stay Probabilities) — а-ля Daw et al. 2011
2. V_G Dynamics (Расплавление) — концепт Gate-Rheology
3. Reversal Learning Curves — а-ля Le et al. 2023

Запуск:
    python -m stage2.analysis.figures --output-dir logs/figures/ --n-seeds 30
    python -m stage2.analysis.figures --nodebug --dpi 300
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from stage2.twostep.env_twostep import TwoStepEnv
from stage2.reversal.env_reversal import ReversalEnv
from stage2.reversal.run_reversal import run_reversal_single
from stage2.core.baselines import (
    RheologicalAgent, 
    RheologicalAgent_NoVG,
    MFAgent, 
    MBAgent
)
from stage2.core.args import parse_args, print_always, print_debug

# ============================================================================
# СТИЛЬ ДЛЯ ПУБЛИКАЦИЙ
# ============================================================================

def setup_publication_style():
    """Настраивает стиль matplotlib для научных публикаций."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.figsize': (10, 6),
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

# ============================================================================
# УНИВЕРСАЛЬНЫЙ РАННЕР
# ============================================================================

def run_agent_for_figures(env, agent, n_trials=2000, verbose=False, nodebug=True):
    """
    Универсальный раннер, поддерживающий как простых, так и реологических агентов.
    
    Args:
        env: Среда (TwoStepEnv или ReversalEnv)
        agent: Агент
        n_trials: Количество триалов
        verbose: Выводить ли отладочную информацию
        nodebug: Отключить ли вывод
        
    Returns:
        DataFrame с данными триалов
    """
    data = []
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
        
        # Отладочный вывод
        if verbose and not nodebug:
            if trial % 500 == 0:
                print(f"  Trial {trial}...")
        
    return pd.DataFrame(data)

# ============================================================================
# FIGURE 2: MB/MF SIGNATURES (Daw et al. 2011 style)
# ============================================================================

def plot_stay_probabilities(ax, data, title):
    """
    Отрисовка классического графика Stay Probability.
    
    Args:
        ax: Matplotlib axis
        data: DataFrame с колонками reward, trans_factor, stay
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


def generate_figure_2_signatures(output_dir='logs/figures/', n_trials=5000, 
                                  seed=42, dpi=300, verbose=False, nodebug=True):
    """
    Сбор данных и отрисовка Figure 2: MB/MF Signatures.
    
    Args:
        output_dir: Директория для сохранения
        n_trials: Количество триалов на агента
        seed: Random seed
        dpi: Разрешение сохранения
        verbose: Выводить ли прогресс
        nodebug: Отключить ли вывод
    """
    print_always("Генерация Figure 2: Two-Step Signatures...")
    
    env = TwoStepEnv(seed=seed, n_trials=n_trials, with_changepoint=False)
    
    mf_data = run_agent_for_figures(env, MFAgent(beta=4.0, seed=seed), 
                                     n_trials=n_trials, verbose=verbose, nodebug=nodebug)
    env.reset()
    mb_data = run_agent_for_figures(env, MBAgent(beta=4.0, seed=seed), 
                                     n_trials=n_trials, verbose=verbose, nodebug=nodebug)
    env.reset()
    rheo_data = run_agent_for_figures(env, RheologicalAgent(beta=4.0, theta_mb=0.30, seed=seed), 
                                       n_trials=n_trials, verbose=verbose, nodebug=nodebug)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_stay_probabilities(axes[0], mf_data, 'MF Agent (Model-Free)')
    plot_stay_probabilities(axes[1], mb_data, 'MB Agent (Model-Based)')
    plot_stay_probabilities(axes[2], rheo_data, 'Rheological Agent (Our Model)')
    
    plt.suptitle('Figure 2: MB/MF Signatures in Two-Step Task', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'Figure_2_Signatures.png')
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print_always(f"Сохранено: {filepath}")
    plt.close()

# ============================================================================
# FIGURE 3: V_G DYNAMICS AND HYSTERESIS
# ============================================================================

def generate_figure_3_vg_dynamics(output_dir='logs/figures/', n_trials=2000, 
                                   changepoint=1000, seed=42, theta_mb=0.30,
                                   dpi=300, verbose=False, nodebug=True):
    """
    Сбор данных и отрисовка Figure 3: Динамика расплавления V_G.
    
    Args:
        output_dir: Директория для сохранения
        n_trials: Количество триалов
        changepoint: Триал смены правил
        seed: Random seed
        theta_mb: Порог переключения
        dpi: Разрешение сохранения
        verbose: Выводить ли прогресс
        nodebug: Отключить ли вывод
    """
    print_always("Генерация Figure 3: V_G Dynamics and Hysteresis...")
    
    env = TwoStepEnv(n_trials=n_trials, changepoint_trial=changepoint, 
                     seed=seed, with_changepoint=True)
    agent = RheologicalAgent(theta_mb=theta_mb, seed=seed)
    
    v_g_history = []
    mode_history = []
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
        
        if verbose and not nodebug:
            if trial in [changepoint-10, changepoint, changepoint+10, changepoint+50]:
                print(f"  Trial {trial}: V_G={agent.V_G:.3f}, Mode={mode}")

    # Окно вокруг смены правил
    window_start = max(0, changepoint - 200)
    window_end = min(n_trials, changepoint + 400)
    window = range(window_start, window_end)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.plot(window, np.array(v_g_history)[window], color='#1f77b4', 
             label='V_G (Control Inertia)', linewidth=2.5)
    ax1.axvline(x=changepoint, color='black', linestyle='--', linewidth=2, 
                label='Changepoint (Rule Reversal)')
    
    ax1.set_xlabel('Trial', fontsize=14)
    ax1.set_ylabel('Viscosity (V_G)', color='#1f77b4', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    explore_smooth = pd.Series(mode_history).rolling(window=20, min_periods=1).mean()
    ax2.plot(window, explore_smooth[window], color='#ff7f0e', 
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
    
    filepath = os.path.join(output_dir, 'Figure_3_VG_Dynamics.png')
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print_always(f"Сохранено: {filepath}")
    plt.close()

# ============================================================================
# FIGURE 4: REVERSAL LEARNING CURVES (Le et al. 2023 style)
# ============================================================================

def generate_figure_4_reversal(output_dir='logs/figures/', n_trials=300, 
                                rev_trial=150, n_seeds=30, seed_start=42,
                                dpi=300, verbose=False, nodebug=True):
    """
    Сбор данных и отрисовка Figure 4: Reversal Learning Curves.
    
    Args:
        output_dir: Директория для сохранения
        n_trials: Количество триалов
        rev_trial: Триал реверсала
        n_seeds: Количество семян для усреднения
        seed_start: Начальное значение seed
        dpi: Разрешение сохранения
        verbose: Выводить ли прогресс
        nodebug: Отключить ли вывод
    """
    print_always(f"Генерация Figure 4: Reversal Learning Curves ({n_seeds} seeds)...")
    
    accuracy_full = np.zeros(n_trials)
    accuracy_novg = np.zeros(n_trials)
    
    for i in range(n_seeds):
        seed = seed_start + i
        
        if verbose and not nodebug:
            if (i + 1) % 10 == 0:
                print(f"  Seed прогресс: {i + 1}/{n_seeds}")
        
        # Full Agent
        df_full = run_reversal_single(RheologicalAgent, n_trials=n_trials, 
                                       reversal_trial=rev_trial, seed=seed)
        correct_action_full = df_full.apply(
            lambda row: 1 if (row['a1'] == 0 and row['trial'] <= rev_trial) or 
                         (row['a1'] == 1 and row['trial'] > rev_trial) else 0, 
            axis=1
        )
        accuracy_full += correct_action_full.values
        
        # NoVG Agent
        df_novg = run_reversal_single(RheologicalAgent_NoVG, n_trials=n_trials, 
                                       reversal_trial=rev_trial, seed=seed)
        correct_action_novg = df_novg.apply(
            lambda row: 1 if (row['a1'] == 0 and row['trial'] <= rev_trial) or 
                         (row['a1'] == 1 and row['trial'] > rev_trial) else 0, 
            axis=1
        )
        accuracy_novg += correct_action_novg.values

    accuracy_full /= n_seeds
    accuracy_novg /= n_seeds
    
    # Сглаживание
    acc_full_smooth = pd.Series(accuracy_full).rolling(window=10, min_periods=1).mean()
    acc_novg_smooth = pd.Series(accuracy_novg).rolling(window=10, min_periods=1).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(acc_full_smooth, label='Full Agent (with V_G)', color='#1f77b4', 
             linewidth=2.5, alpha=0.9)
    plt.plot(acc_novg_smooth, label='NoVG Agent (Ablation)', color='#d62728', 
             linestyle='--', linewidth=2.5, alpha=0.9)
    
    plt.axvline(x=rev_trial, color='black', linestyle=':', linewidth=2, 
                label='Reversal Point')
    plt.axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, label='Chance Level')
    plt.axhline(y=0.8, color='green', linestyle='-.', alpha=0.5, label='Criterion (80%)')
    
    plt.ylim(0, 1.05)
    plt.xlabel('Trial', fontsize=14)
    plt.ylabel('P(Correct Choice)', fontsize=14)
    plt.title('Figure 4: Reversal Learning & Perseveration', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'Figure_4_Reversal.png')
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print_always(f"Сохранено: {filepath}")
    plt.close()

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Точка входа для генерации всех фигур."""
    args = parse_args(description="Stage 2: Generate Publication-Ready Figures")
    
    # Дополнительные аргументы для фигур
    import argparse
    parser = argparse.ArgumentParser(parents=[argparse.ArgumentParser(add_help=False)])
    parser.add_argument('--output-dir', type=str, default='logs/figures/',
                       help='Директория для сохранения графиков')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Разрешение сохранения (default: 300)')
    parser.add_argument('--n-seeds', type=int, default=30,
                       help='Количество семян для Reversal (default: 30)')
    parser.add_argument('--figure', type=str, default='all',
                       choices=['all', '2', '3', '4'],
                       help='Какую фигуру генерировать (default: all)')
    
    # Парсим известные аргументы, игнорируем неизвестные
    fig_args, _ = parser.parse_known_args()
    
    # Создаем директорию
    os.makedirs(fig_args.output_dir, exist_ok=True)
    
    # Настраиваем стиль
    setup_publication_style()
    
    print_always("=" * 70)
    print_always("Stage 2: Генерация публикационных графиков")
    print_always("=" * 70)
    print_always(f"Output Directory: {fig_args.output_dir}")
    print_always(f"DPI: {fig_args.dpi}")
    print_always(f"N Seeds (Reversal): {fig_args.n_seeds}")
    print_always("")
    
    if fig_args.figure in ['all', '2']:
        generate_figure_2_signatures(
            output_dir=fig_args.output_dir,
            dpi=fig_args.dpi,
            verbose=args.verbose,
            nodebug=args.nodebug
        )
        print_always("")
    
    if fig_args.figure in ['all', '3']:
        generate_figure_3_vg_dynamics(
            output_dir=fig_args.output_dir,
            dpi=fig_args.dpi,
            theta_mb=args.theta_mb,
            verbose=args.verbose,
            nodebug=args.nodebug
        )
        print_always("")
    
    if fig_args.figure in ['all', '4']:
        generate_figure_4_reversal(
            output_dir=fig_args.output_dir,
            n_seeds=fig_args.n_seeds,
            dpi=fig_args.dpi,
            verbose=args.verbose,
            nodebug=args.nodebug
        )
        print_always("")
    
    print_always("=" * 70)
    print_always("✓ Все графики сгенерированы!")
    print_always(f"Проверьте папку: {os.path.abspath(fig_args.output_dir)}")
    print_always("=" * 70)


if __name__ == "__main__":
    main()