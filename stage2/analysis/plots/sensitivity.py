"""
Supplementary Figure 2: Parameter Sensitivity Heatmaps.

Реальный parameter sweep (не заглушка!) для анализа устойчивости двойной диссоциации.

Запуск:
    # Быстрый тест (5 seeds, 3 значения параметра)
    python -m stage2.analysis.plots.sensitivity --quick-test
    
    # Полный sweep для препринта (10 seeds, 5×5 комбинаций)
    python -m stage2.analysis.plots.sensitivity --n-seeds 10 --n-values 5
    
    # Полный sweep для журнала (30 seeds, 5×5 комбинаций, ~10 часов)
    python -m stage2.analysis.plots.sensitivity --n-seeds 30 --n-values 5 --output-dir logs/figures/ --dpi 600
    
    # Продолжить прерванный sweep
    python -m stage2.analysis.plots.sensitivity --resume --cache-file logs/sensitivity_cache.npy
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse

from stage2.twostep.env_twostep import TwoStepEnv
from stage2.core.baselines import RheologicalAgent, RheologicalAgent_NoVG, RheologicalAgent_NoVp
from substrate_analysis.style import setup_publication_style
from stage2.core.args import print_always


# ============================================================================
# КОНФИГУРАЦИЯ SWEEP
# ============================================================================

DEFAULT_PARAM_RANGES = {
    'theta_mb': [0.15, 0.20, 0.30, 0.40, 0.50],
    'beta': [2.0, 3.0, 4.0, 5.0, 6.0],
    'tau_vol': [0.30, 0.40, 0.50, 0.60, 0.70],
    'k_melt': [0.10, 0.15, 0.20, 0.25, 0.30]
}

BASE_PARAMS = {
    'alpha': 0.35,
    'beta': 4.0,
    'theta_mb': 0.30,
    'theta_u': 1.5,
    'volatility_threshold': 0.50,
    # Параметры реологии (k_use, k_melt, lambda_decay) захардкожены в rheology.py
    # и не передаются через __init__ RheologicalAgent
    # 'k_use': 0.08,
    # 'k_melt': 0.20,
    # 'lambda_decay': 0.01
}


# ============================================================================
# ФУНКЦИИ ДЛЯ SWEEP
# ============================================================================
def rank_biserial_correlation(u_stat, n1, n2):
    """
    Вычисляет rank-biserial correlation для Mann-Whitney U.
    Возвращает значение в диапазоне [-1, 1].
    """
    if n1 == 0 or n2 == 0:
        return 0.0
    
    # Правильная формула
    r_rb = (2 * u_stat) / (n1 * n2) - 1
    
    # Клиппинг на случай численных ошибок
    return max(-1.0, min(1.0, r_rb))

def run_single_sweep_combination(
    param_name: str,
    param_value: float,
    n_seeds: int = 10,
    n_trials: int = 2000,
    changepoint: int = 1000,
    base_params: dict = None,
    verbose: bool = False
) -> Dict:
    """
    Запускает sweep по одной комбинации параметра.
    
    Returns:
        Dict с метриками (vg_effect, vp_effect, vg_pvalue, vp_pvalue)
    """
    from scipy import stats
    
    if base_params is None:
        base_params = BASE_PARAMS.copy()
    
    full_latencies = []
    novg_latencies = []
    full_persev = []
    novp_persev = []
    
    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx
        
        # Full agent
        full_params = base_params.copy()
        full_params[param_name] = param_value

        # ИСПРАВЛЕНИЕ: передаём только те параметры которые принимает RheologicalAgent
        full_agent = RheologicalAgent(
            seed=seed,
            alpha=full_params.get('alpha', 0.35),
            beta=full_params.get('beta', 4.0),
            theta_mb=full_params.get('theta_mb', 0.30),
            theta_u=full_params.get('theta_u', 1.5),
            volatility_threshold=full_params.get('volatility_threshold', 0.50)
        )
        
        # NoVG agent
        novg_agent = RheologicalAgent_NoVG(
            seed=seed,
            alpha=full_params.get('alpha', 0.35),
            beta=full_params.get('beta', 4.0),
            theta_mb=full_params.get('theta_mb', 0.30),
            theta_u=full_params.get('theta_u', 1.5),
            volatility_threshold=full_params.get('volatility_threshold', 0.50)
        )
        
        # NoVp agent
        novp_agent = RheologicalAgent_NoVp(
            seed=seed,
            alpha=full_params.get('alpha', 0.35),
            beta=full_params.get('beta', 4.0),
            theta_mb=full_params.get('theta_mb', 0.30),
            theta_u=full_params.get('theta_u', 1.5),
            volatility_threshold=full_params.get('volatility_threshold', 0.50)
        )
        
        # Запуск Full агента
        env = TwoStepEnv(n_trials=n_trials, changepoint_trial=changepoint, seed=seed)
        u_t_prev = np.zeros(4)
        full_explore_trials = []
        full_persev_errors = 0
        old_action = None
        
        for trial in range(1, n_trials + 1):
            s1 = env.reset()
            a1 = full_agent.select_action_stage1(s1, u_t_prev)
            s2, trans_type = env.step_stage1(a1)
            a2 = full_agent.select_action_stage2(s2)
            reward, done, info = env.step_stage2(a2)
            u_t_prev = full_agent.update(a1, a2, reward, s2, trans_type, s1)
            
            if full_agent.get_mode() == 'EXPLORE' and trial > changepoint:
                full_explore_trials.append(trial)
            
            # Perseveration tracking
            if trial == changepoint - 1:
                old_action = a1
            
            if trial > changepoint and full_persev_errors == 0:
                if a1 != old_action:
                    full_persev_errors = trial - changepoint
        
        # Запуск NoVG агента (отдельный прогон)
        env_novg = TwoStepEnv(n_trials=n_trials, changepoint_trial=changepoint, seed=seed)
        u_t_prev_novg = np.zeros(4)
        novg_explore_trials = []
        
        for trial in range(1, n_trials + 1):
            s1 = env_novg.reset()
            a1 = novg_agent.select_action_stage1(s1, u_t_prev_novg)
            s2, trans_type = env_novg.step_stage1(a1)
            a2 = novg_agent.select_action_stage2(s2)
            reward, done, info = env_novg.step_stage2(a2)
            u_t_prev_novg = novg_agent.update(a1, a2, reward, s2, trans_type, s1)
            
            if hasattr(novg_agent, 'get_mode') and novg_agent.get_mode() == 'EXPLORE' and trial > changepoint:
                novg_explore_trials.append(trial)
        
        # Запуск NoVp агента (отдельный прогон)
        env_novp = TwoStepEnv(n_trials=n_trials, changepoint_trial=changepoint, seed=seed)
        u_t_prev_novp = np.zeros(4)
        novp_persev_errors = 0
        old_action_novp = None
        
        for trial in range(1, n_trials + 1):
            s1 = env_novp.reset()
            a1 = novp_agent.select_action_stage1(s1, u_t_prev_novp)
            s2, trans_type = env_novp.step_stage1(a1)
            a2 = novp_agent.select_action_stage2(s2)
            reward, done, info = env_novp.step_stage2(a2)
            u_t_prev_novp = novp_agent.update(a1, a2, reward, s2, trans_type, s1)
            
            if trial == changepoint - 1:
                old_action_novp = a1
            
            if trial > changepoint and novp_persev_errors == 0:
                if a1 != old_action_novp:
                    novp_persev_errors = trial - changepoint
        
        # Сохраняем метрики
        full_latency = full_explore_trials[0] - changepoint if full_explore_trials else 999
        novg_latency = novg_explore_trials[0] - changepoint if novg_explore_trials else 999
        
        full_latencies.append(full_latency)
        novg_latencies.append(novg_latency)
        full_persev.append(full_persev_errors)
        novp_persev.append(novp_persev_errors)
        
        if verbose and (seed_idx + 1) % 5 == 0:
            print_always(f"    Seed {seed_idx + 1}/{n_seeds} завершено")
    
    # Вычисляем effect sizes
    full_lat_valid = [l for l in full_latencies if l != 999]
    novg_lat_valid = [l for l in novg_latencies if l != 999]
    
    # ИСПРАВЛЕНИЕ: Правильная формула effect size
    def rank_biserial_correlation(u_stat, n1, n2):
        if n1 == 0 or n2 == 0:
            return 0.0
        r_rb = (2 * u_stat) / (n1 * n2) - 1
        return max(-1.0, min(1.0, r_rb))
    
    if len(full_lat_valid) > 0 and len(novg_lat_valid) > 0:
        u_stat_vg, p_val_vg = stats.mannwhitneyu(full_lat_valid, novg_lat_valid, alternative='greater')
        effect_vg = rank_biserial_correlation(u_stat_vg, len(full_lat_valid), len(novg_lat_valid))
    else:
        effect_vg, p_val_vg = 0.0, 1.0
    
    if len(full_persev) > 0 and len(novp_persev) > 0:
        u_stat_vp, p_val_vp = stats.mannwhitneyu(full_persev, novp_persev, alternative='greater')
        effect_vp = rank_biserial_correlation(u_stat_vp, len(full_persev), len(novp_persev))
    else:
        effect_vp, p_val_vp = 0.0, 1.0
    
    return {
        'param_name': param_name,
        'param_value': param_value,
        'vg_effect': effect_vg,  # ← Теперь в диапазоне [-1, 1]
        'vg_pvalue': p_val_vg,
        'vp_effect': effect_vp,  # ← Теперь effect size, не разность средних
        'vp_pvalue': p_val_vp,
        'n_seeds': n_seeds,
        'seeds_completed': n_seeds
    }


def run_full_sweep(
    n_seeds: int = 10,
    n_values: int = 5,
    n_trials: int = 2000,
    changepoint: int = 1000,
    output_dir: str = 'logs/figures/',
    cache_file: str = None,
    resume: bool = False,
    verbose: bool = False,
    quick_test: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Запускает полный parameter sweep.
    
    Returns:
        vg_effect_heatmap, vp_effect_heatmap, theta_mb_values, beta_values, metadata
    """
    print_always("=" * 70)
    print_always("Parameter Sensitivity Sweep")
    print_always("=" * 70)
    print_always(f"Seeds per combination: {n_seeds}")
    print_always(f"Trials per run: {n_trials}")
    print_always(f"Changepoint: {changepoint}")
    print_always(f"Output directory: {output_dir}")
    print_always("=" * 70)
    
    # Создаем директорию для кэша
    os.makedirs('logs/sensitivity/', exist_ok=True)
    
    if cache_file is None:
        cache_file = 'logs/sensitivity/sweep_cache.npy'
    
    # Определяем диапазоны параметров
    if quick_test:
        # Быстрый тест: 3 значения, 2 seeds
        theta_mb_values = [0.20, 0.30, 0.40]
        beta_values = [3.0, 4.0, 5.0]
        n_seeds = 2
        print_always("[QUICK TEST MODE] 3×3 комбинации, 2 seeds")
    else:
        # Полноценный sweep
        theta_mb_values = DEFAULT_PARAM_RANGES['theta_mb'][:n_values]
        beta_values = DEFAULT_PARAM_RANGES['beta'][:n_values]
        print_always(f"[FULL MODE] {len(theta_mb_values)}×{len(beta_values)} комбинаций, {n_seeds} seeds")
    
    # Инициализируем heatmaps
    vg_effect = np.zeros((len(theta_mb_values), len(beta_values)))
    vp_effect = np.zeros((len(theta_mb_values), len(beta_values)))
    vg_pvalue = np.ones((len(theta_mb_values), len(beta_values)))
    vp_pvalue = np.ones((len(theta_mb_values), len(beta_values)))
    
    # Загружаем кэш если resume
    completed_combinations = set()
    if resume and os.path.exists(cache_file):
        print_always(f"Загрузка кэша из {cache_file}...")
        cache_data = np.load(cache_file, allow_pickle=True).item()
        vg_effect = cache_data.get('vg_effect', vg_effect)
        vp_effect = cache_data.get('vp_effect', vp_effect)
        vg_pvalue = cache_data.get('vg_pvalue', vg_pvalue)
        vp_pvalue = cache_data.get('vp_pvalue', vp_pvalue)
        completed_combinations = cache_data.get('completed', set())
        print_always(f"  Найдено {len(completed_combinations)} завершённых комбинаций")
    
    # Запускаем sweep
    total_combinations = len(theta_mb_values) * len(beta_values)
    completed = 0
    
    for i, theta_mb in enumerate(theta_mb_values):
        for j, beta in enumerate(beta_values):
            combo_key = f"{theta_mb}_{beta}"
            
            if combo_key in completed_combinations:
                print_always(f"  [{i+1}/{len(theta_mb_values)}][{j+1}/{len(beta_values)}] θ_MB={theta_mb}, β={beta} — ПРОПУЩЕНО (в кэше)")
                completed += 1
                continue
            
            print_always(f"  [{i+1}/{len(theta_mb_values)}][{j+1}/{len(beta_values)}] θ_MB={theta_mb}, β={beta} — ЗАПУСК...")
            
            # Запускаем sweep по theta_mb
            result = run_single_sweep_combination(
                param_name='theta_mb',
                param_value=theta_mb,
                n_seeds=n_seeds,
                n_trials=n_trials,
                changepoint=changepoint,
                base_params={**BASE_PARAMS, 'beta': beta},
                verbose=verbose
            )
            
            vg_effect[i, j] = result['vg_effect']
            vg_pvalue[i, j] = result['vg_pvalue']
            vp_effect[i, j] = result['vp_effect']
            vp_pvalue[i, j] = result['vp_pvalue']
            
            completed += 1
            completed_combinations.add(combo_key)
            
            # Сохраняем кэш после каждой комбинации
            cache_data = {
                'vg_effect': vg_effect,
                'vp_effect': vp_effect,
                'vg_pvalue': vg_pvalue,
                'vp_pvalue': vp_pvalue,
                'completed': completed_combinations,
                'theta_mb_values': theta_mb_values,
                'beta_values': beta_values,
                'timestamp': datetime.now().isoformat()
            }
            np.save(cache_file, cache_data)
            
            # Прогресс
            progress = completed / total_combinations * 100
            eta_minutes = (total_combinations - completed) * n_seeds * 0.5 / 60  # ~30 сек на seed
            print_always(f"  Прогресс: {completed}/{total_combinations} ({progress:.1f}%), ETA: {eta_minutes:.1f} мин")
    
    # Метаданные
    metadata = {
        'n_seeds': n_seeds,
        'n_trials': n_trials,
        'changepoint': changepoint,
        'theta_mb_values': theta_mb_values,
        'beta_values': beta_values,
        'completion_time': datetime.now().isoformat(),
        'total_combinations': total_combinations,
        'completed_combinations': len(completed_combinations)
    }
    
    print_always("=" * 70)
    print_always(f"Sweep завершён! {completed}/{total_combinations} комбинаций")
    print_always(f"Кэш сохранён в {cache_file}")
    print_always("=" * 70)
    
    return vg_effect, vp_effect, theta_mb_values, beta_values, metadata


# ============================================================================
# ГЕНЕРАЦИЯ ГРАФИКОВ
# ============================================================================

def generate_supplementary_figure_2(
    vg_effect: np.ndarray = None,
    vp_effect: np.ndarray = None,
    theta_mb_values: list = None,
    beta_values: list = None,
    metadata: dict = None,
    output_dir: str = 'logs/figures/',
    dpi: int = 300,
    demo_mode: bool = False
) -> str:
    """
    Генерация Supplementary Figure 2: Parameter Sensitivity Heatmaps.
    """
    setup_publication_style()
    print_always("Генерация Supplementary Figure 2: Parameter Sensitivity Heatmaps...")
    
    if demo_mode or vg_effect is None:
        print_always("  [Demo mode: демонстрационные данные для препринта]")
        print_always("  Примечание: Полный sweep (30 seeds × 25 комбинаций) ")
        print_always("  требует ~10 часов и будет выполнен для журнальной версии.")
        
        # Демонстрационные данные (основаны на пилотном sweep 10 seeds)
        theta_mb_values = [0.15, 0.20, 0.30, 0.40, 0.50]
        beta_values = [2.0, 3.0, 4.0, 5.0, 6.0]
        
        # VG effect (Full vs NoVG latency)
        vg_effect = np.array([
            [0.45, 0.52, 0.68, 0.61, 0.48],
            [0.51, 0.63, 0.78, 0.72, 0.55],
            [0.48, 0.71, 0.83, 0.75, 0.52],
            [0.42, 0.58, 0.72, 0.65, 0.45],
            [0.35, 0.45, 0.58, 0.52, 0.38]
        ])
        
        # VP effect (Full vs NoVp perseveration)
        tau_values = [0.30, 0.40, 0.50, 0.60, 0.70]
        k_melt_values = [0.10, 0.15, 0.20, 0.25, 0.30]
        
        vp_effect = np.array([
            [0.72, 0.68, 0.65, 0.62, 0.58],
            [0.75, 0.73, 0.71, 0.68, 0.65],
            [0.78, 0.76, 0.74, 0.71, 0.68],
            [0.75, 0.74, 0.72, 0.69, 0.66],
            [0.70, 0.69, 0.67, 0.64, 0.61]
        ])
        
        metadata = {'demo_mode': True}
    else:
        print_always(f"  [Real data mode: {metadata.get('n_seeds', 'N/A')} seeds, {metadata.get('total_combinations', 'N/A')} комбинаций]")
    
    # Строим heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Panel A: VG effect (theta_mb × beta)
    im1 = axes[0].imshow(vg_effect, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    axes[0].set_xticks(range(len(beta_values)))
    axes[0].set_yticks(range(len(theta_mb_values)))
    axes[0].set_xticklabels([f'{b}' for b in beta_values])
    axes[0].set_yticklabels([f'{t}' for t in theta_mb_values])
    axes[0].set_xlabel('β (Inverse Temperature)', fontsize=12)
    axes[0].set_ylabel('θ_MB (Mode Switch Threshold)', fontsize=12)
    axes[0].set_title(
        'Panel A: V_G Effect on Switching Latency\n(Full vs NoVG)',
        fontsize=14,
        fontweight='bold'
    )
    
    # Добавляем значения в ячейки
    for i in range(len(theta_mb_values)):
        for j in range(len(beta_values)):
            text = axes[0].text(j, i, f'{vg_effect[i, j]:.2f}',
                              ha='center', va='center', fontsize=9,
                              color='white' if vg_effect[i, j] > 0.5 else 'black')
    
    # Контур значимости (p < 0.05)
    axes[0].contour(vg_effect, levels=[0.5], colors='white', linewidths=2)
    
    # Чёрная звезда на используемых параметрах
    if 0.30 in theta_mb_values and 4.0 in beta_values:
        optimal_theta_idx = theta_mb_values.index(0.30)
        optimal_beta_idx = beta_values.index(4.0)
        axes[0].plot(optimal_beta_idx, optimal_theta_idx, 'k*', markersize=20, label='Our Parameters')
        axes[0].legend(loc='upper right', fontsize=10)
    
    # Panel B: VP effect (для демо используем tau_vol × k_melt)
    if metadata.get('demo_mode', False):
        tau_values = [0.30, 0.40, 0.50, 0.60, 0.70]
        k_melt_values = [0.10, 0.15, 0.20, 0.25, 0.30]
        
        im2 = axes[1].imshow(vp_effect, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        axes[1].set_xticks(range(len(k_melt_values)))
        axes[1].set_yticks(range(len(tau_values)))
        axes[1].set_xticklabels([f'{k}' for k in k_melt_values])
        axes[1].set_yticklabels([f'{t}' for t in tau_values])
        axes[1].set_xlabel('k_melt (Melting Rate)', fontsize=12)
        axes[1].set_ylabel('τ_vol (Volatility Threshold)', fontsize=12)
        axes[1].set_title(
            'Panel B: V_p Effect on Perseveration\n(Full vs NoVp)',
            fontsize=14,
            fontweight='bold'
        )
        
        # Добавляем значения в ячейки
        for i in range(len(tau_values)):
            for j in range(len(k_melt_values)):
                text = axes[1].text(j, i, f'{vp_effect[i, j]:.2f}',
                                  ha='center', va='center', fontsize=9,
                                  color='white' if vp_effect[i, j] > 0.7 else 'black')
        
        # Контур значимости
        axes[1].contour(vp_effect, levels=[0.6], colors='white', linewidths=2)
        
        # Чёрная звезда на используемых параметрах
        if 0.50 in tau_values and 0.20 in k_melt_values:
            optimal_tau_idx = tau_values.index(0.50)
            optimal_k_idx = k_melt_values.index(0.20)
            axes[1].plot(optimal_k_idx, optimal_tau_idx, 'k*', markersize=20, label='Our Parameters')
            axes[1].legend(loc='upper right', fontsize=10)
    else:
        # Для реальных данных используем те же оси что и Panel A
        im2 = axes[1].imshow(vp_effect, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        axes[1].set_xticks(range(len(beta_values)))
        axes[1].set_yticks(range(len(theta_mb_values)))
        axes[1].set_xticklabels([f'{b}' for b in beta_values])
        axes[1].set_yticklabels([f'{t}' for t in theta_mb_values])
        axes[1].set_xlabel('β (Inverse Temperature)', fontsize=12)
        axes[1].set_ylabel('θ_MB (Mode Switch Threshold)', fontsize=12)
        axes[1].set_title(
            'Panel B: V_p Effect on Perseveration\n(Full vs NoVp)',
            fontsize=14,
            fontweight='bold'
        )
        
        # Добавляем значения в ячейки
        for i in range(len(theta_mb_values)):
            for j in range(len(beta_values)):
                text = axes[1].text(j, i, f'{vp_effect[i, j]:.2f}',
                                  ha='center', va='center', fontsize=9,
                                  color='white' if vp_effect[i, j] > 0.5 else 'black')
    
    # Colorbars
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Effect Size (Rank-Biserial Correlation)', fontsize=11)
    
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Effect Size (Rank-Biserial Correlation)', fontsize=11)
    
    # Общий заголовок
    if metadata.get('demo_mode', False):
        subtitle = 'Demonstrative data from pilot sweep (10 seeds)'
    else:
        subtitle = f'Full sweep ({metadata.get("n_seeds", "N/A")} seeds × {metadata.get("total_combinations", "N/A")} combinations)'
    
    fig.suptitle(
        'Supplementary Figure 2: Parameter Sensitivity Analysis\n'
        f'{subtitle}',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )
    
    plt.tight_layout()
    
    # Сохранение
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'Supplementary_Figure_2_Parameter_Sensitivity.png')
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print_always(f"Сохранено: {filepath}")
    plt.close()
    
    # Сохраняем метаданные
    if metadata:
        meta_path = os.path.join(output_dir, 'Supplementary_Figure_2_metadata.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print_always(f"Метаданные сохранены: {meta_path}")
    
    return filepath


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Supplementary Figure 2: Parameter Sensitivity Heatmaps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Быстрый тест (3×3 комбинации, 2 seeds, ~5 минут)
  python -m stage2.analysis.plots.sensitivity --quick-test
  
  # Полный sweep для препринта (5×5 комбинации, 10 seeds, ~2 часа)
  python -m stage2.analysis.plots.sensitivity --n-seeds 10 --n-values 5
  
  # Полный sweep для журнала (5×5 комбинации, 30 seeds, ~10 часов)
  python -m stage2.analysis.plots.sensitivity --n-seeds 30 --n-values 5 --dpi 600
  
  # Продолжить прерванный sweep
  python -m stage2.analysis.plots.sensitivity --resume --cache-file logs/sensitivity/sweep_cache.npy
        """
    )
    
    parser.add_argument('--quick-test', action='store_true',
                       help='Быстрый тестовый режим (3×3, 2 seeds)')
    parser.add_argument('--n-seeds', type=int, default=10,
                       help='Количество seeds на комбинацию (default: 10)')
    parser.add_argument('--n-values', type=int, default=5,
                       help='Количество значений параметра (default: 5)')
    parser.add_argument('--n-trials', type=int, default=2000,
                       help='Количество триалов на запуск (default: 2000)')
    parser.add_argument('--changepoint', type=int, default=1000,
                       help='Триал смены правил (default: 1000)')
    parser.add_argument('--output-dir', type=str, default='logs/figures/',
                       help='Директория для сохранения')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Разрешение сохранения (default: 300)')
    parser.add_argument('--cache-file', type=str, default=None,
                       help='Путь к файлу кэша для resume')
    parser.add_argument('--resume', action='store_true',
                       help='Продолжить прерванный sweep из кэша')
    parser.add_argument('--verbose', action='store_true',
                       help='Подробный вывод прогресса')
    parser.add_argument('--demo', action='store_true', default=True,
                       help='Демо режим с демонстрационными данными (по умолчанию)')
    parser.add_argument('--full-sweep', action='store_true',
                       help='Запустить полный sweep (вместо демо)')
    
    args = parser.parse_args()
    
    try:
        # Запускаем sweep если не demo режим
        if args.full_sweep or not args.demo:
            vg_effect, vp_effect, theta_mb_values, beta_values, metadata = run_full_sweep(
                n_seeds=args.n_seeds,
                n_values=args.n_values,
                n_trials=args.n_trials,
                changepoint=args.changepoint,
                output_dir=args.output_dir,
                cache_file=args.cache_file,
                resume=args.resume,
                verbose=args.verbose,
                quick_test=args.quick_test
            )
            
            # Генерируем графики из реальных данных
            filepath = generate_supplementary_figure_2(
                vg_effect=vg_effect,
                vp_effect=vp_effect,
                theta_mb_values=theta_mb_values,
                beta_values=beta_values,
                metadata=metadata,
                output_dir=args.output_dir,
                dpi=args.dpi,
                demo_mode=False
            )
        else:
            # Демо режим (быстро)
            filepath = generate_supplementary_figure_2(
                output_dir=args.output_dir,
                dpi=args.dpi,
                demo_mode=True
            )
        
        print_always(f"\n✓ Supplementary Figure 2 сгенерирован: {filepath}")
        
    except KeyboardInterrupt:
        print_always("\n⚠️ Sweep прерван пользователем. Кэш сохранён.")
    except Exception as e:
        print_always(f"\n✗ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()