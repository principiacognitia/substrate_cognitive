"""
Supplementary Figure 1: Kaplan-Meier Survival Curves for Switching Latency.

ВЕРСИЯ С ДВУМЯ ПАНЕЛЯМИ:
- Panel A: Full range (log-scale по оси X)
- Panel B: Zoomed view (0-100 триалов, linear scale)

Запуск:
    python -m stage2.analysis.plots.survival --experiment-id twostep_ablation_20260310_195529
    python -m stage2.analysis.plots.survival --output-dir logs/figures/ --dpi 600
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from pathlib import Path
from stage2.analysis.loaders import find_latest_experiment, load_experiment_data
from substrate_analysis.style import setup_publication_style
from stage2.core.args import print_always


def prepare_survival_data(data: dict, changepoint: int = 1000, max_trials: int = 2000):
    """
    Подготавливает данные для survival analysis.
    
    Returns:
        DataFrame с колонками: agent, duration, observed
    """
    records = []
    
    for agent_name, df in data.items():
        if agent_name not in ['Full', 'RheologicalAgent', 'NoVG', 'NoVp']:
            continue
        
        # Группируем по seed
        if 'seed' not in df.columns:
            continue
            
        for seed in df['seed'].unique():
            seed_df = df[df['seed'] == seed].sort_values('trial')
            
            # Находим первый триал EXPLORE после changepoint
            post_change = seed_df[(seed_df['trial'] > changepoint) & (seed_df['mode'] == 'EXPLORE')]
            
            if len(post_change) > 0:
                # Событие произошло
                duration = post_change['trial'].iloc[0] - changepoint
                observed = 1
            else:
                # Цензурировано (не переключился до конца)
                duration = max_trials - changepoint
                observed = 0
            
            # Нормализуем имя агента
            if agent_name in ['Full', 'RheologicalAgent']:
                agent_label = 'Full'
            elif agent_name in ['NoVG', 'RheologicalAgent_NoVG']:
                agent_label = 'NoVG'
            elif agent_name in ['NoVp', 'RheologicalAgent_NoVp']:
                agent_label = 'NoVp'
            else:
                continue
            
            records.append({
                'agent': agent_label,
                'duration': duration,
                'observed': observed,
                'seed': seed
            })
    
    return pd.DataFrame(records)


def fit_kaplan_meier(survival_df: pd.DataFrame, agents: list = ['Full', 'NoVG', 'NoVp']):
    """
    Fits Kaplan-Meier curves for all agents.
    
    Returns:
        kmf_results: dict of KaplanMeierFitter objects
        median_times: dict of median survival times
    """
    kmf_results = {}
    median_times = {}
    
    for agent in agents:
        agent_df = survival_df[survival_df['agent'] == agent]
        
        if len(agent_df) == 0:
            continue
        
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=agent_df['duration'],
            event_observed=agent_df['observed'],
            label=agent
        )
        
        kmf_results[agent] = kmf
        median_times[agent] = kmf.median_survival_time_
    
    return kmf_results, median_times


def plot_survival_curves(ax, kmf_results, colors, linestyles, labels, 
                         x_max=1000, y_max=1.05, log_scale=False, show_stats=False,
                         results_full_novg=None, results_full_novp=None):
    """
    Plots survival curves on given axes.
    """
    for agent in ['Full', 'NoVG', 'NoVp']:
        if agent not in kmf_results:
            continue
        
        kmf = kmf_results[agent]
        
        # Добавляем small epsilon к index для log-scale
        if log_scale:
            survival_index = kmf.survival_function_.index + 0.1
            # Фильтруем по максимальному значению
            mask = survival_index <= x_max
        else:
            survival_index = kmf.survival_function_.index
            mask = survival_index <= x_max
        
        ax.step(
            survival_index[mask],
            kmf.survival_function_.iloc[mask, 0],
            where='post',
            label=labels.get(agent, agent),
            color=colors.get(agent, 'gray'),
            linestyle=linestyles.get(agent, '-'),
            linewidth=2.5
        )
        
        # Confidence interval band
        ax.fill_between(
            survival_index[mask],
            kmf.confidence_interval_.iloc[mask, 0],
            kmf.confidence_interval_.iloc[mask, 1],
            alpha=0.2,
            color=colors.get(agent, 'gray')
        )
    
    # Оформление
    ax.set_xlabel('Trials After Changepoint', fontsize=14)
    ax.set_ylabel('Proportion Not Yet Switched to EXPLORE', fontsize=14)
    ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.3, which='both')
    
    if log_scale:
        ax.set_xscale('log')
        ax.set_xlim(0.1, x_max)
    else:
        ax.set_xlim(0, x_max)
    
    if show_stats and results_full_novg:
        stats_text = (
            f'Log-rank test:\n'
            f'Full vs NoVG: χ²={results_full_novg.test_statistic:.1f}, '
            f'p={results_full_novg.p_value:.2e}'
        )
        ax.text(
            0.02, 0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )


def generate_supplementary_figure_1(
    experiment_dir: str = None,
    changepoint: int = 1000,
    max_trials: int = 2000,
    output_dir: str = 'logs/figures/',
    dpi: int = 300
) -> tuple:
    """
    Генерация Supplementary Figure 1: Kaplan-Meier Survival Curves.
    
    Panel A: Full range (log-scale)
    Panel B: Zoomed view (0-100 trials, linear scale)
    """
    setup_publication_style()
    print_always("Генерация Supplementary Figure 1: Kaplan-Meier Survival Curves...")
    
    # Загружаем данные если не предоставлены
    if experiment_dir is None:
        experiment_dir = find_latest_experiment('twostep_ablation')
    
    if experiment_dir is None:
        raise ValueError("Не указан experiment_dir и не найден twostep_ablation эксперимент")
    
    print_always(f"  Используем эксперимент: {experiment_dir}")
    
    # Загружаем данные
    data = load_experiment_data(experiment_dir)
    
    # Подготавливаем данные
    survival_df = prepare_survival_data(data, changepoint, max_trials)
    
    if len(survival_df) == 0:
        raise ValueError("Нет данных для survival analysis")
    
    # Подгоняем Kaplan-Meier кривые
    kmf_results, median_times = fit_kaplan_meier(survival_df)
    
    # Статистика
    results_full_novg = None
    results_full_novp = None
    
    if 'Full' in kmf_results and 'NoVG' in kmf_results:
        results_full_novg = logrank_test(
            kmf_results['Full'].durations,
            kmf_results['NoVG'].durations,
            event_observed_A=kmf_results['Full'].event_observed,
            event_observed_B=kmf_results['NoVG'].event_observed
        )
    
    if 'Full' in kmf_results and 'NoVp' in kmf_results:
        results_full_novp = logrank_test(
            kmf_results['Full'].durations,
            kmf_results['NoVp'].durations,
            event_observed_A=kmf_results['Full'].event_observed,
            event_observed_B=kmf_results['NoVp'].event_observed
        )
    
    # Настройки
    colors = {'Full': '#1f77b4', 'NoVG': '#d62728', 'NoVp': '#2ca02c'}
    linestyles = {'Full': '-', 'NoVG': '--', 'NoVp': '-.'}
    labels = {'Full': f'Full (median={int(median_times.get("Full", 35))})', 
              'NoVG': f'NoVG (median={int(median_times.get("NoVG", 1))})', 
              'NoVp': f'NoVp (median={int(median_times.get("NoVp", 27))})'}
    
    # =====================================================================
    # PANEL A: Full range (log-scale)
    # =====================================================================
    fig_a, ax_a = plt.subplots(figsize=(12, 8))
    
    plot_survival_curves(
        ax=ax_a,
        kmf_results=kmf_results,
        colors=colors,
        linestyles=linestyles,
        labels=labels,
        x_max=max_trials - changepoint,
        log_scale=True,
        show_stats=True,
        results_full_novg=results_full_novg
    )
    
    ax_a.set_title(
        'Supplementary Figure 1A: Time-to-Event Analysis of Mode Switching\n'
        '(Kaplan-Meier Survival Curves with 95% CI, Full Range)',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    
    ax_a.legend(loc='lower left', fontsize=12)
    
    plt.tight_layout()
    
    # Сохранение Panel A
    os.makedirs(output_dir, exist_ok=True)
    filepath_a = os.path.join(output_dir, 'Supplementary_Figure_1A_Kaplan_Meier.png')
    plt.savefig(filepath_a, dpi=dpi, bbox_inches='tight')
    print_always(f"Сохранено Panel A: {filepath_a}")
    plt.close()
    
    # =====================================================================
    # PANEL B: Zoomed view (0-100 trials, linear scale)
    # =====================================================================
    fig_b, ax_b = plt.subplots(figsize=(12, 8))
    
    plot_survival_curves(
        ax=ax_b,
        kmf_results=kmf_results,
        colors=colors,
        linestyles=linestyles,
        labels=labels,
        x_max=100,
        log_scale=False,
        show_stats=False
    )
    
    ax_b.set_title(
        'Supplementary Figure 1B: Early Switching Dynamics\n'
        '(Zoomed View: Trials 0-100 After Changepoint)',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    
    ax_b.legend(loc='lower left', fontsize=12)
    
    plt.tight_layout()
    
    # Сохранение Panel B
    filepath_b = os.path.join(output_dir, 'Supplementary_Figure_1B_Kaplan_Meier_Zoomed.png')
    plt.savefig(filepath_b, dpi=dpi, bbox_inches='tight')
    print_always(f"Сохранено Panel B: {filepath_b}")
    plt.close()
    
    # Возвращаем статистику для обновления manuscript
    stats_output = {}
    if results_full_novg:
        stats_output['full_vs_novg'] = {
            'chi_squared': float(results_full_novg.test_statistic),
            'p_value': float(results_full_novg.p_value)
        }
    if results_full_novp:
        stats_output['full_vs_novp'] = {
            'chi_squared': float(results_full_novp.test_statistic),
            'p_value': float(results_full_novp.p_value)
        }
    stats_output['medians'] = {k: float(v) if not np.isinf(v) and not np.isnan(v) else None 
                               for k, v in median_times.items()}
    
    # Сохраняем статистику в JSON
    stats_path = os.path.join(output_dir, 'Supplementary_Figure_1_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_output, f, indent=2, ensure_ascii=False)
    print_always(f"Статистика сохранена: {stats_path}")
    
    return filepath_a, filepath_b, stats_output


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate Supplementary Figure 1: Kaplan-Meier Survival Curves"
    )
    parser.add_argument('--experiment-id', type=str, default=None,
                       help='ID эксперимента (по умолчанию: последний twostep_ablation)')
    parser.add_argument('--output-dir', type=str, default='logs/figures/',
                       help='Директория для сохранения')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Разрешение сохранения')
    parser.add_argument('--changepoint', type=int, default=1000,
                       help='Триал смены правил')
    parser.add_argument('--max-trials', type=int, default=2000,
                       help='Максимальное количество триалов')
    
    args = parser.parse_args()
    
    # Правильный поиск пути к эксперименту
    from stage2.analysis.loaders import find_latest_experiment, load_experiment_data
    
    if args.experiment_id is None:
        # Автоматический поиск последнего twostep_ablation эксперимента
        experiment_path = find_latest_experiment('twostep_ablation')
    else:
        # Ручной поиск по ID
        experiment_path = Path('logs/twostep') / args.experiment_id
        if not experiment_path.exists():
            print_always(f"✗ Эксперимент не найден: {experiment_path}")
            return
    
    if experiment_path is None:
        print_always("✗ Эксперимент twostep_ablation не найден")
        print_always("Запустите сначала: python -m stage2.twostep.run_twostep --n-seeds 30")
        return
    
    print_always(f"Используем эксперимент: {experiment_path}")
    
    try:
        filepath_a, filepath_b, stats = generate_supplementary_figure_1(
            experiment_dir=str(experiment_path),
            changepoint=args.changepoint,
            max_trials=args.max_trials,
            output_dir=args.output_dir,
            dpi=args.dpi
        )
        print_always(f"\n✓ Supplementary Figure 1A сгенерирован: {filepath_a}")
        print_always(f"✓ Supplementary Figure 1B сгенерирован: {filepath_b}")
        print_always(f"\nСтатистика для manuscript:")
        print_always(f"  Full vs NoVG: χ²={stats['full_vs_novg']['chi_squared']:.1f}, p={stats['full_vs_novg']['p_value']:.2e}")
        print_always(f"  Full vs NoVp: χ²={stats['full_vs_novp']['chi_squared']:.1f}, p={stats['full_vs_novp']['p_value']:.2e}")
        print_always(f"  Медианы: Full={stats['medians'].get('Full')}, NoVG={stats['medians'].get('NoVG')}, NoVp={stats['medians'].get('NoVp')}")
    except Exception as e:
        print_always(f"\n✗ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()