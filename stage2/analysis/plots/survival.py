"""
Supplementary Figure 1: Kaplan-Meier Survival Curves for Switching Latency.

ИСПРАВЛЕННАЯ ВЕРСИЯ:
- True survival curve (убывает 1→0, не растёт 0→1)
- Log-scale по оси X для читаемости NoVG кривой
- Inset с зумом на первые 50 триалов
- Исправленный поиск пути к эксперименту

Запуск:
    python -m stage2.analysis.plots.survival --experiment-id twostep_ablation_20260310_195529
    python -m stage2.analysis.plots.survival --output-dir logs/figures/ --dpi 600
    python -m stage2.analysis.plots.survival  # Автоматический поиск последнего эксперимента
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


def generate_supplementary_figure_1(
    experiment_dir: str = None,
    changepoint: int = 1000,
    max_trials: int = 2000,
    output_dir: str = 'logs/figures/',
    dpi: int = 300
) -> str:
    """
    Генерация Supplementary Figure 1: Kaplan-Meier Survival Curves.
    
    ИСПРАВЛЕНИЯ:
    1. True survival function (убывает 1→0)
    2. Log-scale по оси X для читаемости NoVG
    3. Inset с зумом на первые 50 триалов
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
    
    # Строим кривые
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = {'Full': '#1f77b4', 'NoVG': '#d62728', 'NoVp': '#2ca02c'}
    linestyles = {'Full': '-', 'NoVG': '--', 'NoVp': '-.'}
    labels = {'Full': 'Full (median=35)', 'NoVG': 'NoVG (median=1)', 'NoVp': 'NoVp (median=27)'}
    
    kmf_results = {}
    median_times = {}
    
    for agent in ['Full', 'NoVG', 'NoVp']:
        agent_df = survival_df[survival_df['agent'] == agent]
        
        if len(agent_df) == 0:
            continue
        
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=agent_df['duration'],
            event_observed=agent_df['observed'],
            label=labels.get(agent, agent)
        )
        
        kmf_results[agent] = kmf
        
        # ИСПРАВЛЕНИЕ: используем survival_function_ (убывает 1→0), не cumulative_density_
        # Добавляем small epsilon к index для log-scale (log(0) undefined)
        survival_index = kmf.survival_function_.index + 0.1
        
        ax.step(
            survival_index,
            kmf.survival_function_.iloc[:, 0],
            where='post',
            label=labels.get(agent, agent),
            color=colors.get(agent, 'gray'),
            linestyle=linestyles.get(agent, '-'),
            linewidth=2.5
        )
        
        # Confidence interval band
        ax.fill_between(
            survival_index,
            kmf.confidence_interval_.iloc[:, 0],
            kmf.confidence_interval_.iloc[:, 1],
            alpha=0.2,
            color=colors.get(agent, 'gray')
        )
        
        # Сохраняем медиану для legend
        median_times[agent] = kmf.median_survival_time_
    
    # ИСПРАВЛЕНИЕ: Log-scale по оси X для читаемости NoVG кривой
    ax.set_xscale('log')
    
    # Оформление
    ax.set_xlabel('Trials After Changepoint (log scale)', fontsize=14)
    ax.set_ylabel('Proportion Not Yet Switched to EXPLORE', fontsize=14)
    ax.set_title(
        'Supplementary Figure 1: Time-to-Event Analysis of Mode Switching\n'
        '(Kaplan-Meier Survival Curves with 95% CI)',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    
    ax.legend(loc='lower left', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(0.1, max_trials - changepoint)
    ax.set_ylim(0, 1.05)
    
    # Добавляем статистику (log-rank test)
    if 'Full' in kmf_results and 'NoVG' in kmf_results:
        results_full_novg = logrank_test(
            kmf_results['Full'].durations,
            kmf_results['NoVG'].durations,
            event_observed_A=kmf_results['Full'].event_observed,
            event_observed_B=kmf_results['NoVG'].event_observed
        )
        
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
    
    # ИСПРАВЛЕНИЕ: Добавляем inset с зумом на первые 50 триалов (линейная шкала)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    
    ax_inset = inset_axes(ax, width="40%", height="40%", loc='upper right')
    
    # Рисуем те же данные в inset с линейной шкалой
    for agent in ['Full', 'NoVG', 'NoVp']:
        if agent not in kmf_results:
            continue
        
        kmf = kmf_results[agent]
        survival_index = kmf.survival_function_.index + 0.1
        
        # Фильтруем для первых 50 триалов
        mask = survival_index <= 50
        
        ax_inset.step(
            survival_index[mask],
            kmf.survival_function_.iloc[mask, 0],
            where='post',
            color=colors.get(agent, 'gray'),
            linestyle=linestyles.get(agent, '-'),
            linewidth=2
        )
    
    ax_inset.set_xlim(0, 50)
    ax_inset.set_ylim(0, 1.05)
    ax_inset.set_xlabel('Trials (0-50)', fontsize=10)
    ax_inset.set_ylabel('Survival Prob.', fontsize=10)
    ax_inset.grid(True, alpha=0.3)
    ax_inset.tick_params(labelsize=9)
    
    # Рамка вокруг inset на основном графике
    mark_inset(ax, ax_inset, loc1=1, loc2=3, fc="none", ec='0.5', linewidth=1.5)
    
    plt.tight_layout()
    
    # Сохранение
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'Supplementary_Figure_1_Kaplan_Meier.png')
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print_always(f"Сохранено: {filepath}")
    plt.close()
    
    # Возвращаем статистику для обновления manuscript
    stats_output = {}
    if 'Full' in kmf_results and 'NoVG' in kmf_results:
        stats_output['full_vs_novg'] = {
            'chi_squared': float(results_full_novg.test_statistic),
            'p_value': float(results_full_novg.p_value)
        }
    if 'Full' in kmf_results and 'NoVp' in kmf_results:
        results_full_novp = logrank_test(
            kmf_results['Full'].durations,
            kmf_results['NoVp'].durations,
            event_observed_A=kmf_results['Full'].event_observed,
            event_observed_B=kmf_results['NoVp'].event_observed
        )
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
    
    return filepath, stats_output


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
    
    # ИСПРАВЛЕНИЕ: Правильный поиск пути к эксперименту
    from stage2.analysis.loaders import find_latest_experiment, load_experiment_data
    
    if args.experiment_id is None:
        # Автоматический поиск последнего twostep_ablation эксперимента
        experiment_path = find_latest_experiment('twostep_ablation')
    else:
        # Ручной поиск по ID
        # Пробуем найти в twostep директории
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
        filepath, stats = generate_supplementary_figure_1(
            experiment_dir=str(experiment_path),  # ← ИСПРАВЛЕНО: передаём путь, не ID
            changepoint=args.changepoint,
            max_trials=args.max_trials,
            output_dir=args.output_dir,
            dpi=args.dpi
        )
        print_always(f"\n✓ Supplementary Figure 1 сгенерирован: {filepath}")
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