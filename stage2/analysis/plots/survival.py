"""
Supplementary Figure 1: Kaplan-Meier Survival Curves for Switching Latency.

Запуск:
    python -m stage2.analysis.plots.survival --experiment-id twostep_ablation_20260310_195529
    python -m stage2.analysis.plots.survival --output-dir logs/figures/ --dpi 600
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from pathlib import Path

from seaborn import colors
from stage2.analysis.loaders import find_latest_experiment, load_experiment_data, find_experiment_by_id
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
    data: dict = None,
    experiment_id: str = None,
    changepoint: int = 1000,
    max_trials: int = 2000,
    output_dir: str = 'logs/figures/',
    dpi: int = 300
) -> str:
    """
    Генерация Supplementary Figure 1: Kaplan-Meier Survival Curves.
    """
    setup_publication_style()
    print_always("Генерация Supplementary Figure 1: Kaplan-Meier Survival Curves...")
    
    # Загружаем данные если не предоставлены
    if data is None:
        if experiment_id is None:
            experiment_id = find_experiment_by_id('twostep_ablation')
        
        if experiment_id is None:
            raise ValueError("Не указан experiment_id и не найден twostep_ablation эксперимент")
        
        data = load_experiment_data(experiment_id)
    
    # Подготавливаем данные
    survival_df = prepare_survival_data(data, changepoint, max_trials)
    
    if len(survival_df) == 0:
        raise ValueError("Нет данных для survival analysis")
    
    # Строим кривые
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {'Full': '#1f77b4', 'NoVG': '#d62728', 'NoVp': '#2ca02c'}
    linestyles = {'Full': '-', 'NoVG': '--', 'NoVp': '-.'}
    
    kmf_results = {}
    
    for agent in ['Full', 'NoVG', 'NoVp']:
        agent_df = survival_df[survival_df['agent'] == agent]
        
        if len(agent_df) == 0:
            continue
        
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=agent_df['duration'],
            event_observed=agent_df['observed'],
            label=f'{agent} (n={len(agent_df)})'
        )
        
        kmf_results[agent] = kmf
        
        # ИСПРАВЛЕНИЕ: median_survival_time_ уже скаляр в lifelines 0.27+
        median_time = kmf.median_survival_time_
        
        # Обработка случая когда медиана не определена (все цензурировано)
        if np.isinf(median_time) or pd.isna(median_time):
            median_str = 'N/A'
        else:
            median_str = f'{median_time:.0f}'
        
        ax.step(
            kmf.cumulative_density_.index,
            kmf.cumulative_density_.iloc[:, 0],
            where='post',
            label=f'{agent} (median={median_str})',
            color=colors.get(agent, 'gray'),
            linestyle=linestyles.get(agent, '-'),
            linewidth=2.5
        )
        
        # Confidence interval
        ax.fill_between(
            kmf.confidence_interval_.index,
            kmf.confidence_interval_.iloc[:, 0],
            kmf.confidence_interval_.iloc[:, 1],
            alpha=0.2,
            color=colors.get(agent, 'gray')
        )
            
    # Оформление
    ax.set_xlabel('Trials After Changepoint', fontsize=14)
    ax.set_ylabel('Proportion Switched to EXPLORE', fontsize=14)
    ax.set_title(
        'Supplementary Figure 1: Time-to-Event Analysis of Mode Switching\n'
        '(Kaplan-Meier Survival Curves with 95% CI)',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_trials - changepoint)
    ax.set_ylim(0, 1.05)
    
    # Добавляем статистику (log-rank test)
    from lifelines.statistics import logrank_test
    
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
    
    plt.tight_layout()
    
    # Сохранение
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'Supplementary_Figure_1_Kaplan_Meier.png')
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print_always(f"Сохранено: {filepath}")
    plt.close()
    
    return filepath


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
    
    # ИСПРАВЛЕНИЕ: Используем find_experiment_by_pattern вместо find_experiment_by_id
    from stage2.analysis.loaders import find_latest_experiment, load_experiment_data
    
    if args.experiment_id is None:
        # Автоматический поиск последнего twostep_ablation эксперимента
        experiment_path = find_latest_experiment('twostep_ablation')
    else:
        # Ручной поиск по ID
        from pathlib import Path
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
        filepath = generate_supplementary_figure_1(
            experiment_id=str(experiment_path),  # ← ИСПРАВЛЕНО: передаём путь, не ID
            changepoint=args.changepoint,
            max_trials=args.max_trials,
            output_dir=args.output_dir,
            dpi=args.dpi
        )
        print_always(f"\n✓ Supplementary Figure 1 сгенерирован: {filepath}")
    except Exception as e:
        print_always(f"\n✗ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()