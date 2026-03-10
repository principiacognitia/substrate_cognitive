"""
Генерация публикационных графиков (Publication-ready Figures) для статьи.

Читает готовые CSV логи из экспериментов (не запускает агентов!).
Агрегирует данные по всем seeds внутри эксперимента.

Режимы работы:
    --latest              (по умолчанию) Ищет последний эксперимент каждого типа
    --experiment-id ID    Обрабатывает конкретный эксперимент
    --input-dir PATH      Прямая директория с логами

Запуск:
    python -m stage2.analysis.figures --latest
    python -m stage2.analysis.figures --experiment-id twostep_ablation_20260308_201037
    python -m stage2.analysis.figures --input-dir logs/twostep/twostep_ablation_20260308_201037/
    python -m stage2.analysis.figures --latest --output-dir logs/figures/ --dpi 600
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path

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
# УТИЛИТЫ ДЛЯ ПОИСКА ЭКСПЕРИМЕНТОВ
# ============================================================================

def find_latest_experiment(task: str, base_dir: str = 'logs/') -> Optional[str]:
    """
    Ищет последний эксперимент указанного типа по timestamp.
    
    Args:
        task: 'twostep' или 'reversal'
        base_dir: Базовая директория логов
        
    Returns:
        Путь к директории последнего эксперимента или None
    """
    search_dir = Path(base_dir) / task
    
    if not search_dir.exists():
        return None
    
    # Ищем все папки с экспериментами
    exp_dirs = [d for d in search_dir.iterdir() 
                if d.is_dir() and d.name.startswith(f'{task}_')]
    
    if not exp_dirs:
        return None
    
    # Сортируем по имени (timestamp в имени)
    exp_dirs.sort(key=lambda x: x.name, reverse=True)
    
    return str(exp_dirs[0])


def find_experiment_by_id(experiment_id: str, base_dir: str = 'logs/') -> Optional[str]:
    """
    Ищет эксперимент по точному ID.
    
    Args:
        experiment_id: ID эксперимента (например, 'twostep_ablation_20260308_201037')
        base_dir: Базовая директория логов
        
    Returns:
        Путь к директории эксперимента или None
    """
    # Пробуем в twostep
    path = Path(base_dir) / 'twostep' / experiment_id
    if path.exists():
        return str(path)
    
    # Пробуем в reversal
    path = Path(base_dir) / 'reversal' / experiment_id
    if path.exists():
        return str(path)
    
    return None


def load_experiment_data(exp_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Загружает все CSV логи из директории эксперимента.
    
    Args:
        exp_dir: Путь к директории эксперимента
        
    Returns:
        Dict {agent_name: DataFrame с агрегированными данными}
    """
    exp_path = Path(exp_dir)
    
    if not exp_path.exists():
        raise FileNotFoundError(f"Эксперимент не найден: {exp_dir}")
    
    # Ищем все CSV файлы с триалами
    csv_files = list(exp_path.glob('*_trials.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"CSV файлы не найдены в {exp_dir}")
    
    # Группируем по агентам (Full, NoVG, NoVp)
    agent_data = {}
    
    for csv_file in csv_files:
        # Извлекаем имя агента из имени файла
        # Формат: {exp_id}_{agent}_seed{N}_trials.csv
        parts = csv_file.stem.split('_')
        
        # Ищем часть с именем агента
        agent_name = None
        for i, part in enumerate(parts):
            if part in ['Full', 'NoVG', 'NoVp']:
                agent_name = part
                break
        
        if agent_name is None:
            print_always(f"⚠️ Не удалось определить агента для {csv_file.name}")
            continue
        
        # Загружаем CSV
        df = pd.read_csv(csv_file)
        
        if agent_name not in agent_data:
            agent_data[agent_name] = []
        agent_data[agent_name].append(df)
    
    # Агрегируем по seeds
    aggregated = {}
    for agent_name, dfs in agent_data.items():
        aggregated[agent_name] = pd.concat(dfs, ignore_index=True)
        print_always(f"  {agent_name}: {len(dfs)} seeds, {len(aggregated[agent_name])} триалов")
    
    return aggregated


def load_meta_data(exp_dir: str) -> Optional[Dict]:
    """
    Загружает метаданные эксперимента из JSON.
    """
    meta_path = Path(exp_dir) / 'experiment_meta.json'
    
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    return None


# ============================================================================
# FIGURE 2: MB/MF SIGNATURES (Daw et al. 2011 style)
# ============================================================================

def generate_figure_2_signatures(data: Dict[str, pd.DataFrame], 
                                  output_dir: str = 'logs/figures/',
                                  dpi: int = 300) -> str:
    """
    Отрисовка Figure 2: MB/MF Signatures.
    
    Args:
        data: Dict {agent_name: DataFrame}
        output_dir: Директория для сохранения
        dpi: Разрешение
        
    Returns:
        Путь к сохранённому файлу
    """
    print_always("Генерация Figure 2: MB/MF Signatures...")
    
    if 'Full' not in data:
        raise ValueError("Нет данных для агента 'Full'")
    
    df = data['Full']
    
    # Вычисляем stay probabilities по 4 условиям
    conditions = []
    for reward in [1.0, 0.0]:
        for trans in [1, -1]:
            subset = df[(df['reward'] == reward) & (df['trans_factor'] == trans)]
            if len(subset) > 0:
                prob = subset['stay'].mean()
                sem = subset['stay'].sem()  # Standard Error of Mean
                n = len(subset)
            else:
                prob, sem, n = 0, 0, 0
            conditions.append({
                'reward': reward,
                'trans': trans,
                'prob': prob,
                'sem': sem,
                'n': n
            })
    
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
           label='Rewarded', color='#2ca02c', capsize=5, alpha=0.8, error_kw={'elinewidth': 2})
    ax.bar(x + width/2, unrewarded_means, width, yerr=unrewarded_sems,
           label='Unrewarded', color='#d62728', capsize=5, alpha=0.8, error_kw={'elinewidth': 2})
    
    ax.set_ylabel('Stay Probability', fontsize=14)
    ax.set_xlabel('Transition Type', fontsize=14)
    ax.set_title('Figure 2: MB/MF Signatures (Rheological Agent)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower left', fontsize=12)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'Figure_2_Signatures.png')
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print_always(f"Сохранено: {filepath}")
    plt.close()
    
    return filepath


# ============================================================================
# FIGURE 3: V_G DYNAMICS AND HYSTERESIS
# ============================================================================

def generate_figure_3_vg_dynamics(data: Dict[str, pd.DataFrame],
                                   changepoint: int = 1000,
                                   output_dir: str = 'logs/figures/',
                                   dpi: int = 300) -> str:
    """
    Отрисовка Figure 3: Динамика V_G с агрегацией по seeds.
    
    Args:
        data: Dict {agent_name: DataFrame}
        changepoint: Триал смены правил
        output_dir: Директория для сохранения
        dpi: Разрешение
        
    Returns:
        Путь к сохранённому файлу
    """
    print_always("Генерация Figure 3: V_G Dynamics and Hysteresis...")
    
    if 'Full' not in data:
        raise ValueError("Нет данных для агента 'Full'")
    
    df = data['Full']
    
    # Агрегируем по trial и seed
    # Группируем по trial, вычисляем mean и sem для V_G и mode
    agg = df.groupby('trial').agg({
        'V_G': ['mean', 'sem'],
        'mode': ['mean']  # mode уже 0/1, mean = probability
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
    
    filepath = os.path.join(output_dir, 'Figure_3_VG_Dynamics.png')
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print_always(f"Сохранено: {filepath}")
    plt.close()
    
    return filepath


# ============================================================================
# FIGURE 4: REVERSAL LEARNING CURVES (Le et al. 2023 style)
# ============================================================================

def generate_figure_4_reversal(data: Dict[str, pd.DataFrame],
                                reversal_trial: int = 1000,
                                output_dir: str = 'logs/figures/',
                                dpi: int = 300) -> str:
    """
    Отрисовка Figure 4: Reversal Learning Curves.
    
    Args:
        data: Dict {agent_name: DataFrame}
        reversal_trial: Триал реверсала
        output_dir: Директория для сохранения
        dpi: Разрешение
        
    Returns:
        Путь к сохранённому файлу
    """
    print_always("Генерация Figure 4: Reversal Learning Curves...")
    
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
    
    # Агрегируем по trial
    full_agg = full_df.groupby('trial')['correct'].agg(['mean', 'sem']).reset_index()
    novg_agg = novg_df.groupby('trial')['correct'].agg(['mean', 'sem']).reset_index()
    
    # Сглаживание (rolling mean)
    window = 10
    full_agg['smooth'] = full_agg['mean'].rolling(window=window, min_periods=1).mean()
    novg_agg['smooth'] = novg_agg['mean'].rolling(window=window, min_periods=1).mean()
    
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
    
    ax.axvline(x=reversal_trial, color='black', linestyle=':', linewidth=2,
               label='Reversal Point')
    ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, label='Chance Level')
    ax.axhline(y=0.8, color='green', linestyle='-.', alpha=0.5, label='Criterion (80%)')
    
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Trial', fontsize=14)
    ax.set_ylabel('P(Correct Choice)', fontsize=14)
    ax.set_title('Figure 4: Reversal Learning & Perseveration',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'Figure_4_Reversal.png')
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print_always(f"Сохранено: {filepath}")
    plt.close()
    
    return filepath


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Точка входа для генерации всех фигур."""
    parser = parse_args(
        description="Stage 2: Generate Publication-Ready Figures",
        default_output_dir='logs/figures/'
    )
    
    # Дополнительные аргументы для figures
    parser.add_argument('--input-dir', type=str, default=None,
                       help='Прямая директория с логами (переопределяет --latest/--experiment-id)')
    parser.add_argument('--latest', action='store_true', default=True,
                       help='Использовать последний эксперимент (по умолчанию)')
    parser.add_argument('--experiment-id', type=str, default=None,
                       help='ID конкретного эксперимента')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Разрешение сохранения (default: 300)')
    parser.add_argument('--figure', type=str, default='all',
                       choices=['all', '2', '3', '4'],
                       help='Какую фигуру генерировать (default: all)')
    
    args = parser.parse_args()
    
    # Определяем директорию эксперимента
    exp_dir = None
    
    if args.input_dir:
        exp_dir = args.input_dir
        print_always(f"Используем директорию: {exp_dir}")
    elif args.experiment_id:
        exp_dir = find_experiment_by_id(args.experiment_id)
        if exp_dir:
            print_always(f"Найден эксперимент: {args.experiment_id}")
        else:
            print_always(f"✗ Эксперимент не найден: {args.experiment_id}")
            return
    else:  # --latest (по умолчанию)
        # Пробуем twostep
        exp_dir = find_latest_experiment('twostep')
        if exp_dir:
            print_always(f"Последний эксперимент (twostep): {Path(exp_dir).name}")
        else:
            print_always("✗ Эксперименты twostep не найдены")
            return
    
    # Создаём директорию для фигур
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Загружаем данные
    print_always("=" * 70)
    print_always("Загрузка данных эксперимента...")
    print_always("=" * 70)
    
    try:
        data = load_experiment_data(exp_dir)
        meta = load_meta_data(exp_dir)
        
        if meta:
            print_always(f"Эксперимент: {meta.get('experiment_name', 'N/A')}")
            print_always(f"Описание: {meta.get('description', 'N/A')}")
            print_always(f"Конфигурация: {meta.get('config', {})}")
        
        print_always("--" * 35)
    except Exception as e:
        print_always(f"✗ Ошибка загрузки данных: {e}")
        return
    
    # Настраиваем стиль
    setup_publication_style()
    
    # Получаем параметры из метаданных или используем дефолты
    changepoint = 1000
    reversal_trial = 1000
    
    if meta and 'config' in meta:
        changepoint = meta['config'].get('changepoint', 1000)
        reversal_trial = meta['config'].get('reversal_trial', 1000)
    
    # Генерируем фигуры
    print_always("\n" + "=" * 70)
    print_always("Генерация фигур")
    print_always("=" * 70)
    
    try:
        if args.figure in ['all', '2']:
            generate_figure_2_signatures(
                data=data,
                output_dir=args.output_dir,
                dpi=args.dpi
            )
            print_always()
        
        if args.figure in ['all', '3']:
            generate_figure_3_vg_dynamics(
                data=data,
                changepoint=changepoint,
                output_dir=args.output_dir,
                dpi=args.dpi
            )
            print_always()
        
        if args.figure in ['all', '4']:
            generate_figure_4_reversal(
                data=data,
                reversal_trial=reversal_trial,
                output_dir=args.output_dir,
                dpi=args.dpi
            )
            print_always()
        
        print_always("=" * 70)
        print_always("✓ Все графики сгенерированы!")
        print_always(f"Проверьте папку: {os.path.abspath(args.output_dir)}")
        print_always("=" * 70)
        
    except Exception as e:
        print_always(f"✗ Ошибка генерации фигур: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()