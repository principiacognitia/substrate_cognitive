"""
Генерация всех графиков для Stage 2.

Запуск:
    python -m stage2.analysis.run_all --latest
    python -m stage2.analysis.run_all --experiment-id twostep_ablation_20260310_195529
    python -m stage2.analysis.run_all --task twostep --figure 3
"""

import argparse
import sys
from pathlib import Path

from stage2.analysis.loaders import (
    find_latest_experiment,
    find_experiment_by_id,
    load_experiment_data,
    load_meta_data
)
from stage2.analysis.plots import (
    generate_figure_2,
    generate_figure_3,
    generate_figure_4
)
from substrate_analysis.style import setup_publication_style


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Generate Publication-Ready Figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python -m stage2.analysis.run_all --latest
  python -m stage2.analysis.run_all --experiment-id twostep_ablation_20260310_195529
  python -m stage2.analysis.run_all --task reversal --figure 4
        """
    )
    
    parser.add_argument('--task', type=str, default='all',
                       choices=['all', 'twostep', 'reversal', 'sanity'],
                       help='Какую задачу анализировать (default: all)')
    parser.add_argument('--figure', type=str, default='all',
                       choices=['all', '2', '3', '4'],
                       help='Какую фигуру генерировать (default: all)')
    parser.add_argument('--experiment-id', type=str, default=None,
                       help='ID конкретного эксперимента')
    parser.add_argument('--latest', action='store_true', default=True,
                       help='Использовать последний эксперимент (по умолчанию)')
    parser.add_argument('--output-dir', type=str, default='logs/figures/',
                       help='Директория для сохранения графиков')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Разрешение сохранения (default: 300)')
    parser.add_argument('--nodebug', action='store_true',
                       help='Отключить отладочный вывод')
    
    args = parser.parse_args()
    
    # Определяем директорию эксперимента
    exp_dir = None
    
    if args.experiment_id:
        exp_dir = find_experiment_by_id(args.experiment_id)
        if not exp_dir:
            print(f"✗ Эксперимент не найден: {args.experiment_id}")
            return
    else:  # --latest (по умолчанию)
        if args.task in ['all', 'twostep']:
            exp_dir = find_latest_experiment('twostep')
        elif args.task == 'reversal':
            exp_dir = find_latest_experiment('reversal')
        
        if not exp_dir:
            print(f"✗ Эксперименты не найдены")
            return
    
    print(f"Используем эксперимент: {Path(exp_dir).name}")
    
    # Создаём директорию для фигур
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Загружаем данные
    print("=" * 70)
    print("Загрузка данных эксперимента...")
    print("=" * 70)
    
    try:
        data = load_experiment_data(exp_dir)
        meta = load_meta_data(exp_dir)
        
        for agent_name, df in data.items():
            print(f"  {agent_name}: {len(df)} триалов")
        
        if meta:
            print(f"\nОписание: {meta.get('description', 'N/A')}")
        
        print("=" * 70)
    except Exception as e:
        print(f"✗ Ошибка загрузки данных: {e}")
        return
    
    # Настраиваем стиль
    setup_publication_style()
    
    # Получаем параметры из метаданных
    changepoint = 1000
    reversal_trial = 1000
    
    if meta and 'config' in meta:
        changepoint = meta['config'].get('changepoint', 1000)
        reversal_trial = meta['config'].get('reversal_trial', 1000)
    
    # Генерируем фигуры
    print("\nГенерация фигур...")
    print("=" * 70)
    
    try:
        if args.figure in ['all', '2'] and args.task in ['all', 'twostep', 'sanity']:
            filepath = generate_figure_2(data, args.output_dir, args.dpi)
            print(f"✓ Figure 2: {filepath}")
        
        if args.figure in ['all', '3'] and args.task in ['all', 'twostep']:
            filepath = generate_figure_3(data, changepoint, args.output_dir, args.dpi)
            print(f"✓ Figure 3: {filepath}")
        
        if args.figure in ['all', '4'] and args.task in ['all', 'reversal']:
            filepath = generate_figure_4(data, reversal_trial, args.output_dir, args.dpi)
            print(f"✓ Figure 4: {filepath}")
        
        print("=" * 70)
        print(f"✓ Все графики сгенерированы!")
        print(f"Проверьте папку: {Path(args.output_dir).absolute()}")
        
    except Exception as e:
        print(f"✗ Ошибка генерации фигур: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()