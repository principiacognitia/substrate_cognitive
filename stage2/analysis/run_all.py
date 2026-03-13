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
from stage2.analysis.plots.reversal import generate_figure_4b
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
                       help='ID конкретного эксперимента (переопределяет --latest)')
    parser.add_argument('--twostep-exp', type=str, default=None,
                       help='ID Two-Step эксперимента (для Figure 2/3)')
    parser.add_argument('--reversal-exp', type=str, default=None,
                       help='ID Reversal эксперимента (для Figure 4)')
    parser.add_argument('--output-dir', type=str, default='logs/figures/',
                       help='Директория для сохранения графиков')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Разрешение сохранения (default: 300)')
    parser.add_argument('--nodebug', action='store_true',
                       help='Отключить отладочный вывод')
    
    args = parser.parse_args()
    
    # Создаём директорию для фигур
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Настраиваем стиль
    setup_publication_style()
    
    print("=" * 70)
    print("Генерация фигур для Stage 2")
    print("=" * 70)
    
    # Определяем какие эксперименты использовать
    twostep_exp = args.twostep_exp or args.experiment_id
    reversal_exp = args.reversal_exp or args.experiment_id
    
    # Если не указано явно, ищем последние эксперименты каждого типа
    if not twostep_exp and args.task in ['all', 'twostep', 'sanity']:
        twostep_exp = find_latest_experiment('twostep')
        if twostep_exp:
            print(f"Найден Two-Step эксперимент: {Path(twostep_exp).name}")
    
    if not reversal_exp and args.task in ['all', 'reversal']:
        reversal_exp = find_latest_experiment('reversal')
        if reversal_exp:
            print(f"Найден Reversal эксперимент: {Path(reversal_exp).name}")
    
    # Генерируем фигуры
    print("\n" + "=" * 70)
    print("Генерация фигур...")
    print("=" * 70)
    
    try:
        # Figure 2: MB/MF Signatures (нужен sanity_check или ablation)
        if args.figure in ['all', '2'] and args.task in ['all', 'twostep', 'sanity']:
            if twostep_exp:
                data = load_experiment_data(twostep_exp)
                meta = load_meta_data(twostep_exp)
                
                # Проверяем есть ли данные для Figure 2
                has_mf_mb = 'MF' in data or 'MFAgent' in data or 'MB' in data or 'MBAgent' in data
                has_full = 'Full' in data or 'RheologicalAgent' in data
                
                if has_mf_mb or has_full:
                    filepath = generate_figure_2(args.output_dir, args.dpi)
                    print(f"✓ Figure 2: {filepath}")
                else:
                    print(f"⚠️ Пропуск Figure 2: нет данных MF/MB/Full в {twostep_exp}")
            else:
                print("⚠️ Пропуск Figure 2: не найден Two-Step эксперимент")
        
        # Figure 3: V_G Dynamics (нужен ablation с Full агентом)
        if args.figure in ['all', '3'] and args.task in ['all', 'twostep']:
            if twostep_exp:
                data = load_experiment_data(twostep_exp)
                meta = load_meta_data(twostep_exp)
                
                if 'Full' in data or 'RheologicalAgent' in data:
                    changepoint = 1000
                    if meta and 'config' in meta:
                        changepoint = int(meta['config'].get('changepoint', 1000))
                    
                    filepath = generate_figure_3(data, changepoint, args.output_dir, args.dpi)
                    print(f"✓ Figure 3: {filepath}")
                else:
                    print(f"⚠️ Пропуск Figure 3: нет данных Full/RheologicalAgent в {twostep_exp}")
            else:
                print("⚠️ Пропуск Figure 3: не найден Two-Step эксперимент")
        
        # Figure 4: Reversal Learning (нужен reversal эксперимент)
        if args.figure in ['all', '4'] and args.task in ['all', 'reversal']:
            if reversal_exp:
                data = load_experiment_data(reversal_exp)
                meta = load_meta_data(reversal_exp)
                
                if 'Full' in data and 'NoVG' in data:
                    reversal_trial = 1000
                    if meta and 'config' in meta:
                        changepoint = int(meta['config'].get('changepoint', 1000))
                        reversal_trial = int(meta['config'].get('reversal_trial', 1000))
                    
                    filepath = generate_figure_4(data, reversal_trial, args.output_dir, args.dpi)
                    print(f"✓ Figure 4: {filepath}")
                    
                    filepath = generate_figure_4b(data, reversal_trial, args.output_dir, args.dpi)
                    print(f"✓ Figure 4B: {filepath}")
                else:
                    print(f"⚠️ Пропуск Figure 4: нет данных Full/NoVG в {reversal_exp}")
            else:
                print("⚠️ Пропуск Figure 4: не найден Reversal эксперимент")
        
        print("=" * 70)
        print(f"✓ Все доступные графики сгенерированы!")
        print(f"Проверьте папку: {Path(args.output_dir).absolute()}")
        
    except Exception as e:
        print(f"✗ Ошибка генерации фигур: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()