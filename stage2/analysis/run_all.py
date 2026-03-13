"""
Генерация всех графиков для Stage 2.

Автоматически находит последние эксперименты каждого типа:
  - Figure 2: Генерируется заново (запускает агентов)
  - Figure 3: Ищет папку twostep_ablation_*
  - Figure 4/4B: Ищет папку reversal_*

Запуск:
    python -m stage2.analysis.run_all
    python -m stage2.analysis.run_all --figure 3 --dpi 600
    python -m stage2.analysis.run_all --figure 4 --output-dir logs/figures/
"""

import os
import argparse
from pathlib import Path
from typing import Optional, Dict

from stage2.analysis.loaders import (
    find_latest_experiment,
    load_experiment_data,
    load_meta_data
)
from stage2.analysis.plots.stay_prob import generate_figure_2
from stage2.analysis.plots.vg_dynamics import generate_figure_3
from stage2.analysis.plots.reversal import generate_figure_4, generate_figure_4b
from substrate_analysis.style import setup_publication_style
from stage2.core.args import print_always


def find_experiment_by_pattern(pattern: str, base_dir: str = 'logs/') -> Optional[str]:
    """
    Ищет последнюю папку, соответствующую паттерну.
    
    Args:
        pattern: Паттерн имени (например, 'twostep_ablation', 'reversal')
        base_dir: Базовая директория логов
        
    Returns:
        Путь к директории или None
    """
    search_dir = Path(base_dir) / 'twostep' if 'twostep' in pattern else Path(base_dir) / 'reversal'
    
    if not search_dir.exists():
        return None
    
    # Ищем все папки с паттерном в имени
    exp_dirs = [d for d in search_dir.iterdir() 
                if d.is_dir() and pattern in d.name]
    
    if not exp_dirs:
        return None
    
    # Сортируем по времени модификации (последняя = самая свежая)
    exp_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return str(exp_dirs[0])


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Generate Publication-Ready Figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python -m stage2.analysis.run_all                    # Все фигуры
  python -m stage2.analysis.run_all --figure 3         # Только Figure 3
  python -m stage2.analysis.run_all --figure 4 --dpi 600
        """
    )
    
    parser.add_argument('--figure', type=str, default='all',
                       choices=['all', '2', '3', '4'],
                       help='Какую фигуру генерировать (default: all)')
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
    
    print_always("=" * 70)
    print_always("Stage 2: Генерация публикационных графиков")
    print_always("=" * 70)
    print_always(f"Output Directory: {args.output_dir}")
    print_always(f"DPI: {args.dpi}")
    print_always("=" * 70)
    print_always()
    
    # Генерируем фигуры
    try:
        # Figure 2: MB/MF Signatures (генерируется заново, не требует логов)
        if args.figure in ['all', '2']:
            print_always("Генерация Figure 2: MB/MF Signatures...")
            filepath = generate_figure_2(args.output_dir, args.dpi)
            print_always(f"✓ Figure 2: {filepath}")
            print_always("")
        
        # Figure 3: V_G Dynamics (требует twostep_ablation_*)
        if args.figure in ['all', '3']:
            print_always("Поиск эксперимента для Figure 3 (twostep_ablation_*)...")
            exp_dir = find_experiment_by_pattern('twostep_ablation')
            
            if exp_dir:
                print_always(f"  Найдено: {Path(exp_dir).name}")
                data = load_experiment_data(exp_dir)
                meta = load_meta_data(exp_dir)
                
                if 'Full' in data or 'RheologicalAgent' in data:
                    changepoint = 1000
                    if meta and 'config' in meta:
                        changepoint = int(meta['config'].get('changepoint', 1000))
                        reversal_trial = int(meta['config'].get('reversal_trial', 1000))
                    
                    filepath = generate_figure_3(data, changepoint, args.output_dir, args.dpi)
                    print_always(f"✓ Figure 3: {filepath}")
                else:
                    print_always(f"⚠️ Пропуск Figure 3: нет данных Full/RheologicalAgent")
            else:
                print_always("⚠️ Пропуск Figure 3: не найдена папка twostep_ablation_*")
            print_always("")
        
        # Figure 4 & 4B: Reversal Learning (требует reversal_*)
        if args.figure in ['all', '4']:
            print_always("Поиск эксперимента для Figure 4 (reversal_*)...")
            exp_dir = find_experiment_by_pattern('reversal')
            
            if exp_dir:
                print_always(f"  Найдено: {Path(exp_dir).name}")
                data = load_experiment_data(exp_dir)
                meta = load_meta_data(exp_dir)
                
                if 'Full' in data and 'NoVG' in data:
                    reversal_trial = 1000
                    if meta and 'config' in meta:
                        reversal_trial = int(meta['config'].get('reversal_trial', 1000))
                        reversal_trial = int(meta['config'].get('reversal_trial', 1000))
                    
                    filepath = generate_figure_4(data, reversal_trial, args.output_dir, args.dpi)
                    print_always(f"✓ Figure 4: {filepath}")
                    
                    filepath = generate_figure_4b(data, reversal_trial, 50, 100, args.output_dir, args.dpi)
                    print_always(f"✓ Figure 4B: {filepath}")
                else:
                    print_always(f"⚠️ Пропуск Figure 4: нет данных Full/NoVG")
            else:
                print_always("⚠️ Пропуск Figure 4: не найдена папка reversal_*")
            print_always("")
        
        print_always("=" * 70)
        print_always("✓ Все доступные графики сгенерированы!")
        print_always(f"Проверьте папку: {Path(args.output_dir).absolute()}")
        print_always("=" * 70)
        
    except Exception as e:
        print_always(f"✗ Ошибка генерации фигур: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()