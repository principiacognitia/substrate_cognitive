"""
Модуль для парсинга аргументов командной строки.
Единая точка конфигурации для всех экспериментов Stage 2.
"""

import argparse
from typing import Optional, Dict, Any, List


def get_default_parser(description: str = "Stage 2 Experiment") -> argparse.ArgumentParser:
    """
    Создает базовый парсер с общими аргументами для всех скриптов.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python -m stage2.twostep.run_twostep --seed 42 --n-trials 2000
  python -m stage2.twostep.run_twostep --no-log --nodebug
        """
    )
    
    # ===== Общие параметры =====
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed для воспроизводимости (default: 42)'
    )
    
    parser.add_argument(
        '--n-seeds',
        type=int,
        default=30,
        help='Количество seeds в эксперименте (default: 30)'
    )
    
    parser.add_argument(
        '--n-trials',
        type=int,
        default=2000,
        help='Количество триалов в эксперименте (default: 2000)'
    )
    
    parser.add_argument(
        '--changepoint',
        type=int,
        default=None,  # ← Изменено: будет вычисляться как n_trials/2
        help='Триал смены правил (default: n_trials/2)'
    )
    
    # ===== Параметры агента =====
    parser.add_argument(
        '--theta-mb',
        type=float,
        default=0.30,
        help='Порог переключения в MB-режим (default: 0.30)'
    )
    
    parser.add_argument(
        '--theta-u',
        type=float,
        default=1.5,
        help='Базовый порог неопределенности (default: 1.5)'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.35,
        help='Learning rate для Q/R обновлений (default: 0.35)'
    )
    
    parser.add_argument(
        '--beta',
        type=float,
        default=4.0,
        help='Inverse temperature softmax выбора (default: 4.0)'
    )
    
    parser.add_argument(
        '--volatility-threshold',
        type=float,
        default=0.50,
        help='Порог волатильности для стабильности среды (default: 0.50)'
    )
    
    # ===== Режимы отладки =====
    parser.add_argument(
        '--nodebug',
        action='store_true',
        help='Отключить отладочный вывод в консоль'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Включить расширенный вывод (триал за триалом)'
    )
    
    # ===== Логирование =====
    parser.add_argument(
        '--no-log',
        action='store_true',
        help='Отключить логирование триалов в CSV (default: logging enabled)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='logs/twostep/',
        help='Директория для сохранения логов (default: logs/twostep/)'
    )
    
    parser.add_argument(
        '--experiment-id',
        type=str,
        default=None,
        help='Идентификатор эксперимента (default: авто-генерация)'
    )
    
    return parser


def parse_args(args: Optional[list] = None, description: str = "Stage 2 Experiment") -> argparse.Namespace:
    """
    Парсит аргументы командной строки.
    """
    parser = get_default_parser(description)
    parsed = parser.parse_args(args)
    
    # ===== Валидация и вычисление defaults =====
    
    # Если changepoint не задан, вычисляем как n_trials/2
    if parsed.changepoint is None:
        parsed.changepoint = parsed.n_trials // 2
    
    # Проверка что changepoint <= n_trials
    if parsed.changepoint >= parsed.n_trials:
        raise ValueError(
            f"changepoint ({parsed.changepoint}) должен быть меньше n_trials ({parsed.n_trials})"
        )
    
    if parsed.changepoint < 1:
        raise ValueError(
            f"changepoint ({parsed.changepoint}) должен быть >= 1"
        )
    
    return parsed


def print_debug(msg: str, args: argparse.Namespace, verbose: bool = False) -> None:
    """
    Выводит отладочное сообщение, если не установлен флаг --nodebug.
    """
    if not args.nodebug:
        if verbose and not args.verbose:
            return
        print(msg)


def print_always(msg: str = "") -> None:
    """
    Выводит сообщение всегда (независимо от флагов).
    """
    print(msg)