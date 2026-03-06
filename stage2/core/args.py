"""
Модуль для парсинга аргументов командной строки.
Единая точка конфигурации для всех экспериментов Stage 2.
"""

import argparse
import sys
from typing import Optional, Dict, Any, List


def get_default_parser(description: str = "Stage 2 Experiment") -> argparse.ArgumentParser:
    """
    Создает базовый парсер с общими аргументами для всех скриптов.
    
    Args:
        description: Описание скрипта для help
        
    Returns:
        Настроенный ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python -m stage2.twostep.debug_ablation --seed 42 --n-trials 2000
  python -m stage2.twostep.debug_ablation --nodebug --theta-mb 0.30
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
        '--n-trials',
        type=int,
        default=2000,
        help='Количество триалов в эксперименте (default: 2000)'
    )
    
    parser.add_argument(
        '--changepoint',
        type=int,
        default=1000,
        help='Триал смены правил (default: 1000)'
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
        help='Температура softmax выбора (default: 4.0)'
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
    
    parser.add_argument(
        '--log-trials',
        action='store_true',
        help='Сохранять логи всех триалов в CSV'
    )
    
    # ===== Вывод =====
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
    
    Args:
        args: Список аргументов (по умолчанию sys.argv[1:])
        description: Описание для парсера
        
    Returns:
        Namespace с распарсенными аргументами
    """
    parser = get_default_parser(description)
    return parser.parse_args(args)


def print_debug(msg: str, args: argparse.Namespace, verbose: bool = False) -> None:
    """
    Выводит отладочное сообщение, если не установлен флаг --nodebug.
    
    Args:
        msg: Сообщение для вывода
        args: Распарсенные аргументы
        verbose: Если True, выводит только в режиме --verbose
    """
    if not args.nodebug:
        if verbose and not args.verbose:
            return
        print(msg)


def print_always(msg: str) -> None:
    """
    Выводит сообщение всегда (независимо от флагов).
    """
    print(msg)