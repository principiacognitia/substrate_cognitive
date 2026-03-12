"""
Статистические утилиты для анализа поведенческих данных.
"""

import numpy as np
from typing import Union, Dict, Any


def mann_whitney_effect_size(u_stat: float, n1: int, n2: int) -> float:
    """
    Вычисляет effect size (rank-biserial correlation) для Mann-Whitney U теста.
    
    Args:
        u_stat: U-статистика
        n1: Размер первой выборки
        n2: Размер второй выборки
        
    Returns:
        Effect size от -1 до 1
    """
    return 1 - (2 * u_stat) / (n1 * n2)


def convert_numpy_types(obj: Any) -> Union[Dict, list, int, float, str]:
    """
    Рекурсивно конвертирует numpy типы в Python native types для JSON сериализации.
    
    Args:
        obj: Объект для конвертации
        
    Returns:
        Конвертированный объект
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.bool_, np.integer)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj