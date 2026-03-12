"""
Substrate Analysis — общие утилиты для анализа данных.
Используется во всех этапах (Stage 2, Stage 3, etc.).
"""

from .style import setup_publication_style
from .stats import mann_whitney_effect_size, convert_numpy_types

__all__ = [
    'setup_publication_style',
    'mann_whitney_effect_size',
    'convert_numpy_types'
]