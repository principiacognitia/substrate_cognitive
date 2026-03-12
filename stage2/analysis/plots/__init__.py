"""
Модули для генерации публикационных графиков.
"""

from .stay_prob import generate_figure_2
from .vg_dynamics import generate_figure_3
from .reversal import generate_figure_4

__all__ = [
    'generate_figure_2',
    'generate_figure_3',
    'generate_figure_4'
]