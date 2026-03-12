"""
Публикационный стиль для matplotlib.
Единая точка настройки визуализации для всех статей.
"""

import matplotlib.pyplot as plt


def setup_publication_style():
    """
    Настраивает стиль matplotlib для научных публикаций.
    Соответствует требованиям PLOS Comp Biol / eLife.
    """
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
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False
    })