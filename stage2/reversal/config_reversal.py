"""
Конфигурация среды Reversal Task.
Используется как детерминированный частный случай Two-Step Task.
"""

N_TRIALS_REV = 300
REVERSAL_TRIAL = 150

PROB_HIGH = 0.80
PROB_LOW = 0.20

# Детерминированные переходы: a1=0 -> s2=0; a1=1 -> s2=1
TRANSITIONS_REV = {
    0: {0: (0, 0), 1: (1, 1)}  # (common, rare) одинаковы
}

# Начальные награды. a2 игнорируется (награда зависит только от s2)
INITIAL_REWARDS_REV = [[PROB_HIGH, PROB_HIGH],  # s2=0 (High)
    [PROB_LOW, PROB_LOW]     # s2=1 (Low)
]