"""
Конфигурация среды Reversal Task.
Масштаб выровнен со Stage 2A (Two-Step) для чистоты сравнения.
"""

N_TRIALS_REV = 2000
REVERSAL_TRIAL = 1000

# Стохастичность наград
PROB_HIGH = 0.80
PROB_LOW = 0.20

# Детерминированные переходы (превращают задачу в 2-armed bandit)
# a1=0 -> s2=0; a1=1 -> s2=1
TRANSITIONS_REV = {
    0: {0: (0, 0), 1: (1, 1)}  
}

# Начальные вероятности (a2 игнорируется, но структура сохранена для совместимости)
INITIAL_REWARDS_REV = [
    [PROB_HIGH, PROB_HIGH],  # s2=0 (High reward zone)
    [PROB_LOW, PROB_LOW]     # s2=1 (Low reward zone)
]