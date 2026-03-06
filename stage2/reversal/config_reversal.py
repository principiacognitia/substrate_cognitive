"""
Конфигурация среды Reversal Task.
Масштаб выровнен со Stage 2A (Two-Step) для чистоты сравнения.
"""

# ========== Параметры среды ==========
N_TRIALS_REV: int = 2000
REVERSAL_TRIAL: int = 1000

# ========== Награды ==========
PROB_HIGH: float = 0.80
PROB_LOW: float = 0.20

# ========== Переходы ==========
# Детерминированные переходы (превращают задачу в 2-armed bandit)
# a1=0 -> s2=0; a1=1 -> s2=1
TRANSITIONS_REV: dict = {
    0: {0: (0, 0), 1: (1, 1)}  
}

# Начальные вероятности (a2 игнорируется, но структура сохранена для совместимости)
INITIAL_REWARDS_REV: list = [
    [PROB_HIGH, PROB_HIGH],  # s2=0 (High reward zone)
    [PROB_LOW, PROB_LOW]     # s2=1 (Low reward zone)
]