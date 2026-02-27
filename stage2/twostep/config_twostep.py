"""
Конфигурация Two-Step Task (Daw et al., 2011).
Все гиперпараметры вынесены сюда для удобной настройки.
"""

from typing import Dict, List

# ========== Параметры среды ==========
N_TRIALS: int = 300           # Всего триалов в сессии
SEED_ENV: int = 42            # Seed для среды
SEED_AGENT: int = 42          # Seed для агента

# ========== Структура задачи ==========
N_STAGE1_STATES: int = 2      # Состояния первого выбора (A, B)
N_STAGE2_STATES: int = 4      # Состояния второго выбора (1, 2, 3, 4)
N_ACTIONS: int = 2            # Действия на каждом этапе (Left, Right)

# ========== Переходы ==========
COMMON_TRANS_PROB: float = 0.70  # Вероятность common перехода
RARE_TRANS_PROB: float = 0.30    # Вероятность rare перехода

# Маппинг переходов: action → (common_state, rare_state)
# Для state 0 (A): action 0 → state 1 (common), state 3 (rare)
# Для state 0 (A): action 1 → state 2 (common), state 3 (rare)
STAGE1_TRANSITIONS: Dict[int, Dict[int, tuple]] = {
    0: {0: (1, 3), 1: (2, 3)},  # State A
    1: {0: (1, 3), 1: (2, 3)}   # State B (симметрично для простоты)
}

# ========== Награды ==========
REWARD_MIN: float = 0.0
REWARD_MAX: float = 1.0
REWARD_DRIFT_RATE: float = 0.025  # Скорость дрейфа вероятностей наград
REWARD_DRIFT_SD: float = 0.01     # Стандартное отклонение дрейфа

# Начальные вероятности наград для 4 финальных состояний
INITIAL_REWARD_PROBS: List[float] = [0.75, 0.25, 0.25, 0.75]

# ========== Параметры для отладки ==========
WITH_CHANGEPOINT: bool = False   # Добавить явный changepoint (для абляций)
CHANGEPOINT_TRIAL: int = 150     # Триал явной смены правил
LAPSE_RATE: float = 0.05         # Вероятность моторной/перцептивной ошибки

# ========== Логирование ==========
LOG_DIR: str = 'logs/twostep/'
LOG_LEVEL: str = 'trial'  # 'trial' или 'step'

# ========== Debug-скрипты ==========
DEBUG_INTERFACE_SEED: int = 42
DEBUG_ENV_SEED: int = 42
DEBUG_SANITY_SEED: int = 42
DEBUG_INTEGRATION_SEED: int = 42
DEBUG_ABLATION_SEED: int = 42