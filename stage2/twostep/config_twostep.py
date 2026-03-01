"""
Конфигурация Two-Step Task (Daw et al., 2011).
ИСПРАВЛЕННАЯ ВЕРСИЯ: per-action rewards на этапе 2 (2-armed bandit).
"""

from typing import Dict, List

# ========== Параметры среды ==========
N_TRIALS: int = 2000          # Увеличено для стабильности статистики
SEED_ENV: int = 42
SEED_AGENT: int = 42

# ========== Структура задачи ==========
N_STAGE1_STATES: int = 1      # Одно начальное состояние (канонично)
N_STAGE2_STATES: int = 2      # Два состояния этапа 2
N_STAGE2_ACTIONS: int = 2     # Два действия на этапе 2 (2-armed bandit!)
N_ACTIONS: int = 2            # Действия на этапе 1 (Left / Right)

# ========== Переходы ==========
COMMON_TRANS_PROB: float = 0.70
RARE_TRANS_PROB: float = 0.30

# КАНОНИЧЕСКАЯ СТРУКТУРА (Daw et al., 2011)
# Action 0: State 0 (70% common), State 1 (30% rare)
# Action 1: State 1 (70% common), State 0 (30% rare)
STAGE1_TRANSITIONS: Dict[int, Dict[int, tuple]] = {
    0: {0: (0, 1), 1: (1, 0)}
}

# ========== Награды ==========
REWARD_MIN: float = 0.0
REWARD_MAX: float = 1.0
REWARD_DRIFT_RATE: float = 0.025
REWARD_DRIFT_SD: float = 0.01

# НАГРАДЫ ДЛЯ 2-ARMED BANDIT: [s2=0: [a2=0, a2=1], s2=1: [a2=0, a2=1]]
INITIAL_REWARD_PROBS: List[List[float]] = [
    [0.75, 0.25],  # s2=0: left lever 75%, right lever 25%
    [0.25, 0.75]   # s2=1: left lever 25%, right lever 75%
]

# ========== Параметры для отладки ==========
WITH_CHANGEPOINT: bool = True
CHANGEPOINT_TRIAL: int = 1000
LAPSE_RATE: float = 0.05

# ========== Логирование ==========
LOG_DIR: str = 'logs/twostep/'
LOG_LEVEL: str = 'trial'

# ========== Debug-скрипты ==========
DEBUG_INTERFACE_SEED: int = 42
DEBUG_ENV_SEED: int = 42
DEBUG_SANITY_SEED: int = 42
DEBUG_INTEGRATION_SEED: int = 42
DEBUG_ABLATION_SEED: int = 42