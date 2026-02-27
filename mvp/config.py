"""
Конфигурация — ВЕРСИЯ 5.0
ВСЕ магические константы вынесены сюда с осмысленными именами.
"""

from typing import Dict, List

# ========== Параметры среды ==========
N_TRIALS: int = 150
CHANGEPOINT_TRIAL: int = 60
REWARD_CORRECT: float = 1.0
REWARD_WRONG: float = 0.0

# ========== Параметры агента: Q-learning ==========
ALPHA_MF: float = 0.12          # СНИЖЕНО с 0.20 (медленнее в EXPLOIT)
ALPHA_MB: float = 0.50          # Быстрое обновление в EXPLORE
GAMMA: float = 0.9
BETA_ACTION: float = 5.0
BETA_STICK: float = 4.0

# ========== Параметры агента: Belief state ==========
B_PRIOR: float = 0.5            # Равный приор над H_L / H_R
BELIEF_LR: float = 0.12         # СНИЖЕНО с 0.30 (медленнее сходится)
B_CLIP_MIN: float = 0.01        # Минимальная вероятность гипотезы
B_CLIP_MAX: float = 0.99        # Максимальная вероятность гипотезы

# ========== Параметры гейта ==========
THETA_MB: float = 0.45
WEIGHTS_U: List[float] = [1.5, 3.5, 1.5, 2.0]  # [u_delta, u_s, u_v, u_c]
# u_s получил максимальный вес (3.5) для доминирования при неопределённости

# ========== Салиентность (веса для S_t) ==========
SALIENCE_DELTA: float = 0.30    # Вес u_delta в салиентности
SALIENCE_ENTROPY: float = 0.50  # Вес u_s (УВЕЛИЧЕНО)
SALIENCE_ERROR: float = 0.20    # Вес u_c

# ========== Параметры реологии ==========
K_USE: float = 0.08
K_MELT: float = 0.20
ALPHA_S: float = 15.0
TAU_S: float = 0.5
ETA_0: float = 1.0
ETA_MIN: float = 0.1
ETA_MAX: float = 15.0
LAMBDA_DECAY: float = 0.01

# Коэффициенты для V_G (вязкость гейта)
RHEO_G_HARDEN_CORRECT: float = 1.5
RHEO_G_MELT_ERROR: float = 1.0

# Коэффициенты для V_p (вязкость паттерна) — БОЛЕЕ АГРЕССИВНЫЕ
RHEO_P_HARDEN_CORRECT: float = 2.5  # УВЕЛИЧЕНО с 2.0
RHEO_P_MELT_ERROR: float = 1.8      # УВЕЛИЧЕНО с 1.5

# ========== Инициализация ==========
V_G_INIT: float = 0.10
V_P_INIT: float = 0.10
Q_INIT: float = 0.0

# ========== Диагностический вектор ==========
U_C_ERROR: float = 3.0          # u_c при ошибке
U_C_CORRECT: float = 0.0        # u_c при правильном действии
EMA_ALPHA: float = 0.30         # Для delta_ema

# ========== Sigmoid стабильность ==========
SIGMOID_CLIP_MIN: float = -500
SIGMOID_CLIP_MAX: float = 500

# ========== Метрики ==========
ROLLING_WINDOW: int = 10
HYSTERESIS_WINDOW: int = 5

# ========== Критерии Go/No-Go ==========
CRIT_EARLY_TRIAL_START: int = 20
CRIT_EARLY_TRIAL_END: int = 30
CRIT_LATE_TRIAL_START: int = 50
CRIT_LATE_TRIAL_END: int = 60
CRIT_ACCURACY_MIN: float = 0.85
CRIT_ACCURACY_GAIN_MIN: float = 0.15

CRIT_EXPLORE_BEFORE_WINDOW: int = 15
CRIT_EXPLORE_AFTER_WINDOW_START: int = 10
CRIT_EXPLORE_AFTER_WINDOW_END: int = 30
CRIT_EXPLORE_BEFORE_MAX: float = 0.30
CRIT_EXPLORE_AFTER_MIN: float = 0.25  # СНИЖЕНО с 0.30

CRIT_VG_AFTER_CHANGE_START: int = 15
CRIT_VG_AFTER_CHANGE_END: int = 35
CRIT_VG_GAIN_MIN: float = 0.15
CRIT_VG_DROP_MIN: float = 0.08  # СНИЖЕНО с 0.10

CRIT_HYSTERESIS_MIN: int = 2
CRIT_HYSTERESIS_MAX: int = 20   # УВЕЛИЧЕНО с 15
CRIT_EXPLORE_THRESHOLD: float = 0.25  # СНИЖЕНО с 0.30