"""
Gate Module — выбор режима (EXPLOIT/EXPLORE) и вычисление диагностического вектора.
Версия для Stage 2 (Two-Step Task).
"""

import numpy as np

# Параметры по умолчанию (будут перезаписаны из config)
THETA_MB = 0.3
WEIGHTS_U = [1.5, 3.5, 1.5, 2.0]  # [u_delta, u_s, u_v, u_c]

def sigmoid(x):
    """Стабильная сигмоида."""
    return float(1 / (1 + np.exp(-np.clip(x, -500, 500))))

def compute_diagnostic_vector(reward, reward_pred, belief_entropy_prev, 
                               action, correct_side, delta_ema_prev,
                               weights=None):
    """
    Вычисляет вектор u_t = [u_delta, u_s, u_v, u_c]
    
    Args:
        reward: Полученная награда
        reward_pred: Предсказанная награда
        belief_entropy_prev: Энтропия belief (до обновления)
        action: Выбранное действие
        correct_side: Правильная сторона (для вычисления ошибки)
        delta_ema_prev: Предыдущее EMA значение delta
        weights: Веса для компонент (опционально)
    
    Returns:
        u_t: np.array([u_delta, u_s, u_v, u_c])
        delta_ema: Обновлённое EMA
    """
    # u_delta: ошибка предсказания награды
    u_delta = abs(reward - reward_pred)
    
    # u_s: неопределённость о скрытом правиле (энтропия belief)
    u_s = belief_entropy_prev
    
    # u_v: волатильность (EMA от delta)
    alpha_ema = 0.3
    delta_ema = (1 - alpha_ema) * delta_ema_prev + alpha_ema * u_delta
    u_v = delta_ema
    
    # u_c: ставки — только при ошибке действия
    is_error = (action != correct_side)
    u_c = 3.0 if is_error else 0.0
    
    return np.array([u_delta, u_s, u_v, u_c]), delta_ema

def gate_select(u_t: np.ndarray, V_G: float, theta_mb: float = THETA_MB, weights: list = None) -> str:
    """
    Выбор режима: EXPLOIT (MF) или EXPLORE (MB).
    
    Args:
        u_t: Диагностический вектор
        V_G: Вязкость гейта (0-1)
        theta_mb: Порог переключения
        weights: Веса для компонент
    
    Returns:
        mode: 'EXPLORE' или 'EXPLOIT'
    """
    """Выбор режима на основе полного u_t."""
    if weights is None:
        # Упрощенные веса:[u_delta, u_s, u_v, u_c]
        weights =[1.5, 1.5, 1.5, 0.0]
    
    theta_U = 1.5  # ВОССТАНОВЛЕННЫЙ БИАС! (Порог базового спокойствия)

    # Давление неопределенности теперь может падать близко к 0 в стабильности
    U = sigmoid(float(np.dot(weights, u_t)) - theta_U)
    explore = U * (1 - V_G) > theta_mb
    
    return 'EXPLORE' if explore else 'EXPLOIT'