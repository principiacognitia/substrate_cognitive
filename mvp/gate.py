"""
Модуль гейта — ВЕРСИЯ 5.0
Все параметры импортируются из config.
"""

import numpy as np
from config import (
    THETA_MB, WEIGHTS_U, K_USE, K_MELT, ALPHA_S, TAU_S,
    ETA_0, ETA_MIN, ETA_MAX, LAMBDA_DECAY,
    SALIENCE_DELTA, SALIENCE_ENTROPY, SALIENCE_ERROR,
    RHEO_G_HARDEN_CORRECT, RHEO_G_MELT_ERROR,
    RHEO_P_HARDEN_CORRECT, RHEO_P_MELT_ERROR,
    SIGMOID_CLIP_MIN, SIGMOID_CLIP_MAX
)

def sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-np.clip(x, SIGMOID_CLIP_MIN, SIGMOID_CLIP_MAX))))

def compute_diagnostic_vector(
    reward: float,
    reward_pred: float,
    belief_entropy_prev: float,
    action: int,
    correct_side: int,
    delta_ema_prev: float
) -> tuple:
    """Вычисляет полный вектор u_t = [u_delta, u_s, u_v, u_c]."""
    from config import U_C_ERROR, U_C_CORRECT, EMA_ALPHA
    
    u_delta = abs(reward - reward_pred)
    u_s = belief_entropy_prev
    delta_ema = (1 - EMA_ALPHA) * delta_ema_prev + EMA_ALPHA * u_delta
    u_v = delta_ema
    
    is_error = (action != correct_side)
    u_c = U_C_ERROR if is_error else U_C_CORRECT
    
    return np.array([u_delta, u_s, u_v, u_c]), delta_ema

def gate_select(u_t: np.ndarray, V_G: float, theta_mb: float = THETA_MB, weights: list = WEIGHTS_U) -> str:
    """Выбор режима на основе полного u_t."""
    U = sigmoid(float(np.dot(weights, u_t)))
    explore = U * (1 - V_G) > theta_mb
    return 'EXPLORE' if explore else 'EXPLOIT'

def compute_salience(u_t: np.ndarray) -> float:
    """Салиентность с весами из config."""
    S_t = SALIENCE_DELTA * u_t[0] + SALIENCE_ENTROPY * u_t[1] + SALIENCE_ERROR * u_t[3]
    return float(S_t)

def update_rheology(eta_G: float, eta_p: float, last_action_was_correct: bool) -> tuple:
    """Обновление реологии с разделёнными коэффициентами для V_G и V_p."""
    if last_action_was_correct:
        hardening_G = K_USE * RHEO_G_HARDEN_CORRECT
        melting_G = 0.0
        hardening_p = K_USE * RHEO_P_HARDEN_CORRECT
        melting_p = 0.0
    else:
        hardening_G = 0.0
        melting_G = K_MELT * RHEO_G_MELT_ERROR
        hardening_p = 0.0
        melting_p = K_MELT * RHEO_P_MELT_ERROR
    
    decay_G = LAMBDA_DECAY * eta_G
    decay_p = LAMBDA_DECAY * eta_p
    
    eta_G_new = np.clip(eta_G + hardening_G - melting_G - decay_G, ETA_MIN, ETA_MAX)
    eta_p_new = np.clip(eta_p + hardening_p - melting_p - decay_p, ETA_MIN, ETA_MAX)
    
    V_G = sigmoid(np.log(eta_G_new / ETA_0))
    V_p = sigmoid(np.log(eta_p_new / ETA_0))
    
    return float(eta_G_new), float(V_G), float(eta_p_new), float(V_p)