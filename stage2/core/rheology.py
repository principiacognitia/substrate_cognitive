"""
Rheology Module — обновление вязкости (V_G, V_p).
Версия для Stage 2.
"""

import numpy as np

# Параметры по умолчанию
K_USE = 0.08
K_MELT = 0.20
ALPHA_S = 15.0
TAU_S = 0.5
ETA_0 = 1.0
ETA_MIN = 0.1
ETA_MAX = 15.0
LAMBDA_DECAY = 0.01

# Коэффициенты для V_G
RHEO_G_HARDEN_CORRECT = 1.5
RHEO_G_MELT_ERROR = 1.0

# Коэффициенты для V_p
RHEO_P_HARDEN_CORRECT = 2.5
RHEO_P_MELT_ERROR = 1.8

def sigmoid(x):
    """Стабильная сигмоида."""
    return float(1 / (1 + np.exp(-np.clip(x, -500, 500))))

def eta_to_V(eta, eta_0=ETA_0):
    """Конвертация eta -> V через лог-нормализацию."""
    return sigmoid(np.log(eta / eta_0))

def update_rheology(eta_G, eta_p, last_action_was_correct,
                    k_use=None, k_melt=None, alpha_s=None, tau_s=None,
                    eta_0=None, eta_min=None, eta_max=None, lambda_decay=None,
                    rheo_g_harden=None, rheo_g_melt=None,
                    rheo_p_harden=None, rheo_p_melt=None):
    """
    Обновляет вязкость гейта (eta_G) и паттерна (eta_p).
    
    Args:
        eta_G: Текущая вязкость гейта
        eta_p: Текущая вязкость паттерна
        last_action_was_correct: Было ли действие правильным
        k_use, k_melt, ...: Параметры (опционально, используют значения по умолчанию)
    
    Returns:
        eta_G_new, V_G, eta_p_new, V_p
    """
    # Параметры по умолчанию
    k_use = k_use if k_use is not None else K_USE
    k_melt = k_melt if k_melt is not None else K_MELT
    alpha_s = alpha_s if alpha_s is not None else ALPHA_S
    tau_s = tau_s if tau_s is not None else TAU_S
    eta_0 = eta_0 if eta_0 is not None else ETA_0
    eta_min = eta_min if eta_min is not None else ETA_MIN
    eta_max = eta_max if eta_max is not None else ETA_MAX
    lambda_decay = lambda_decay if lambda_decay is not None else LAMBDA_DECAY
    rheo_g_harden = rheo_g_harden if rheo_g_harden is not None else RHEO_G_HARDEN_CORRECT
    rheo_g_melt = rheo_g_melt if rheo_g_melt is not None else RHEO_G_MELT_ERROR
    rheo_p_harden = rheo_p_harden if rheo_p_harden is not None else RHEO_P_HARDEN_CORRECT
    rheo_p_melt = rheo_p_melt if rheo_p_melt is not None else RHEO_P_MELT_ERROR
    
    # ===== V_G: вязкость гейта =====
    if last_action_was_correct:
        hardening_G = k_use * rheo_g_harden
        melting_G = 0.0
    else:
        hardening_G = 0.0
        melting_G = k_melt * rheo_g_melt
    
    decay_G = lambda_decay * eta_G
    eta_G_new = float(np.clip(eta_G + hardening_G - melting_G - decay_G, eta_min, eta_max))
    
    # ===== V_p: вязкость паттерна =====
    if last_action_was_correct:
        hardening_p = k_use * rheo_p_harden
        melting_p = 0.0
    else:
        hardening_p = 0.0
        melting_p = k_melt * rheo_p_melt
    
    decay_p = lambda_decay * eta_p
    eta_p_new = float(np.clip(eta_p + hardening_p - melting_p - decay_p, eta_min, eta_max))
    
    # ===== Конвертация eta -> V =====
    V_G = eta_to_V(eta_G_new, eta_0)
    V_p = eta_to_V(eta_p_new, eta_0)
    
    return eta_G_new, V_G, eta_p_new, V_p