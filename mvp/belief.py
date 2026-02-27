"""
Модуль belief state — ВЕРСИЯ 5.0
Все параметры импортируются из config.
"""

import numpy as np
from config import (
    B_PRIOR, B_CLIP_MIN, B_CLIP_MAX, BELIEF_LR
)

def init_belief(prior: float = B_PRIOR) -> np.ndarray:
    """Инициализирует belief над двумя гипотезами."""
    return np.array([prior, 1.0 - prior])

def update_belief(b: np.ndarray, action: int, reward: float, lr: float = BELIEF_LR) -> np.ndarray:
    """
    Байесовское обновление belief после наблюдения.
    """
    b_new = b.copy()
    
    if reward > 0:
        if action == 0:
            b_new[0] += lr * (1.0 - b_new[0])
            b_new[1] -= lr * b_new[1]
        else:
            b_new[1] += lr * (1.0 - b_new[1])
            b_new[0] -= lr * b_new[0]
    else:
        if action == 0:
            b_new[0] -= lr * b_new[0]
            b_new[1] += lr * (1.0 - b_new[1])
        else:
            b_new[1] -= lr * b_new[1]
            b_new[0] += lr * (1.0 - b_new[0])
    
    b_new = np.clip(b_new, B_CLIP_MIN, B_CLIP_MAX)
    b_new = b_new / b_new.sum()
    
    return b_new

def belief_entropy(b: np.ndarray) -> float:
    """Вычисляет энтропию belief."""
    b = np.clip(b, 1e-10, 1.0)
    return -np.sum(b * np.log(b))

def predict_reward_from_belief(b: np.ndarray, action: int) -> float:
    """Предсказание награды на основе belief."""
    return float(b[action])

def select_action_from_belief(b: np.ndarray, beta: float = 5.0) -> int:
    """Выбор действия на основе belief (MB-режим)."""
    exp_b = np.exp(beta * b)
    probs = exp_b / exp_b.sum()
    return int(np.random.choice([0, 1], p=probs))