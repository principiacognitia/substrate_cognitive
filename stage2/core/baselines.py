"""
Baseline агенты для валидации two-step сигнатур.
ИСПРАВЛЕННАЯ ВЕРСИЯ: TD-обучение для модели наград в MB.
"""

import numpy as np
from typing import Dict, Tuple, Optional

class MFAgent:
    """
    Model-Free Only — табличный Q-learning.
    """
    def __init__(self, alpha: float = 0.35, beta: float = 4.0, seed: int = 42):
        self.alpha = alpha
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        
        # Q-значения: stage1 (2 действия), stage2 (2 состояния × 2 действия)
        self.Q_stage1 = np.ones(2, dtype=np.float64) * 0.5
        self.Q_stage2 = np.ones((2, 2), dtype=np.float64) * 0.5
        
        self.prev_a1 = None
        self.prev_s1 = None
    
    def select_action_stage1(self, s1: int) -> int:
        q_values = self.Q_stage1
        if self.rng.random() < 0.10:
            return int(self.rng.integers(0, 2))
        
        exp_q = np.exp(self.beta * q_values)
        probs = exp_q / exp_q.sum()
        return int(self.rng.choice([0, 1], p=probs))
    
    def select_action_stage2(self, s2: int) -> int:
        q_values = self.Q_stage2[s2]
        if self.rng.random() < 0.10:
            return int(self.rng.integers(0, 2))
        
        exp_q = np.exp(self.beta * q_values)
        probs = exp_q / exp_q.sum()
        return int(self.rng.choice([0, 1], p=probs))
    
    def update(self, a1: int, a2: int, reward: float, s2: int, trans_type: str, s1: int = 0) -> None:
        # Stage 2 update
        self.Q_stage2[s2, a2] += self.alpha * (reward - self.Q_stage2[s2, a2])
        # Stage 1 update (TD с наградой)
        self.Q_stage1[a1] += self.alpha * (reward - self.Q_stage1[a1])
        
        self.prev_a1 = a1
        self.prev_s1 = s1


class MBAgent:
    """
    Model-Based Only — one-step lookahead с max over a2.
    """
    def __init__(self, alpha: float = 0.35, beta: float = 4.0, seed: int = 42):
        self.alpha = alpha  # ИСПРАВЛЕНИЕ: MB тоже нужна альфа для наград!
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        
        # Матрица переходов: P(s2 | s1, a1) — размер (1, 2, 2)
        self.T = np.ones((1, 2, 2), dtype=np.float64) * 0.5
        # Награды: R(s2, a2) — размер (2, 2)
        self.R = np.ones((2, 2), dtype=np.float64) * 0.5
        
        # Счётчики (только для переходов, так как они стационарны)
        self.T_counts = np.ones((1, 2, 2), dtype=np.float64) * 0.1
        
        self.prev_a1 = None
        self.prev_s1 = None
    
    def _update_transition_model(self, s1: int, a1: int, s2: int, trans_type: str) -> None:
        self.T_counts[s1, a1, s2] += 1
        self.T[s1, a1] = self.T_counts[s1, a1] / self.T_counts[s1, a1].sum()
    
    def _update_reward_model(self, s2: int, a2: int, reward: float) -> None:
        # ИСПРАВЛЕНИЕ: TD-обновление вместо кумулятивного среднего!
        self.R[s2, a2] += self.alpha * (reward - self.R[s2, a2])
    
    def _compute_mb_values(self, s1: int) -> np.ndarray:
        values = np.zeros(2)
        for a1 in [0, 1]:
            expected_value = 0.0
            for s2 in range(2):
                p_s2 = self.T[s1, a1, s2]
                # max over a2 — основа MB-планирования
                best_a2_value = np.max(self.R[s2])
                expected_value += p_s2 * best_a2_value
            values[a1] = expected_value
        return values
    
    def select_action_stage1(self, s1: int) -> int:
        mb_values = self._compute_mb_values(s1)
        if self.rng.random() < 0.10:
            return int(self.rng.integers(0, 2))
        
        exp_v = np.exp(self.beta * mb_values)
        probs = exp_v / exp_v.sum()
        return int(self.rng.choice([0, 1], p=probs))
    
    def select_action_stage2(self, s2: int) -> int:
        r_values = self.R[s2]
        if self.rng.random() < 0.10:
            return int(self.rng.integers(0, 2))
        
        exp_r = np.exp(self.beta * r_values)
        probs = exp_r / exp_r.sum()
        return int(self.rng.choice([0, 1], p=probs))
    
    def update(self, a1: int, a2: int, reward: float, s2: int, trans_type: str, s1: int = 0) -> None:
        self._update_transition_model(s1, a1, s2, trans_type)
        self._update_reward_model(s2, a2, reward)
        self.prev_a1 = a1
        self.prev_s1 = s1


class HybridAgent:
    """Hybrid — взвешенная комбинация MF/MB."""
    def __init__(self, w: float = 0.5, alpha: float = 0.35, beta: float = 4.0, seed: int = 42):
        self.w = w
        self.mf = MFAgent(alpha=alpha, beta=beta, seed=seed)
        self.mb = MBAgent(alpha=alpha, beta=beta, seed=seed) # Исправлено: передаем alpha
        self.rng = np.random.default_rng(seed)
    
    def select_action_stage1(self, s1: int) -> int:
        mf_q = self.mf.Q_stage1.copy()
        mb_v = self.mb._compute_mb_values(s1)
        combined = (1 - self.w) * mf_q + self.w * mb_v
        
        if self.rng.random() < 0.10:
            return int(self.rng.integers(0, 2))
        
        exp_c = np.exp(self.mf.beta * combined)
        probs = exp_c / exp_c.sum()
        return int(self.rng.choice([0, 1], p=probs))
    
    def select_action_stage2(self, s2: int) -> int:
        return self.mf.select_action_stage2(s2)
    
    def update(self, a1: int, a2: int, reward: float, s2: int, trans_type: str, s1: int = 0) -> None:
        self.mf.update(a1, a2, reward, s2, trans_type, s1)
        self.mb.update(a1, a2, reward, s2, trans_type, s1)


class NoVGAgent:
    """Абляция: без V_G инерции."""
    def __init__(self, theta_mb: float = 0.5, alpha: float = 0.35, beta: float = 4.0, seed: int = 42):
        self.theta_mb = theta_mb
        self.mf = MFAgent(alpha=alpha, beta=beta, seed=seed)
        self.mb = MBAgent(alpha=alpha, beta=beta, seed=seed) # Исправлено: передаем alpha
        self.rng = np.random.default_rng(seed)
        self.delta_ema = 0.0
    
    def select_action_stage1(self, s1: int) -> int:
        if self.rng.random() < 0.5:
            return self.mf.select_action_stage1(s1)
        else:
            return self.mb.select_action_stage1(s1)
    
    def select_action_stage2(self, s2: int) -> int:
        return self.mf.select_action_stage2(s2)
    
    def update(self, a1: int, a2: int, reward: float, s2: int, trans_type: str, s1: int = 0) -> None:
        self.mf.update(a1, a2, reward, s2, trans_type, s1)
        self.mb.update(a1, a2, reward, s2, trans_type, s1)