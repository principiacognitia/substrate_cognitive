"""
Baseline агенты для валидации two-step сигнатур.
Включает: MF, MB, Hybrid, NoVG, NoVp — все с единым интерфейсом.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from stage2.core.gate import gate_select, sigmoid
from stage2.core.rheology import update_rheology, ETA_0


# ============================================================================
# БАЗОВЫЕ АГЕНТЫ (MF / MB / Hybrid)
# ============================================================================

class MFAgent:
    """
    Model-Free Only — табличный Q-learning.
    """
    def __init__(self, alpha: float = 0.35, beta: float = 4.0, seed: int = 42):
        self.alpha = alpha
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        
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
        self.Q_stage2[s2, a2] += self.alpha * (reward - self.Q_stage2[s2, a2])
        self.Q_stage1[a1] += self.alpha * (reward - self.Q_stage1[a1])
        self.prev_a1 = a1
        self.prev_s1 = s1


class MBAgent:
    """
    Model-Based Only — one-step lookahead с max over a2.
    """
    def __init__(self, alpha: float = 0.35, beta: float = 4.0, seed: int = 42):
        self.alpha = alpha
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        
        self.T = np.ones((1, 2, 2), dtype=np.float64) * 0.5
        self.R = np.ones((2, 2), dtype=np.float64) * 0.5
        self.T_counts = np.ones((1, 2, 2), dtype=np.float64) * 0.1
        
        self.prev_a1 = None
        self.prev_s1 = None
    
    def _update_transition_model(self, s1: int, a1: int, s2: int, trans_type: str) -> None:
        self.T_counts[s1, a1, s2] += 1
        self.T[s1, a1] = self.T_counts[s1, a1] / self.T_counts[s1, a1].sum()
    
    def _update_reward_model(self, s2: int, a2: int, reward: float) -> None:
        self.R[s2, a2] += self.alpha * (reward - self.R[s2, a2])
    
    def _compute_mb_values(self, s1: int) -> np.ndarray:
        values = np.zeros(2)
        for a1 in [0, 1]:
            expected_value = 0.0
            for s2 in range(2):
                p_s2 = self.T[s1, a1, s2]
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
    """
    Hybrid — взвешенная комбинация MF/MB (Daw et al., 2011).
    """
    def __init__(self, w: float = 0.5, alpha: float = 0.35, beta: float = 4.0, seed: int = 42):
        self.w = w
        self.mf = MFAgent(alpha=alpha, beta=beta, seed=seed)
        self.mb = MBAgent(alpha=alpha, beta=beta, seed=seed)
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


# ============================================================================
# РЕОЛОГИЧЕСКИЙ АГЕНТ (ПОЛНАЯ ВЕРСИЯ)
# ============================================================================

class RheologicalAgent:
    """
    Полный агент с реологией гейта (V_G) и реологией паттерна (V_p).
    """
    def __init__(self, alpha: float = 0.35, beta: float = 4.0, seed: int = 42,
                 theta_mb: float = 0.30, theta_u: float = 1.5,
                 gate_weights: list = None, volatility_threshold: float = 0.50):
        self.alpha = alpha
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        
        # Параметры гейта (из config)
        self.theta_mb = theta_mb
        self.theta_u = theta_u
        self.gate_weights = gate_weights if gate_weights is not None else [1.5, 1.5, 1.5, 0.0]
        self.volatility_threshold = volatility_threshold
        
        # MF компоненты
        self.Q_stage1 = np.ones(2, dtype=np.float64) * 0.5
        self.Q_stage2 = np.ones((2, 2), dtype=np.float64) * 0.5
        
        # MB компоненты
        self.T = np.ones((1, 2, 2), dtype=np.float64) * 0.5
        self.R = np.ones((2, 2), dtype=np.float64) * 0.5
        self.T_counts = np.ones((1, 2, 2), dtype=np.float64) * 0.1
        
        # Реология
        self.eta_G = ETA_0
        self.V_G = 0.5
        self.eta_p = ETA_0
        self.V_p = 0.5
        
        # Диагностика
        self.delta_ema = 0.0
        self.last_mode = 'EXPLOIT'
        
        # История
        self.prev_a1 = None
        self.prev_s1 = None
    
    def _compute_mb_values(self, s1: int) -> np.ndarray:
        values = np.zeros(2)
        for a1 in [0, 1]:
            expected = 0.0
            for s2 in range(2):
                p_s2 = self.T[s1, a1, s2]
                best_a2 = np.max(self.R[s2])
                expected += p_s2 * best_a2
            values[a1] = expected
        return values

    def _get_u_t(self, reward: float, s2: int, a2: int) -> np.ndarray:
        """Вычисляет вектор диагностики u_t = [u_delta, u_s, u_v, u_c]."""
        u_delta = abs(reward - self.Q_stage2[s2, a2])
        
        # Policy entropy (не belief entropy!)
        exp_q = np.exp(self.beta * self.Q_stage1)
        probs = exp_q / exp_q.sum()
        u_s = -np.sum(probs * np.log(probs + 1e-10))
        
        self.delta_ema = 0.7 * self.delta_ema + 0.3 * u_delta
        u_v = self.delta_ema
        u_c = 0.0
        
        return np.array([u_delta, u_s, u_v, u_c])

    def get_mode(self) -> str:
        return self.last_mode

    def select_action_stage1(self, s1: int, u_t_prev: np.ndarray) -> int:
        """Выбор режима (Gate) и действия на этапе 1."""
        U = sigmoid(float(np.dot(self.gate_weights, u_t_prev)) - self.theta_u)
        self.last_mode = 'EXPLORE' if U * (1 - self.V_G) > self.theta_mb else 'EXPLOIT'
        
        if self.last_mode == 'EXPLORE':
            values = self._compute_mb_values(s1)
        else:
            values = self.Q_stage1.copy()
            if self.prev_a1 is not None:
                values[self.prev_a1] += 0.5 * self.V_p
                
        if self.rng.random() < 0.10:
            return int(self.rng.integers(0, 2))
        exp_v = np.exp(self.beta * values)
        probs = exp_v / exp_v.sum()
        return int(self.rng.choice([0, 1], p=probs))

    def select_action_stage2(self, s2: int) -> int:
        q_values = self.Q_stage2[s2]
        if self.rng.random() < 0.10:
            return int(self.rng.integers(0, 2))
        exp_q = np.exp(self.beta * q_values)
        probs = exp_q / exp_q.sum()
        return int(self.rng.choice([0, 1], p=probs))

    def update(self, a1: int, a2: int, reward: float, s2: int, trans_type: str, s1: int = 0) -> np.ndarray:
        """Обновляет модели и реологию. Возвращает новый u_t."""
        u_t = self._get_u_t(reward, s2, a2)
        
        # Стабильность среды через волатильность
        is_environment_stable = (self.delta_ema < self.volatility_threshold)
        
        self.eta_G, self.V_G, self.eta_p, self.V_p = update_rheology(
            self.eta_G, self.eta_p, is_environment_stable
        )
        
        self.Q_stage2[s2, a2] += self.alpha * (reward - self.Q_stage2[s2, a2])
        self.Q_stage1[a1] += self.alpha * (reward - self.Q_stage1[a1])
        
        self.T_counts[s1, a1, s2] += 1
        self.T[s1, a1] = self.T_counts[s1, a1] / self.T_counts[s1, a1].sum()
        self.R[s2, a2] += self.alpha * (reward - self.R[s2, a2])
        
        self.prev_a1 = a1
        self.prev_s1 = s1
        
        return u_t


# ============================================================================
# АБЛЯЦИИ (NoVG / NoVp)
# ============================================================================

class RheologicalAgent_NoVG(RheologicalAgent):
    """
    Абляция: Вязкость гейта фиксирована на 0.
    Наследует всю логику родителя, но V_G принудительно обнуляется.
    """
    def __init__(self, alpha: float = 0.35, beta: float = 4.0, seed: int = 42,
                 theta_mb: float = 0.30, theta_u: float = 1.5,
                 gate_weights: list = None, volatility_threshold: float = 0.50):
        super().__init__(alpha=alpha, beta=beta, seed=seed,
                        theta_mb=theta_mb, theta_u=theta_u,
                        gate_weights=gate_weights, volatility_threshold=volatility_threshold)
        self.V_G = 0.0
        self.eta_G = 0.0
    
    def update(self, a1: int, a2: int, reward: float, s2: int, trans_type: str, s1: int = 0) -> np.ndarray:
        u_t = super().update(a1, a2, reward, s2, trans_type, s1)
        self.V_G = 0.0
        self.eta_G = 0.0
        return u_t


class RheologicalAgent_NoVp(RheologicalAgent):
    """
    Абляция: Вязкость паттерна фиксирована на 0.
    Наследует всю логику родителя, но V_p принудительно обнуляется.
    """
    def __init__(self, alpha: float = 0.35, beta: float = 4.0, seed: int = 42,
                 theta_mb: float = 0.30, theta_u: float = 1.5,
                 gate_weights: list = None, volatility_threshold: float = 0.50):
        super().__init__(alpha=alpha, beta=beta, seed=seed,
                        theta_mb=theta_mb, theta_u=theta_u,
                        gate_weights=gate_weights, volatility_threshold=volatility_threshold)
        self.V_p = 0.0
        self.eta_p = 0.0
    
    def update(self, a1: int, a2: int, reward: float, s2: int, trans_type: str, s1: int = 0) -> np.ndarray:
        u_t = super().update(a1, a2, reward, s2, trans_type, s1)
        self.V_p = 0.0
        self.eta_p = 0.0
        return u_t


class RheologicalAgent_NoReology(RheologicalAgent):
    """
    Абляция: Обе вязкости фиксированы на 0 (полное отключение реологии).
    """
    def __init__(self, alpha: float = 0.35, beta: float = 4.0, seed: int = 42,
                 theta_mb: float = 0.30, theta_u: float = 1.5,
                 gate_weights: list = None, volatility_threshold: float = 0.50):
        super().__init__(alpha=alpha, beta=beta, seed=seed,
                        theta_mb=theta_mb, theta_u=theta_u,
                        gate_weights=gate_weights, volatility_threshold=volatility_threshold)
        self.V_G = 0.0
        self.eta_G = 0.0
        self.V_p = 0.0
        self.eta_p = 0.0
    
    def update(self, a1: int, a2: int, reward: float, s2: int, trans_type: str, s1: int = 0) -> np.ndarray:
        u_t = super().update(a1, a2, reward, s2, trans_type, s1)
        self.V_G = 0.0
        self.eta_G = 0.0
        self.V_p = 0.0
        self.eta_p = 0.0
        return u_t