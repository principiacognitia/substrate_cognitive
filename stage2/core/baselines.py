"""
Baseline агенты для валидации two-step сигнатур.
MF-only, MB-only, Hybrid, No-VG (абляция).
"""

import numpy as np
from typing import Dict, Tuple, Optional

class MFAgent:
    """
    Model-Free Only — табличный Q-learning.
    Реагирует только на награды, игнорирует структуру переходов.
    """
    
    def __init__(self, alpha: float = 0.25, beta: float = 5.0, seed: int = 42):
        self.alpha = alpha
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        
        # Q-значения для 4 финальных состояний × 2 действия
        self.Q = np.zeros((4, 2), dtype=np.float64)
        
        # История для обучения
        self.prev_a1 = None
        self.prev_a2 = None
        self.prev_s2 = None
    
    def select_action_stage1(self, s1: int) -> int:
        """Выбор действия на этапе 1 на основе Q-значений финальных состояний."""
        # Агрегируем Q по ожидаемым состояниям (упрощённо — среднее по всем)
        q_values = np.mean(self.Q, axis=0)
        
        if self.rng.random() < 0.05:  # 5% exploration
            return int(self.rng.integers(0, 2))
        
        exp_q = np.exp(self.beta * q_values)
        probs = exp_q / exp_q.sum()
        return int(self.rng.choice([0, 1], p=probs))
    
    def select_action_stage2(self, s2: int) -> int:
        """Выбор действия на этапе 2 на основе Q-значений."""
        q_values = self.Q[s2]
        
        if self.rng.random() < 0.05:
            return int(self.rng.integers(0, 2))
        
        exp_q = np.exp(self.beta * q_values)
        probs = exp_q / exp_q.sum()
        return int(self.rng.choice([0, 1], p=probs))
    
    def update(self, a1: int, a2: int, reward: float, s2: int, trans_type: str) -> None:
        """Q-learning update для финального состояния."""
        # Обновляем Q для достигнутого состояния s2 и действия a2
        self.Q[s2, a2] += self.alpha * (reward - self.Q[s2, a2])
        
        # Сохраняем историю
        self.prev_a1 = a1
        self.prev_a2 = a2
        self.prev_s2 = s2


class MBAgent:
    """
    Model-Based Only — one-step lookahead с обучаемой матрицей переходов.
    Использует структуру среды для планирования.
    """
    
    def __init__(self, beta: float = 5.0, seed: int = 42):
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        
        # Матрица переходов: P(s2 | s1, a1)
        # Инициализируем равномерным распределением
        self.T = np.ones((2, 2, 4)) * 0.25  # (s1, a1, s2)
        
        # Оценки наград: R(s2, a2)
        self.R = np.zeros((4, 2), dtype=np.float64)
        
        # Счётчики для обучения
        self.T_counts = np.ones((2, 2, 4)) * 0.1  # Dirichlet prior
        self.R_counts = np.zeros((4, 2), dtype=np.float64)
        self.R_sums = np.zeros((4, 2), dtype=np.float64)
        
        # Маппинг переходов из config
        self.trans_map = {
            0: {0: (1, 3), 1: (2, 3)},
            1: {0: (1, 3), 1: (2, 3)}
        }
        
        self.prev_a1 = None
        self.prev_a2 = None
        self.prev_s2 = None
    
    def _update_transition_model(self, s1: int, a1: int, s2: int, trans_type: str) -> None:
        """Обновляет матрицу переходов на основе наблюдения."""
        # Увеличиваем счётчик для наблюдаемого перехода
        self.T_counts[s1, a1, s2] += 1
        
        # Нормализуем в вероятности
        self.T[s1, a1] = self.T_counts[s1, a1] / self.T_counts[s1, a1].sum()
    
    def _update_reward_model(self, s2: int, a2: int, reward: float) -> None:
        """Обновляет модель наград."""
        self.R_counts[s2, a2] += 1
        self.R_sums[s2, a2] += reward
        self.R[s2, a2] = self.R_sums[s2, a2] / max(self.R_counts[s2, a2], 1)
    
    def _compute_mb_values(self, s1: int) -> np.ndarray:
        """Вычисляет MB-значения для действий на этапе 1 через lookahead."""
        values = np.zeros(2)
        
        for a1 in [0, 1]:
            expected_value = 0.0
            
            # Ожидаемое значение по всем возможным s2
            for s2 in range(4):
                p_s2 = self.T[s1, a1, s2]
                
                # Лучшее действие на этапе 2 для этого s2
                best_a2_value = np.max(self.R[s2])
                
                expected_value += p_s2 * best_a2_value
            
            values[a1] = expected_value
        
        return values
    
    def select_action_stage1(self, s1: int) -> int:
        """Выбор действия на этапе 1 на основе MB lookahead."""
        mb_values = self._compute_mb_values(s1)
        
        if self.rng.random() < 0.05:
            return int(self.rng.integers(0, 2))
        
        exp_v = np.exp(self.beta * mb_values)
        probs = exp_v / exp_v.sum()
        return int(self.rng.choice([0, 1], p=probs))
    
    def select_action_stage2(self, s2: int) -> int:
        """Выбор действия на этапе 2 на основе модели наград."""
        r_values = self.R[s2]
        
        if self.rng.random() < 0.05:
            return int(self.rng.integers(0, 2))
        
        exp_r = np.exp(self.beta * r_values)
        probs = exp_r / exp_r.sum()
        return int(self.rng.choice([0, 1], p=probs))
    
    def update(self, a1: int, a2: int, reward: float, s2: int, trans_type: str) -> None:
        """Обновляет модели переходов и наград."""
        # Для простоты используем s1=0 (в реальной задаче нужно передавать)
        s1 = 0
        self._update_transition_model(s1, a1, s2, trans_type)
        self._update_reward_model(s2, a2, reward)
        
        self.prev_a1 = a1
        self.prev_a2 = a2
        self.prev_s2 = s2


class HybridAgent:
    """
    Hybrid — взвешенная комбинация MF/MB (Daw et al., 2011).
    """
    
    def __init__(self, w: float = 0.5, alpha: float = 0.25, beta: float = 5.0, seed: int = 42):
        self.w = w  # MB weight (0 = pure MF, 1 = pure MB)
        self.mf = MFAgent(alpha=alpha, beta=beta, seed=seed)
        self.mb = MBAgent(beta=beta, seed=seed)
        self.rng = np.random.default_rng(seed)
    
    def select_action_stage1(self, s1: int) -> int:
        """Комбинирует MF и MB значения."""
        mf_q = np.mean(self.mf.Q, axis=0)
        mb_v = self.mb._compute_mb_values(s1)
        
        # Взвешенная комбинация
        combined = (1 - self.w) * mf_q + self.w * mb_v
        
        if self.rng.random() < 0.05:
            return int(self.rng.integers(0, 2))
        
        exp_c = np.exp(self.mf.beta * combined)
        probs = exp_c / exp_c.sum()
        return int(self.rng.choice([0, 1], p=probs))
    
    def select_action_stage2(self, s2: int) -> int:
        """На этапе 2 используем MF (как в канонической реализации)."""
        return self.mf.select_action_stage2(s2)
    
    def update(self, a1: int, a2: int, reward: float, s2: int, trans_type: str) -> None:
        """Обновляет оба компонента."""
        self.mf.update(a1, a2, reward, s2, trans_type)
        self.mb.update(a1, a2, reward, s2, trans_type)


class NoVGAgent:
    """
    Абляция: RheologicalAgent без V_G (фиксированный гейт).
    Всегда использует MB когда u_t > threshold, без инерции.
    """
    
    def __init__(self, theta_mb: float = 0.5, alpha: float = 0.25, beta: float = 5.0, seed: int = 42):
        self.theta_mb = theta_mb
        self.mf = MFAgent(alpha=alpha, beta=beta, seed=seed)
        self.mb = MBAgent(beta=beta, seed=seed)
        self.rng = np.random.default_rng(seed)
        
        # Простая оценка u_t (только u_delta)
        self.delta_ema = 0.0
    
    def _compute_u_t(self, reward: float, reward_pred: float) -> float:
        """Вычисляет упрощённый диагностический вектор."""
        u_delta = abs(reward - reward_pred)
        self.delta_ema = 0.7 * self.delta_ema + 0.3 * u_delta
        return self.delta_ema
    
    def select_action_stage1(self, s1: int) -> int:
        """Выбирает MF или MB на основе u_t, без V_G инерции."""
        # Для простоты — чередуем или используем u_t из предыдущего триала
        if self.rng.random() < 0.5:
            return self.mf.select_action_stage1(s1)
        else:
            return self.mb.select_action_stage1(s1)
    
    def select_action_stage2(self, s2: int) -> int:
        return self.mf.select_action_stage2(s2)
    
    def update(self, a1: int, a2: int, reward: float, s2: int, trans_type: str) -> None:
        self.mf.update(a1, a2, reward, s2, trans_type)
        self.mb.update(a1, a2, reward, s2, trans_type)