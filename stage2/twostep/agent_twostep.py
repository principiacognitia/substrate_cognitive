"""
Rheological Agent для Two-Step Task.
Реализует S-O-R + Gate + V_G (мета-ригидность).
"""

import numpy as np
from stage2.core.gate import gate_select
from stage2.core.rheology import update_rheology, ETA_0

class RheologicalAgent:
    def __init__(self, alpha: float = 0.35, beta: float = 4.0, seed: int = 42):
        self.alpha = alpha
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        
        # MF компоненты
        self.Q_stage1 = np.ones(2, dtype=np.float64) * 0.5
        self.Q_stage2 = np.ones((2, 2), dtype=np.float64) * 0.5
        
        # MB компоненты
        self.T = np.ones((1, 2, 2), dtype=np.float64) * 0.5
        self.R = np.ones((2, 2), dtype=np.float64) * 0.5
        self.T_counts = np.ones((1, 2, 2), dtype=np.float64) * 0.1
        
        # Реология (V_G и V_p)
        self.eta_G = ETA_0
        self.V_G = 0.5
        self.eta_p = ETA_0
        self.V_p = 0.5
        
        # Диагностика
        self.delta_ema = 0.0
        self.last_mode = 'EXPLOIT'
        
        # История (для update)
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
        """Вычисляет вектор диагностики u_t."""
        # Ошибка предсказания (Surprise) относительно MF-ожиданий
        u_delta = abs(reward - self.Q_stage2[s2, a2])
        
        # Волатильность
        self.delta_ema = 0.7 * self.delta_ema + 0.3 * u_delta
        u_v = self.delta_ema
        
        # Энтропия политики (State/Policy uncertainty)
        exp_q = np.exp(self.beta * self.Q_stage1)
        probs = exp_q / exp_q.sum()
        u_s = -np.sum(probs * np.log(probs + 1e-10))
        
        # Ставки (Cost/Stakes) - пока константа для toy-модели
        u_c = 1.0 
        
        return np.array([u_delta, u_s, u_v, u_c])

    def get_mode(self) -> str:
        """Возвращает текущий режим агента."""
        return self.last_mode

    def select_action_stage1(self, s1: int, u_t_prev: np.ndarray) -> int:
        """Выбор режима (Gate) и действия на этапе 1."""
        # 1. Гейт принимает решение на основе прошлого u_t и текущего V_G
        self.last_mode = gate_select(u_t_prev, self.V_G)
        
        # 2. Исполнение политики выбранного режима
        if self.last_mode == 'EXPLORE':
            values = self._compute_mb_values(s1)
        else:
            values = self.Q_stage1.copy()
            # Добавляем инерцию действия (V_p) в режиме EXPLOIT
            if self.prev_a1 is not None:
                values[self.prev_a1] += 0.5 * self.V_p 
                
        # 3. Softmax выбор
        if self.rng.random() < 0.10:
            action = int(self.rng.integers(0, 2))
        else:
            exp_v = np.exp(self.beta * values)
            probs = exp_v / exp_v.sum()
            action = int(self.rng.choice([0, 1], p=probs))
            
        return action

    def select_action_stage2(self, s2: int) -> int:
        # Второй этап всегда реактивный (MF)
        q_values = self.Q_stage2[s2]
        if self.rng.random() < 0.10:
            return int(self.rng.integers(0, 2))
        
        exp_q = np.exp(self.beta * q_values)
        probs = exp_q / exp_q.sum()
        return int(self.rng.choice([0, 1], p=probs))

    def update(self, a1: int, a2: int, reward: float, s2: int, trans_type: str, s1: int = 0) -> np.ndarray:
        """Обновляет модели и реологию. Возвращает новый u_t."""
        # 1. Вычисляем диагностику ДО обновления весов
        u_t = self._get_u_t(reward, s2, a2)
        
        # Успешность для реологии (субъективная)
        last_action_was_correct = (reward > 0.5)
        
        # 2. Обновляем реологию (V_G и V_p)
        self.eta_G, self.V_G, self.eta_p, self.V_p = update_rheology(
            self.eta_G, self.eta_p, last_action_was_correct
        )
        
        # 3. Обновляем MF
        self.Q_stage2[s2, a2] += self.alpha * (reward - self.Q_stage2[s2, a2])
        self.Q_stage1[a1] += self.alpha * (reward - self.Q_stage1[a1])
        
        # 4. Обновляем MB
        self.T_counts[s1, a1, s2] += 1
        self.T[s1, a1] = self.T_counts[s1, a1] / self.T_counts[s1, a1].sum()
        self.R[s2, a2] += self.alpha * (reward - self.R[s2, a2])
        
        self.prev_a1 = a1
        self.prev_s1 = s1
        
        return u_t