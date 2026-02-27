"""
Two-Step Task Environment (Daw et al., 2011).

Структура:
    Trial:
        Stage 1: State (A/B) → Action (Left/Right) → Transition (Common/Rare)
        Stage 2: State (1/2/3/4) → Action (Left/Right) → Reward (0/1)

Награды дрейфуют независимо для каждого из 4 финальных состояний.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from config_twostep import (
    N_TRIALS, N_STAGE1_STATES, N_STAGE2_STATES, N_ACTIONS,
    COMMON_TRANS_PROB, RARE_TRANS_PROB, STAGE1_TRANSITIONS,
    REWARD_MIN, REWARD_MAX, REWARD_DRIFT_RATE, REWARD_DRIFT_SD,
    INITIAL_REWARD_PROBS, WITH_CHANGEPOINT, CHANGEPOINT_TRIAL, LAPSE_RATE,
    SEED_ENV
)

class TwoStepEnv:
    """
    Two-Step Task Environment.
    
    Атрибуты:
        rng: Генератор случайных чисел
        current_trial: Номер текущего триала
        stage1_state: Состояние на этапе 1 (0=A, 1=B)
        stage2_state: Состояние на этапе 2 (0-3)
        transition_type: Тип перехода ('common' или 'rare')
        reward_probs: Текущие вероятности наград для 4 финальных состояний
    """
    
    def __init__(self, 
                 n_trials: int = N_TRIALS,
                 seed: int = SEED_ENV,
                 with_changepoint: bool = WITH_CHANGEPOINT,
                 changepoint_trial: int = CHANGEPOINT_TRIAL,
                 lapse_rate: float = LAPSE_RATE):
        """
        Инициализация среды.
        
        Args:
            n_trials: Количество триалов в сессии
            seed: Random seed
            with_changepoint: Добавить явный changepoint (инверсия наград)
            changepoint_trial: Триал явной смены правил
            lapse_rate: Вероятность моторной/перцептивной ошибки
        """
        self.n_trials = n_trials
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        self.with_changepoint = with_changepoint
        self.changepoint_trial = changepoint_trial
        self.lapse_rate = lapse_rate
        
        # Инициализация состояния
        self.current_trial = 0
        self.stage1_state = 0
        self.stage2_state = 0
        self.transition_type = None
        self.last_action_stage1 = None
        self.last_action_stage2 = None
        
        # Вероятности наград для 4 финальных состояний
        self.reward_probs = np.array(INITIAL_REWARD_PROBS, dtype=np.float64)
        
        # История для отладки
        self.history = []
    
    def reset(self) -> int:
        """
        Сброс на начало нового триала.
        
        Returns:
            stage1_state: Состояние на этапе 1 (0 или 1)
        """
        self.current_trial += 1
        self.stage1_state = self.rng.integers(0, N_STAGE1_STATES)
        self.stage2_state = None
        self.transition_type = None
        self.last_action_stage1 = None
        self.last_action_stage2 = None
        
        # Дрейф наград (после первого триала)
        if self.current_trial > 1:
            self._drift_rewards()
        
        # Явный changepoint (инверсия наград)
        if self.with_changepoint and self.current_trial == self.changepoint_trial:
            self._apply_changepoint()
        
        return self.stage1_state
    
    def _drift_rewards(self) -> None:
        """
        Медленный случайный дрейф вероятностей наград.
        Использует random walk с отражающими границами.
        """
        drift = self.rng.normal(0, REWARD_DRIFT_SD, size=4)
        self.reward_probs += drift * REWARD_DRIFT_RATE
        self.reward_probs = np.clip(self.reward_probs, REWARD_MIN, REWARD_MAX)
    
    def _apply_changepoint(self) -> None:
        """
        Явная смена правил: инверсия наград.
        """
        self.reward_probs = 1.0 - self.reward_probs
    
    def step_stage1(self, action: int) -> Tuple[int, str]:
        """
        Шаг этапа 1.
        
        Args:
            action: Действие (0 или 1)
        
        Returns:
            stage2_state: Состояние на этапе 2 (0-3)
            transition_type: 'common' или 'rare'
        """
        self.last_action_stage1 = action
        
        # Определяем переход
        trans_map = STAGE1_TRANSITIONS[self.stage1_state][action]
        
        if self.rng.random() < COMMON_TRANS_PROB:
            self.stage2_state = trans_map[0]
            self.transition_type = 'common'
        else:
            self.stage2_state = trans_map[1]
            self.transition_type = 'rare'
        
        return self.stage2_state, self.transition_type
    
    def step_stage2(self, action: int) -> Tuple[float, bool, Dict]:
        """
        Шаг этапа 2.
        
        Args:
            action: Действие (0 или 1)
        
        Returns:
            reward: Награда (0 или 1)
            done: Флаг завершения триала (всегда True)
            info: Дополнительная информация
        """
        self.last_action_stage2 = action
        
        # Лапс (моторная/перцептивная ошибка)
        if self.rng.random() < self.lapse_rate:
            action = 1 - action  # Инверсия действия
        
        # Вычисление награды
        reward_prob = self.reward_probs[self.stage2_state]
        reward = float(self.rng.random() < reward_prob)
        
        info = {
            'stage1_state': self.stage1_state,
            'stage2_state': self.stage2_state,
            'action_stage1': self.last_action_stage1,
            'action_stage2': action,
            'transition_type': self.transition_type,
            'reward_prob': reward_prob,
            'trial': self.current_trial
        }
        
        return reward, True, info
    
    def get_transition_type(self) -> Optional[str]:
        """
        Возвращает тип перехода текущего триала.
        
        Returns:
            'common', 'rare', или None если ещё не было перехода
        """
        return self.transition_type
    
    def get_reward_probs(self) -> np.ndarray:
        """
        Возвращает текущие вероятности наград.
        
        Returns:
            reward_probs: Массив из 4 вероятностей
        """
        return self.reward_probs.copy()
    
    def get_history(self) -> list:
        """
        Возвращает историю триалов.
        
        Returns:
            history: Список словарей с данными триалов
        """
        return self.history.copy()
    
    def log_trial(self, reward: float) -> None:
        """
        Логирует завершённый триал.
        
        Args:
            reward: Полученная награда
        """
        self.history.append({
            'trial': self.current_trial,
            'stage1_state': self.stage1_state,
            'stage2_state': self.stage2_state,
            'action_stage1': self.last_action_stage1,
            'action_stage2': self.last_action_stage2,
            'transition_type': self.transition_type,
            'reward': reward,
            'reward_probs': self.reward_probs.copy()
        })