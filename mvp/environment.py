"""
T-maze среда для MVP.
Простая структура: Start → Choice → Left/Right → Reward
"""

import numpy as np
from config import (
    REWARD_CORRECT, REWARD_WRONG
)

class TMazeEnv:
    def __init__(self, n_trials=100, changepoint=50, seed=42):
        self.n_trials = n_trials
        self.changepoint = changepoint
        self.rng = np.random.default_rng(seed)
        
        # Правильная сторона: 0=Left, 1=Right
        self.correct_side = 0  # Начинаем с Left
        self.current_trial = 0
        
        # Статистика
        self.history = []
        
    def reset(self):
        """Сброс на начало триала."""
        self.state = 'start'
        return self.state
    
    def step(self, action):
        """
        action: 0=Left, 1=Right
        Returns: next_state, reward, done
        """
        if self.state == 'start':
            self.state = 'choice'
            return self.state, 0.0, False
        
        elif self.state == 'choice':
            self.action = action
            reward = REWARD_CORRECT if action == self.correct_side else REWARD_WRONG
            self.state = 'reward'
            return self.state, reward, True
        
        elif self.state == 'reward':
            return self.state, 0.0, True
        
        return self.state, 0.0, True
    
    def end_trial(self):
        """Вызывается в конце каждого триала для обновления среды."""
        self.current_trial += 1
        
        # Смена правил на changepoint
        if self.current_trial == self.changepoint:
            self.correct_side = 1 - self.correct_side  # Flip L↔R
        
        self.history.append({
            'trial': self.current_trial,
            'correct_side': self.correct_side,
            'changepoint': self.current_trial == self.changepoint
        })
    
    def get_correct_side(self):
        return self.correct_side