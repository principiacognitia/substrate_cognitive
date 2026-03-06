"""
Reversal Task Environment.
Частный случай Two-Step Task с детерминированными переходами и инверсией наград.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from stage2.reversal.config_reversal import (
    N_TRIALS_REV, REVERSAL_TRIAL, PROB_HIGH, PROB_LOW, 
    TRANSITIONS_REV, INITIAL_REWARDS_REV
)

class ReversalEnv:
    def __init__(self, n_trials=N_TRIALS_REV, reversal_trial=REVERSAL_TRIAL, seed=42):
        self.n_trials = n_trials
        self.reversal_trial = reversal_trial
        self.rng = np.random.default_rng(seed)
        
        self.current_trial = 0
        self.stage1_state = 0
        self.stage2_state = None
        self.reward_probs = np.array(INITIAL_REWARDS_REV, dtype=np.float64)
        
    def reset(self) -> int:
        self.current_trial += 1
        self.stage1_state = 0
        
        # Скрытый реверсал правил на заданном триале
        if self.current_trial == self.reversal_trial:
            self.reward_probs = np.array([
                [PROB_LOW, PROB_LOW],
                [PROB_HIGH, PROB_HIGH]
            ], dtype=np.float64)
            
        return self.stage1_state
        
    def step_stage1(self, action: int) -> Tuple[int, str]:
        # Детерминированный переход
        self.stage2_state = TRANSITIONS_REV[self.stage1_state][action][0]
        return self.stage2_state, 'common'
        
    def step_stage2(self, action: int) -> Tuple[float, bool, Dict]:
        # Награда зависит только от достигнутого s2 (выбора a1)
        prob = self.reward_probs[self.stage2_state, action]
        reward = float(self.rng.random() < prob)
        
        info = {
            'trial': self.current_trial, 
            'is_reversal': self.current_trial >= self.reversal_trial
        }
        return reward, True, info