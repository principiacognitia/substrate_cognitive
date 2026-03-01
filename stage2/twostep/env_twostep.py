"""
Two-Step Task Environment (Daw et al., 2011).
ИСПРАВЛЕННАЯ ВЕРСИЯ: награда зависит от (s2, a2) — 2-armed bandit.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from config_twostep import (
    N_TRIALS, N_STAGE1_STATES, N_STAGE2_STATES, N_STAGE2_ACTIONS, N_ACTIONS,
    COMMON_TRANS_PROB, RARE_TRANS_PROB, STAGE1_TRANSITIONS,
    REWARD_MIN, REWARD_MAX, REWARD_DRIFT_RATE, REWARD_DRIFT_SD,
    INITIAL_REWARD_PROBS, WITH_CHANGEPOINT, CHANGEPOINT_TRIAL, LAPSE_RATE,
    SEED_ENV
)

class TwoStepEnv:
    def __init__(self, 
                 n_trials: int = N_TRIALS,
                 seed: int = SEED_ENV,
                 with_changepoint: bool = WITH_CHANGEPOINT,
                 changepoint_trial: int = CHANGEPOINT_TRIAL,
                 lapse_rate: float = LAPSE_RATE):
        self.n_trials = n_trials
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        self.with_changepoint = with_changepoint
        self.changepoint_trial = changepoint_trial
        self.lapse_rate = lapse_rate
        
        self.current_trial = 0
        self.stage1_state = 0
        self.stage2_state = 0
        self.transition_type = None
        self.last_action_stage1 = None
        self.last_action_stage2 = None
        
        # Награды: [s2][a2] — 2×2 матрица
        self.reward_probs = np.array(INITIAL_REWARD_PROBS, dtype=np.float64)
        self.history = []
    
    def reset(self) -> int:
        self.current_trial += 1
        self.stage1_state = 0  # Всегда s1=0
        self.stage2_state = None
        self.transition_type = None
        self.last_action_stage1 = None
        self.last_action_stage2 = None
        
        if self.current_trial > 1:
            self._drift_rewards()
        
        if self.with_changepoint and self.current_trial == self.changepoint_trial:
            self._apply_changepoint()
        
        return self.stage1_state
    
    def _drift_rewards(self) -> None:
        """Дрейф для каждой (s2, a2) пары независимо."""
        drift = self.rng.normal(0, REWARD_DRIFT_SD, size=(N_STAGE2_STATES, N_STAGE2_ACTIONS))
        self.reward_probs += drift * REWARD_DRIFT_RATE
        self.reward_probs = np.clip(self.reward_probs, REWARD_MIN, REWARD_MAX)
    
    def _apply_changepoint(self) -> None:
        """Инверсия всех вероятностей наград."""
        self.reward_probs = 1.0 - self.reward_probs
    
    def step_stage1(self, action: int) -> Tuple[int, str]:
        self.last_action_stage1 = action
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
        Награда зависит от (s2, a2) — 2-armed bandit!
        """
        self.last_action_stage2 = action
        
        # Лапс
        if self.rng.random() < self.lapse_rate:
            action = 1 - action
        
        # Награда от (s2, a2)
        reward_prob = self.reward_probs[self.stage2_state, action]
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
        return self.transition_type
    
    def get_reward_probs(self) -> np.ndarray:
        return self.reward_probs.copy()
    
    def get_history(self) -> list:
        return self.history.copy()
    
    def log_trial(self, reward: float) -> None:
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