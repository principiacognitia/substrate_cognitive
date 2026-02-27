"""
Агент — ВЕРСИЯ 5.0
Все параметры импортируются из config.
"""

import numpy as np
from config import (
    ALPHA_MF, ALPHA_MB, BETA_ACTION, BETA_STICK,
    V_G_INIT, V_P_INIT, Q_INIT, ETA_0, B_PRIOR
)
from belief import init_belief, update_belief, belief_entropy, predict_reward_from_belief, select_action_from_belief
from gate import compute_diagnostic_vector, gate_select, update_rheology

class RheologicalAgent:
    def __init__(self, changepoint: int = 60):
        self.Q_MF = {'choice': np.array([Q_INIT, Q_INIT], dtype=np.float64)}
        self.Q_MB = {'choice': np.array([Q_INIT, Q_INIT], dtype=np.float64)}
        
        self.b = init_belief(B_PRIOR)
        self.b_entropy = belief_entropy(self.b)
        
        self.eta_G = ETA_0
        self.V_G = V_G_INIT
        self.eta_p = ETA_0
        self.V_p = V_P_INIT
        
        self.delta_ema = 0.0
        self.last_action = None
        self.last_mode = 'EXPLOIT'
        self.last_u_t = np.array([0.0, 0.0, 0.0, 0.0])
        
        self.last_reward = None
        self.last_reward_pred = None
        self.last_correct_side = None
        
        self.changepoint = changepoint
        
    def predict_reward(self, action: int) -> float:
        return float(self.Q_MF['choice'][action])
    
    def predict_reward_from_belief(self, action: int) -> float:
        return predict_reward_from_belief(self.b, action)
    
    def select_action(self, mode: str) -> int:
        if mode == 'EXPLOIT':
            Q_values = self.Q_MF['choice'].copy()
            if self.last_action is not None:
                Q_values[self.last_action] += BETA_STICK * self.V_p
            exp_q = np.exp(BETA_ACTION * Q_values)
            probs = exp_q / exp_q.sum()
            action = int(np.random.choice([0, 1], p=probs))
        else:
            action = select_action_from_belief(self.b, beta=BETA_ACTION)
        
        self.last_action = action
        return action
    
    def update_Q(self, action: int, reward: float) -> None:
        if self.last_mode == 'EXPLOIT':
            self.Q_MF['choice'][action] += ALPHA_MF * (reward - self.Q_MF['choice'][action])
        else:
            self.Q_MB['choice'][action] += ALPHA_MB * (reward - self.Q_MB['choice'][action])
    
    def update_belief(self, action: int, reward: float) -> None:
        self.b = update_belief(self.b, action, reward)
        self.b_entropy = belief_entropy(self.b)
    
    def get_u_t_for_gate(self) -> np.ndarray:
        return self.last_u_t
    
    def end_trial(self, reward: float, action: int, correct_side: int, trial: int) -> tuple:
        last_action_was_correct = (action == correct_side)
        reward_pred = self.predict_reward_from_belief(action)
        
        u_t, new_delta_ema = compute_diagnostic_vector(
            reward, reward_pred, self.b_entropy, action, correct_side, self.delta_ema
        )
        
        self.last_u_t = u_t
        self.delta_ema = new_delta_ema
        self.last_reward = reward
        self.last_reward_pred = reward_pred
        self.last_correct_side = correct_side
        
        self.update_belief(action, reward)
        self.update_Q(action, reward)
        
        self.eta_G, self.V_G, self.eta_p, self.V_p = update_rheology(
            self.eta_G, self.eta_p, last_action_was_correct=last_action_was_correct
        )
        
        return self.V_G, self.V_p