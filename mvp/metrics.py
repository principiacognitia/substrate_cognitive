"""
Метрики — ВЕРСИЯ 5.0
Все критерии импортируются из config.
"""

import numpy as np
from config import (
    ROLLING_WINDOW, HYSTERESIS_WINDOW, CHANGEPOINT_TRIAL,
    CRIT_EARLY_TRIAL_START, CRIT_EARLY_TRIAL_END,
    CRIT_LATE_TRIAL_START, CRIT_LATE_TRIAL_END,
    CRIT_ACCURACY_MIN, CRIT_ACCURACY_GAIN_MIN,
    CRIT_EXPLORE_BEFORE_WINDOW, CRIT_EXPLORE_AFTER_WINDOW_START, CRIT_EXPLORE_AFTER_WINDOW_END,
    CRIT_EXPLORE_BEFORE_MAX, CRIT_EXPLORE_AFTER_MIN,
    CRIT_VG_AFTER_CHANGE_START, CRIT_VG_AFTER_CHANGE_END,
    CRIT_VG_GAIN_MIN, CRIT_VG_DROP_MIN,
    CRIT_HYSTERESIS_MIN, CRIT_HYSTERESIS_MAX, CRIT_EXPLORE_THRESHOLD
)

class MetricsCollector:
    def __init__(self):
        self.trials = []
        self.actions = []
        self.rewards = []
        self.modes = []
        self.V_G_history = []
        self.V_p_history = []
        self.correct_sides = []
        
    def record(self, trial: int, action: int, reward: float, mode: str, V_G: float, V_p: float, correct_side: int):
        self.trials.append(trial)
        self.actions.append(action)
        self.rewards.append(reward)
        self.modes.append(mode)
        self.V_G_history.append(V_G)
        self.V_p_history.append(V_p)
        self.correct_sides.append(correct_side)
    
    def compute_accuracy(self, window: int = None):
        correct = np.array(self.actions) == np.array(self.correct_sides)
        if window is None:
            return correct.astype(float)
        accuracy = np.convolve(correct.astype(float), np.ones(window)/window, mode='valid')
        accuracy = np.concatenate([np.full(window-1, np.nan), accuracy])
        return accuracy
    
    def compute_explore_rate(self, window: int = None):
        explore = np.array([1 if m == 'EXPLORE' else 0 for m in self.modes])
        if window is None:
            return explore
        rate = np.convolve(explore.astype(float), np.ones(window)/window, mode='valid')
        rate = np.concatenate([np.full(window-1, np.nan), rate])
        return rate
    
    def check_go_no_go(self) -> dict:
        accuracy = self.compute_accuracy(window=ROLLING_WINDOW)
        explore_rate = self.compute_explore_rate(window=HYSTERESIS_WINDOW)
        results = {}
        
        # Критерий A: Обучение
        early_acc = np.nanmean(accuracy[CRIT_EARLY_TRIAL_START:CRIT_EARLY_TRIAL_END])
        late_acc = np.nanmean(accuracy[CRIT_LATE_TRIAL_START:CRIT_LATE_TRIAL_END])
        results['learning'] = {
            'early_accuracy': float(early_acc),
            'late_accuracy': float(late_acc),
            'passed': bool(late_acc > CRIT_ACCURACY_MIN and (late_acc - early_acc) > CRIT_ACCURACY_GAIN_MIN)
        }
        
        # Критерий B: Переключение гейта
        explore_before = np.nanmean(explore_rate[CHANGEPOINT_TRIAL-CRIT_EXPLORE_BEFORE_WINDOW:CHANGEPOINT_TRIAL])
        explore_after = np.nanmean(explore_rate[CHANGEPOINT_TRIAL+CRIT_EXPLORE_AFTER_WINDOW_START:CHANGEPOINT_TRIAL+CRIT_EXPLORE_AFTER_WINDOW_END])
        results['gate_switch'] = {
            'explore_before': float(explore_before),
            'explore_after': float(explore_after),
            'passed': bool(explore_before < CRIT_EXPLORE_BEFORE_MAX and explore_after > CRIT_EXPLORE_AFTER_MIN)
        }
        
        # Критерий C: Обучение гейта
        V_G_arr = np.array(self.V_G_history)
        V_G_early = float(np.mean(V_G_arr[10:25]))
        V_G_late_habit = float(np.mean(V_G_arr[50:60]))
        V_G_after_change = float(np.mean(V_G_arr[CHANGEPOINT_TRIAL+CRIT_VG_AFTER_CHANGE_START:CHANGEPOINT_TRIAL+CRIT_VG_AFTER_CHANGE_END]))
        results['gate_learning'] = {
            'V_G_early': V_G_early,
            'V_G_late_habit': V_G_late_habit,
            'V_G_after_change': V_G_after_change,
            'passed': bool((V_G_late_habit > V_G_early + CRIT_VG_GAIN_MIN) and (V_G_after_change < V_G_late_habit - CRIT_VG_DROP_MIN))
        }
        
        # Критерий D: Гистерезис
        explore_post = explore_rate[CHANGEPOINT_TRIAL:]
        stable_switch_idx = np.where(explore_post > CRIT_EXPLORE_THRESHOLD)[0]
        if len(stable_switch_idx) > 0:
            latency = int(stable_switch_idx[0])
            results['hysteresis'] = {
                'latency_trials': latency,
                'passed': bool(CRIT_HYSTERESIS_MIN <= latency <= CRIT_HYSTERESIS_MAX)
            }
        else:
            results['hysteresis'] = {'latency_trials': None, 'passed': False}
        
        all_passed = all(r['passed'] for r in results.values())
        results['verdict'] = 'GO' if all_passed else 'NO-GO'
        results['passed_count'] = sum(1 for r in results.values() if isinstance(r, dict) and r.get('passed', False))
        results['total_criteria'] = len([r for r in results.values() if isinstance(r, dict)])
        
        return results