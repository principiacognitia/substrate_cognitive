"""
Level 4: Ablation Test — вклад отдельных компонентов.
Сравнивает: (i) Full, (ii) No V_G, (iii) No Belief
"""

import numpy as np
import csv
from env_twostep import TwoStepEnv
from agent_twostep import RheologicalAgent, RheologicalAgent_NoVG, RheologicalAgent_NoBelief

def run_and_measure_hysteresis(agent_class, seed=42, n_trials=300):
    """Запускает агента и измеряет гистерезис переключения."""
    env = TwoStepEnv(seed=seed, n_trials=n_trials, with_changepoint=True)
    agent = agent_class()
    
    mode_history = []
    changepoint_trial = 150
    
    for trial in range(n_trials):
        s1 = env.reset()
        a1 = agent.select_action_stage1(s1)
        s2, trans_type = env.step_stage1(a1)
        a2 = agent.select_action_stage2(s2)
        reward, done, _ = env.step_stage2(a2)
        agent.update(a1, a2, reward, s2, trans_type)
        
        mode_history.append(1 if agent.get_mode() == 'EXPLORE' else 0)
    
    # Измерение гистерезиса
    pre_cp = mode_history[changepoint_trial-20:changepoint_trial]
    post_cp = mode_history[changepoint_trial:changepoint_trial+30]
    
    explore_pre = np.mean(pre_cp)
    explore_post = np.mean(post_cp)
    hysteresis_index = explore_post - explore_pre
    
    return {
        'explore_pre': explore_pre,
        'explore_post': explore_post,
        'hysteresis_index': hysteresis_index
    }

def main():
    print("=" * 70)
    print("Level 4: Ablation Test — Вклад компонентов")
    print("=" * 70)
    
    # Full модель
    full = run_and_measure_hysteresis(RheologicalAgent)
    print(f"\nFull Model:")
    print(f"  EXPLORE pre-CP: {full['explore_pre']:.3f}")
    print(f"  EXPLORE post-CP: {full['explore_post']:.3f}")
    print(f"  Hysteresis Index: {full['hysteresis_index']:.3f}")
    
    # Без V_G (фиксированный гейт)
    no_vg = run_and_measure_hysteresis(RheologicalAgent_NoVG)
    print(f"\nNo V_G (fixed gate):")
    print(f"  EXPLORE pre-CP: {no_vg['explore_pre']:.3f}")
    print(f"  EXPLORE post-CP: {no_vg['explore_post']:.3f}")
    print(f"  Hysteresis Index: {no_vg['hysteresis_index']:.3f}")
    
    # Без Belief (только u_delta)
    no_belief = run_and_measure_hysteresis(RheologicalAgent_NoBelief)
    print(f"\nNo Belief (u_delta only):")
    print(f"  EXPLORE pre-CP: {no_belief['explore_pre']:.3f}")
    print(f"  EXPLORE post-CP: {no_belief['explore_post']:.3f}")
    print(f"  Hysteresis Index: {no_belief['hysteresis_index']:.3f}")
    
    # Валидация (Grok пункт 4)
    assert full['hysteresis_index'] > 0.15, "Full модель должна показывать гистерезис!"
    assert no_vg['hysteresis_index'] < full['hysteresis_index'] - 0.05, "Без V_G гистерезис должен уменьшаться!"
    assert no_belief['hysteresis_index'] < full['hysteresis_index'] - 0.05, "Без Belief гистерезис должен уменьшаться!"
    
    print("\n✓ Level 4: Ablation Test PASSED")
    print("  Каждый компонент вносит измеримый вклад в гистерезис")

if __name__ == "__main__":
    main()