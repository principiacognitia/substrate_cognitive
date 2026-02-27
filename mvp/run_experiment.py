"""
Главный скрипт — ВЕРСИЯ 5.0
Выводит V_p и belief entropy в консоль.
"""

import numpy as np
from config import N_TRIALS, CHANGEPOINT_TRIAL
from environment import TMazeEnv
from agent import RheologicalAgent
from metrics import MetricsCollector
from visualize import plot_results
from gate import gate_select
from belief import belief_entropy

def run_trial(env, agent, trial_num):
    state = env.reset()
    total_reward = 0
    done = False
    action = None
    correct_side = env.get_correct_side()
    
    u_t = agent.get_u_t_for_gate()
    mode = gate_select(u_t, agent.V_G)
    agent.last_mode = mode
    
    while not done:
        if state == 'start':
            next_state, reward, done = env.step(None)
            state = next_state
        elif state == 'choice':
            action = agent.select_action(mode)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state
        elif state == 'reward':
            done = True
    
    env.end_trial()
    V_G, V_p = agent.end_trial(total_reward, action, correct_side, trial_num)
    
    return total_reward, mode, V_G, V_p, correct_side, action

def main():
    print("=" * 70)
    print("T-maze MVP: Rheological Cognitive Architecture v5.0")
    print("=" * 70)
    print(f"Trials: {N_TRIALS}, Changepoint: {CHANGEPOINT_TRIAL}")
    print(f"Belief LR: 0.12 (замедлено), ALPHA_MF: 0.12 (замедлено)")
    print(f"u_s weight: 3.5 (усилено)")
    print()
    print(f"{'Trial':>6} | {'Mode':>7} | {'V_G':>6} | {'V_p':>6} | {'b_entropy':>10} | {'Acc(10)':>7}")
    print("-" * 70)
    
    env = TMazeEnv(n_trials=N_TRIALS, changepoint=CHANGEPOINT_TRIAL, seed=42)
    agent = RheologicalAgent(changepoint=CHANGEPOINT_TRIAL)
    metrics = MetricsCollector()
    
    for trial in range(1, N_TRIALS + 1):
        reward, mode, V_G, V_p, correct_side, action = run_trial(env, agent, trial)
        
        metrics.record(
            trial=trial, action=action, reward=reward, mode=mode,
            V_G=V_G, V_p=V_p, correct_side=correct_side
        )
        
        if trial % 10 == 0 or trial == CHANGEPOINT_TRIAL or trial in [65, 70, 75, 80, 85, 90]:
            acc = np.mean([metrics.rewards[i] for i in range(max(0, trial-10), trial)])
            b_ent = belief_entropy(agent.b)
            print(f"{trial:6d} | {mode:>7} | {V_G:6.3f} | {V_p:6.3f} | {b_ent:10.4f} | {acc:7.2f}")
    
    print()
    print("=" * 60)
    print("Go/No-Go Analysis")
    print("=" * 60)
    
    results = metrics.check_go_no_go()
    
    for criterion, data in results.items():
        if criterion in ['verdict', 'passed_count', 'total_criteria']:
            continue
        status = "✓ PASS" if data['passed'] else "✗ FAIL"
        print(f"\n{criterion.upper()}: {status}")
        for key, val in data.items():
            if key != 'passed':
                print(f"  {key}: {val}")
    
    print()
    print("=" * 60)
    print(f"VERDICT: {results['verdict']} ({results['passed_count']}/{results['total_criteria']} criteria passed)")
    print("=" * 60)
    
    plot_results(metrics, results)
    
    return results

if __name__ == "__main__":
    results = main()