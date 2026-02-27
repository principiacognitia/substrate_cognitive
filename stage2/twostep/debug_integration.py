"""
Level 3: Integration Test — RheologicalAgent в Two-Step среде.
Полное CSV-логгирование всех латентов.
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
from env_twostep import TwoStepEnv
from agent_twostep import RheologicalAgent

def main():
    env = TwoStepEnv(seed=42, n_trials=300)
    agent = RheologicalAgent()
    
    V_G_history = []
    mode_history = []
    
    # CSV logger (Grok пункт 3)
    with open('debug_integration_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['trial', 'a1', 'a2', 's2', 'reward', 'u_delta', 'u_s', 'u_v', 'V_G', 'mode', 'belief_entropy'])
        
        for trial in range(300):
            s1 = env.reset()
            a1 = agent.select_action_stage1(s1)
            s2, trans_type = env.step_stage1(a1)
            a2 = agent.select_action_stage2(s2)
            reward, done, _ = env.step_stage2(a2)
            agent.update(a1, a2, reward, s2, trans_type)
            
            # Логгирование всех латентов
            u_t = agent.get_u_t()
            writer.writerow([
                trial, a1, a2, s2, reward,
                u_t[0], u_t[1], u_t[2],  # u_delta, u_s, u_v
                agent.V_G, agent.get_mode(), agent.get_belief_entropy()
            ])
            
            V_G_history.append(agent.V_G)
            mode_history.append(1 if agent.get_mode() == 'EXPLORE' else 0)
    
    # Визуализация
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    axes[0].plot(V_G_history, 'g-', linewidth=2, label='V_G')
    axes[0].set_title('Gate Rheology')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(mode_history, 'orange', drawstyle='steps', label='Mode (1=EXPLORE)')
    axes[1].set_title('Gate Mode')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_title('CSV Log: debug_integration_log.csv')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('debug_integration.png', dpi=150)
    
    # Валидация
    assert np.std(V_G_history) > 0.05, f"V_G должна меняться! std = {np.std(V_G_history):.4f}"
    assert np.sum(mode_history) > 10, f"Должны быть EXPLORE триалы! count = {np.sum(mode_history)}"
    
    print("✓ Level 3: Integration Test PASSED")
    print(f"  V_G std: {np.std(V_G_history):.4f}")
    print(f"  EXPLORE trials: {np.sum(mode_history)}")

if __name__ == "__main__":
    main()