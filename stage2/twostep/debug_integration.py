"""
Level 3: Integration Test — V_G Hysteresis and Mode Switching.
Демонстрация плавления мета-ригидности.
"""

import sys
sys.path.insert(0, 'E:/CRS-1/substrate_cognitive')

import numpy as np
import pandas as pd
from stage2.core.gate import THETA_MB
from stage2.twostep.env_twostep import TwoStepEnv
from stage2.twostep.agent_twostep import RheologicalAgent
from stage2.twostep.config_twostep import DEBUG_INTEGRATION_SEED

def main():
    print("=" * 70)
    print("Level 3: Integration Test — Rheological Agent")
    print("=" * 70)
    
    n_trials = 2000
    changepoint = 1000
    
    env = TwoStepEnv(n_trials=n_trials, seed=DEBUG_INTEGRATION_SEED, 
                     with_changepoint=True, changepoint_trial=changepoint)
    
    agent = RheologicalAgent(seed=DEBUG_INTEGRATION_SEED)
    
    data = []
    u_t_prev = np.array([0.0, 0.0, 0.0, 0.0])
    
    for trial in range(1, n_trials + 1):
        s1 = env.reset()
        a1 = agent.select_action_stage1(s1, u_t_prev)
        s2, trans_type = env.step_stage1(a1)
        a2 = agent.select_action_stage2(s2)
        reward, done, info = env.step_stage2(a2)
        
        mode = agent.get_mode()
        V_G = agent.V_G
        
        u_t_prev = agent.update(a1, a2, reward, s2, trans_type, s1)
        
        data.append({
            'trial': trial,
            'V_G': V_G,
            'u_delta': u_t_prev[0],
            'mode': 1 if mode == 'EXPLORE' else 0
        })
        
        # Логируем ключевые точки: до, во время и после смены правил
        if trial in[950, 990, 1000, 1005, 1010, 1020, 1050, 1100]:
            print(f"Trial {trial:4d} | Mode: {mode:7s} | V_G: {V_G:.3f} | u_delta: {u_t_prev[0]:.3f}")

    df = pd.DataFrame(data)
    
    # 1. Фаза поздней стабильности (до смены правил)
    explore_before = df[(df['trial'] >= 900) & (df['trial'] <= 1000)]['mode'].mean()
    
    # 2. Фаза шока и адаптации (сразу после смены правил)
    explore_shock = df[(df['trial'] > 1000) & (df['trial'] <= 1030)]['mode'].mean()
    
    # 3. Фаза выздоровления (агент выучил новые правила)
    explore_recovery = df[(df['trial'] >= 1070) & (df['trial'] <= 1100)]['mode'].mean()
    
    # Вычисление латентности (гистерезиса)
    post_change = df[df['trial'] > 1000]
    explore_trials = post_change[post_change['mode'] == 1]['trial'].values
    latency = explore_trials[0] - 1000 if len(explore_trials) > 0 else None
    
    print("\n" + "=" * 70)
    print(f"Анализ динамики Гейта с theta_mb = {THETA_MB:.2f}")
    print("=" * 70)
    print(f"EXPLORE до смены правил (900-1000):   {explore_before:.1%} (Ожидается < 20%)")
    print(f"EXPLORE в фазе шока (1001-1030):      {explore_shock:.1%} (Ожидается резкий рост)")
    print(f"EXPLORE после адаптации (1070-1100):  {explore_recovery:.1%} (Ожидается спад)")
    
    if latency is not None:
        print(f"\nЗАДЕРЖКА ПЕРЕКЛЮЧЕНИЯ (Hysteresis): {latency} триалов")
        if latency > 1:
            print("✓ УСПЕХ: Мета-ригидность подтверждена! Агент сопротивлялся переключению.")
        else:
            print("✗ ПРОВАЛ: Агент переключился мгновенно, инерция V_G не сработала.")
    else:
        print("\n✗ ПРОВАЛ: Агент вообще не включил EXPLORE.")

if __name__ == "__main__":
    main()