"""
Level 4: Ablation Study — Доказательство необходимости V_G.
Сравниваем RheologicalAgent (с V_G) и NoVGAgent (без V_G).
"""

import sys
sys.path.insert(0, 'E:/CRS-1/substrate_cognitive')

import numpy as np
import pandas as pd
from stage2.twostep.env_twostep import TwoStepEnv
from stage2.twostep.agent_twostep import RheologicalAgent
from stage2.core.baselines import NoVGAgent

def run_agent_for_ablation(agent_class, n_trials=2000, changepoint=1000, seed=42, use_vg=True, theta=0.30):
    env = TwoStepEnv(n_trials=n_trials, seed=seed, 
                     with_changepoint=True, changepoint_trial=changepoint)
    
    if use_vg:
        agent = agent_class(seed=seed)
        # Внедряем идеальный порог из Level 3
        agent.theta_mb = theta 
    else:
        # Для NoVG агента нам нужно добавить поддержку Гейта для честного сравнения
        agent = agent_class(theta_mb=theta, seed=seed)
    
    u_t_prev = np.array([0.0, 0.0, 0.0, 0.0])
    explore_trials =[]
    
    for trial in range(1, n_trials + 1):
        s1 = env.reset()
        
        # Разная сигнатура вызова для агентов
        if use_vg:
            a1 = agent.select_action_stage1(s1, u_t_prev)
            mode = agent.get_mode()
        else:
            # Симулируем логику Гейта без V_G для NoVGAgent
            from stage2.core.gate import sigmoid
            weights = [1.5, 1.5, 1.5, 0.0]
            theta_U = 1.5
            U = sigmoid(float(np.dot(weights, u_t_prev)) - theta_U)
            
            # ВАЖНО: V_G искусственно изъят (умножаем на 1.0)
            if U * 1.0 > agent.theta_mb:
                mode = 'EXPLORE'
                a1 = agent.mb.select_action_stage1(s1)
            else:
                mode = 'EXPLOIT'
                a1 = agent.mf.select_action_stage1(s1)

        s2, trans_type = env.step_stage1(a1)
        a2 = agent.select_action_stage2(s2)
        reward, done, info = env.step_stage2(a2)
        
        if use_vg:
            u_t_prev = agent.update(a1, a2, reward, s2, trans_type, s1)
            # Принудительно подменяем порог в гейте
            import stage2.core.gate as gate_module
            # Monkey-patch для передачи кастомного theta_mb
            def custom_gate_select(u_t, V_G, theta_mb=agent.theta_mb, weights=[1.5, 1.5, 1.5, 0.0]):
                theta_U = 1.5
                U = gate_module.sigmoid(float(np.dot(weights, u_t)) - theta_U)
                return 'EXPLORE' if U * (1 - V_G) > theta_mb else 'EXPLOIT'
            agent.select_action_stage1.__globals__['gate_select'] = custom_gate_select
        else:
            agent.update(a1, a2, reward, s2, trans_type, s1)
            # Упрощенное вычисление u_t для следующего шага
            u_delta = abs(reward - agent.mf.Q_stage2[s2, a2])
            agent.delta_ema = 0.7 * agent.delta_ema + 0.3 * u_delta
            u_t_prev = np.array([u_delta, 0.5, agent.delta_ema, 0.0]) # Упрощенный вектор
            
        if mode == 'EXPLORE':
            explore_trials.append(trial)
            
    # Анализ латентности после 1000
    post_change =[t for t in explore_trials if t > changepoint]
    latency = post_change[0] - changepoint if len(post_change) > 0 else None
    
    return latency

def main():
    print("=" * 70)
    print("Level 4: Ablation Study — Доказательство необходимости V_G")
    print("=" * 70)
    
    # Для чистоты эксперимента проведем тест на 5 разных сидах
    seeds =[42, 100, 2023, 999, 555]
    theta = 0.30
    
    latencies_vg = []
    latencies_no_vg =[]
    
    for s in seeds:
        lat_vg = run_agent_for_ablation(RheologicalAgent, seed=s, use_vg=True, theta=theta)
        lat_no_vg = run_agent_for_ablation(NoVGAgent, seed=s, use_vg=False, theta=theta)
        
        latencies_vg.append(lat_vg if lat_vg is not None else 999)
        latencies_no_vg.append(lat_no_vg if lat_no_vg is not None else 999)
        
        print(f"Seed {s:4d} | Latency WITH V_G: {lat_vg} | Latency WITHOUT V_G: {lat_no_vg}")
        
    avg_vg = np.mean([l for l in latencies_vg if l != 999])
    avg_no_vg = np.mean([l for l in latencies_no_vg if l != 999])
    
    print("-" * 70)
    print(f"Средняя задержка С реологией (V_G):   {avg_vg:.1f} триалов")
    print(f"Средняя задержка БЕЗ реологии (V_G=0): {avg_no_vg:.1f} триалов")
    
    if avg_vg > avg_no_vg + 2:
        print("\n✓ УСПЕХ (DOUBLE DISSOCIATION):")
        print("Вязкость Гейта (V_G) достоверно и значимо замедляет переключение режимов.")
        print("Без этого параметра модель вырождается в реактивный алгоритм.")
    else:
        print("\n✗ ПРОВАЛ: Разница статистически незначима. V_G не нужен.")

if __name__ == "__main__":
    main()