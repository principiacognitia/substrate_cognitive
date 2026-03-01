"""
Level 4: Ablation Study — Финальная версия для публикации.
"""

import sys
sys.path.insert(0, 'E:/CRS-1/substrate_cognitive')

import numpy as np
import pandas as pd
from scipy import stats
from stage2.twostep.env_twostep import TwoStepEnv
from stage2.twostep.agent_twostep import RheologicalAgent
from stage2.core.baselines import NoVGAgent
from stage2.core.gate import sigmoid

def run_agent_for_ablation(agent_class, n_trials=2000, changepoint=1000, seed=42, use_vg=True, theta=0.30):
    env = TwoStepEnv(n_trials=n_trials, seed=seed, 
                     with_changepoint=True, changepoint_trial=changepoint)
    
    if use_vg:
        agent = agent_class(seed=seed)
    else:
        agent = agent_class(theta_mb=theta, seed=seed)
    
    u_t_prev = np.array([0.0, 0.0, 0.0, 0.0])
    explore_trials = []
    
    # Monkey-patch для RheologicalAgent с кастомным theta_mb
    if use_vg:
        original_select = agent.select_action_stage1
        def patched_select(s1, u_t):
            weights = [1.5, 1.5, 1.5, 0.0]
            theta_U = 1.5
            U = sigmoid(float(np.dot(weights, u_t)) - theta_U)
            agent.last_mode = 'EXPLORE' if U * (1 - agent.V_G) > theta else 'EXPLOIT'
            
            if agent.last_mode == 'EXPLORE':
                values = agent._compute_mb_values(s1)
            else:
                values = agent.Q_stage1.copy()
                if agent.prev_a1 is not None:
                    values[agent.prev_a1] += 0.5 * agent.V_p
                    
            if agent.rng.random() < 0.10:
                return int(agent.rng.integers(0, 2))
            exp_v = np.exp(agent.beta * values)
            probs = exp_v / exp_v.sum()
            return int(agent.rng.choice([0, 1], p=probs))
        agent.select_action_stage1 = patched_select
    
    for trial in range(1, n_trials + 1):
        s1 = env.reset()
        
        if use_vg:
            a1 = agent.select_action_stage1(s1, u_t_prev)
            mode = agent.last_mode
        else:
            weights = [1.5, 1.5, 1.5, 0.0]
            theta_U = 1.5
            U = sigmoid(float(np.dot(weights, u_t_prev)) - theta_U)
            
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
        else:
            agent.update(a1, a2, reward, s2, trans_type, s1)
            u_delta = abs(reward - agent.mf.Q_stage2[s2, a2])
            agent.delta_ema = 0.7 * agent.delta_ema + 0.3 * u_delta
            u_t_prev = np.array([u_delta, 0.5, agent.delta_ema, 0.0])
            
        if mode == 'EXPLORE':
            explore_trials.append(trial)
            
    post_change = [t for t in explore_trials if t > changepoint]
    latency = post_change[0] - changepoint if len(post_change) > 0 else 999
    
    return latency

def main():
    print("=" * 70)
    print("Level 4: Ablation Study — Доказательство необходимости V_G")
    print("=" * 70)
    print()
    
    # 30 семян для статистической мощности
    seeds = list(range(42, 42 + 30))
    theta = 0.30
    
    latencies_vg = []
    latencies_no_vg = []
    
    print("Запуск 30 симуляций (это займёт ~2 минуты)...")
    print()
    
    for i, s in enumerate(seeds):
        lat_vg = run_agent_for_ablation(RheologicalAgent, seed=s, use_vg=True, theta=theta)
        lat_no_vg = run_agent_for_ablation(NoVGAgent, seed=s, use_vg=False, theta=theta)
        
        latencies_vg.append(lat_vg)
        latencies_no_vg.append(lat_no_vg)
        
        if (i + 1) % 10 == 0:
            print(f"  Прогресс: {i + 1}/{len(seeds)}")
    
    # Фильтрация 999 (агент никогда не переключился)
    latencies_vg_valid = [l for l in latencies_vg if l != 999]
    latencies_no_vg_valid = [l for l in latencies_no_vg if l != 999]
    
    # Статистика
    mean_vg = np.mean(latencies_vg_valid) if latencies_vg_valid else 999
    mean_no_vg = np.mean(latencies_no_vg_valid) if latencies_no_vg_valid else 999
    median_vg = np.median(latencies_vg_valid) if latencies_vg_valid else 999
    median_no_vg = np.median(latencies_no_vg_valid) if latencies_no_vg_valid else 999
    std_vg = np.std(latencies_vg_valid) if latencies_vg_valid else 0
    std_no_vg = np.std(latencies_no_vg_valid) if latencies_no_vg_valid else 0
    
    # Статистический тест (Mann-Whitney U, т.к. распределение не нормальное)
    if len(latencies_vg_valid) > 3 and len(latencies_no_vg_valid) > 3:
        u_stat, p_value = stats.mannwhitneyu(latencies_vg_valid, latencies_no_vg_valid, alternative='greater')
    else:
        u_stat, p_value = None, None
    
    print()
    print("=" * 70)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 70)
    print(f"Количество семян: {len(seeds)}")
    print(f"Успешных переключений (WITH V_G): {len(latencies_vg_valid)}/{len(seeds)}")
    print(f"Успешных переключений (WITHOUT V_G): {len(latencies_no_vg_valid)}/{len(seeds)}")
    print()
    print(f"Средняя задержка С V_G:    {mean_vg:.1f} ± {std_vg:.1f} триалов")
    print(f"Средняя задержка БЕЗ V_G:  {mean_no_vg:.1f} ± {std_no_vg:.1f} триалов")
    print()
    print(f"Медиана С V_G:    {median_vg:.1f} триалов")
    print(f"Медиана БЕЗ V_G:  {median_no_vg:.1f} триалов")
    print()
    
    if p_value is not None:
        print(f"Статистический тест: Mann-Whitney U = {u_stat:.1f}, p = {p_value:.2e}")
        print()
        
        if p_value < 0.001:
            print("=" * 70)
            print("✓ УСПЕХ (p < 0.001):")
            print("Вязкость Гейта (V_G) достоверно и значимо замедляет переключение режимов.")
            print("Эффект готов для публикации.")
            print("=" * 70)
        elif p_value < 0.05:
            print("=" * 70)
            print("✓ УСПЕХ (p < 0.05):")
            print("Эффект статистически значим, но рекомендуется увеличить N семян.")
            print("=" * 70)
        else:
            print("=" * 70)
            print("✗ ПРОВАЛ (p > 0.05):")
            print("Разница не достигает статистической значимости.")
            print("=" * 70)
    else:
        print("Недостаточно данных для статистического теста.")

if __name__ == "__main__":
    main()