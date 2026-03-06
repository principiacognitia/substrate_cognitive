"""
Тестирование агентов в задаче Reversal Learning.
Доказывает наличие фазы "Mixture of Strategies" (Персеверация -> Исследование -> Новая привычка).
"""

import sys
sys.path.insert(0, 'E:/CRS-1/substrate_cognitive')

import numpy as np
import pandas as pd
from stage2.reversal.env_reversal import ReversalEnv
from stage2.core.baselines import RheologicalAgent, RheologicalAgent_NoVG, RheologicalAgent_NoVp
from stage2.core.args import print_always

def run_reversal_single(agent_class, seed=42):
    env = ReversalEnv(seed=seed)
    
    # Используем ТЕ ЖЕ параметры, что и в Two-Step! Это критично.
    agent = agent_class(theta_mb=0.30, theta_u=1.5, alpha=0.35, beta=4.0, seed=seed)
    
    u_t_prev = np.zeros(4)
    data =[]
    
    for trial in range(1, env.n_trials + 1):
        s1 = env.reset()
        
        # Симуляция выбора для NoVG и NoVp (по аналогии с debug_ablation)
        if hasattr(agent, 'get_mode'):
            a1 = agent.select_action_stage1(s1, u_t_prev)
            mode = agent.get_mode()
        else:
            # Для NoVG/NoVp мы используем monkey-patching только внутри runner'a или
            # предполагаем, что вы обновили их классы, чтобы они возвращали mode.
            # Так как мы переписали их как чистых наследников, get_mode() у них работает!
            a1 = agent.select_action_stage1(s1, u_t_prev)
            mode = agent.get_mode()
            
        s2, trans_type = env.step_stage1(a1)
        a2 = agent.select_action_stage2(s2)
        reward, done, info = env.step_stage2(a2)
        
        u_t_prev = agent.update(a1, a2, reward, s2, trans_type, s1)
        
        data.append({
            'trial': trial,
            'a1': a1,
            'reward': reward,
            'mode': mode,
            'is_reversal': info['is_reversal']
        })
        
    return pd.DataFrame(data)

def analyze_reversal(df, reversal_trial=150):
    post_rev = df[df['trial'] >= reversal_trial].copy()
    post_rev = post_rev.reset_index(drop=True)
    
    # 1. Perseverative Errors: сколько раз агент нажал старый рычаг ДО первого переключения
    old_correct_action = 0 # Левый рычаг был правильным до реверса
    first_switch_idx = post_rev[post_rev['a1'] != old_correct_action].index.min()
    
    if pd.isna(first_switch_idx):
        perseverative_errors = len(post_rev) # Не переключился вообще
    else:
        perseverative_errors = first_switch_idx

    # 2. Latency to Explore: сколько триалов потребовалось до расплавления V_G
    explore_idx = post_rev[post_rev['mode'] == 'EXPLORE'].index.min()
    latency_to_explore = explore_idx if not pd.isna(explore_idx) else 999
    
    return perseverative_errors, latency_to_explore

def main():
    print_always("=" * 70)
    print_always("Stage 2B: Reversal Task (30 seeds)")
    print_always("=" * 70)
    
    agents =[
        ('Full', RheologicalAgent),
        ('NoVG', RheologicalAgent_NoVG),
        ('NoVp', RheologicalAgent_NoVp)
    ]
    
    for agent_name, agent_class in agents:
        pers_errors = []
        latencies =[]
        
        for seed in range(42, 72):
            df = run_reversal_single(agent_class, seed=seed)
            pe, lat = analyze_reversal(df, reversal_trial=150)
            pers_errors.append(pe)
            latencies.append(lat)
            
        print_always(f"\n{agent_name} Agent:")
        print_always(f"  Персеверативные ошибки: {np.mean(pers_errors):.1f} ± {np.std(pers_errors):.1f}")
        
        valid_lat =[l for l in latencies if l != 999]
        if len(valid_lat) > 0:
            print_always(f"  Латентность до EXPLORE: {np.mean(valid_lat):.1f} ± {np.std(valid_lat):.1f}")
        else:
            print_always(f"  Латентность до EXPLORE: НИКОГДА (999)")

if __name__ == "__main__":
    main()