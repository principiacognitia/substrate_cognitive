"""
Stage 2B: Reversal Learning Task.
Демонстрирует Двойную Диссоциацию (V_G vs V_p) на метриках персеверации и латентности.

Запуск:
    python -m stage2.reversal.run_reversal --n-seeds 30 --n-trials 2000 --changepoint 1000
    python -m stage2.reversal.run_reversal --nodebug --verbose
"""

import numpy as np
import pandas as pd
from scipy import stats
from stage2.reversal.env_reversal import ReversalEnv
from stage2.core.baselines import (
    RheologicalAgent, 
    RheologicalAgent_NoVG, 
    RheologicalAgent_NoVp
)
from stage2.core.args import parse_args, print_always, print_debug


def run_reversal_single(
    agent_class, 
    n_trials: int = 2000, 
    reversal_trial: int = 1000, 
    seed: int = 42,
    theta_mb: float = 0.30,
    theta_u: float = 1.5,
    alpha: float = 0.35,
    beta: float = 4.0,
    verbose: bool = False,
    nodebug: bool = False
):
    """
    Запускает один эксперимент Reversal.
    
    Returns:
        DataFrame с данными триалов
    """
    env = ReversalEnv(n_trials=n_trials, reversal_trial=reversal_trial, seed=seed)
    agent = agent_class(
        theta_mb=theta_mb, 
        theta_u=theta_u, 
        alpha=alpha, 
        beta=beta, 
        seed=seed
    )
    
    u_t_prev = np.zeros(4)
    data = []
    
    for trial in range(1, env.n_trials + 1):
        s1 = env.reset()
        
        # Получение режима и действия
        if hasattr(agent, 'get_mode'):
            a1 = agent.select_action_stage1(s1, u_t_prev)
            mode = agent.get_mode()
        else:
            a1 = agent.select_action_stage1(s1, u_t_prev)
            mode = getattr(agent, "last_mode", "EXPLOIT")
            
        s2, trans_type = env.step_stage1(a1)
        
        # МЕТОДОЛОГИЧЕСКОЕ ИСПРАВЛЕНИЕ: Фиксируем a2=0, чтобы исключить шум 2-го этапа
        a2 = 0 
        reward, done, info = env.step_stage2(a2)
        
        u_t_prev = agent.update(a1, a2, reward, s2, trans_type, s1)
        
        data.append({
            'trial': trial,
            'a1': a1,
            'reward': reward,
            'mode': mode,
            'is_reversal': info['is_reversal']
        })
        
        # Отладочный вывод
        if verbose and not nodebug:
            if trial in [reversal_trial-10, reversal_trial, reversal_trial+5, reversal_trial+20]:
                print(f"  Trial {trial:4d} | Mode: {mode:7s} | a1: {a1} | Reward: {reward}")
        
    return pd.DataFrame(data)


def infer_old_action(df: pd.DataFrame, reversal_trial: int, window: int = 100) -> int:
    """
    Вычисляет реальное правило, которое выучил агент до реверсала.
    
    Args:
        df: DataFrame с данными
        reversal_trial: Триал реверсала
        window: Окно для анализа пред-реверсального поведения
        
    Returns:
        old_action: Наиболее частое действие перед реверсалом
    """
    pre = df[(df["trial"] >= reversal_trial - window) & (df["trial"] < reversal_trial)]
    if len(pre) == 0:
        return int(df[df["trial"] < reversal_trial]["a1"].mode().iloc[0])
    return int(pre["a1"].mode().iloc[0])


def analyze_reversal(df: pd.DataFrame, reversal_trial: int = 1000, window: int = 100):
    """
    Анализирует результаты Reversal.
    
    Returns:
        perseverative_errors: Количество персеверативных ошибок
        latency_to_explore: Латентность до переключения в EXPLORE
        stickiness: Вероятность повторения действия
    """
    old_action = infer_old_action(df, reversal_trial, window=window)
    post = df[df["trial"] >= reversal_trial].reset_index(drop=True)
    
    # 1. Perseverative Errors (Влияние V_p)
    first_switch = post[post["a1"] != old_action].index.min()
    perseverative_errors = int(first_switch) if not pd.isna(first_switch) else int(len(post))
    
    # 2. Stickiness (Вероятность a1_t == a1_{t-1})
    post_prev = post["a1"].shift(1)
    stickiness = float((post["a1"] == post_prev).mean())
    
    # 3. Latency to Explore (Влияние V_G)
    explore_idx = post[post["mode"] == "EXPLORE"].index.min()
    latency_to_explore = int(explore_idx) if not pd.isna(explore_idx) else 999
    
    return perseverative_errors, latency_to_explore, stickiness


def main():
    """Точка входа для скрипта Reversal."""
    args = parse_args(description="Stage 2B: Reversal Learning Task")
    
    # Дополнительные аргументы специфичные для Reversal
    n_seeds = 30  # Хардкод для стандартизации
    
    print_always("=" * 70)
    print_always("Stage 2B: Reversal Task — Double Dissociation")
    print_always("=" * 70)
    print_always(f"Seeds: {n_seeds}, Trials: {args.n_trials}, Reversal: {args.changepoint}")
    print_always(f"Theta_MB: {args.theta_mb}, Theta_U: {args.theta_u}")
    print_always(f"Alpha: {args.alpha}, Beta: {args.beta}")
    print_always("--" * 35)
    
    agents = [
        ('Full', RheologicalAgent),
        ('NoVG', RheologicalAgent_NoVG),
        ('NoVp', RheologicalAgent_NoVp)
    ]
    
    results = {name: {'pe': [], 'lat': [], 'stick': []} for name, _ in agents}
    
    for agent_name, agent_class in agents:
        print_always(f"Запуск {agent_name} агента...")
        
        for i in range(n_seeds):
            seed = 42 + i
            
            print_debug(f"  Seed {seed}...", args, verbose=True)
            
            df = run_reversal_single(
                agent_class=agent_class, 
                n_trials=args.n_trials, 
                reversal_trial=args.changepoint, 
                seed=seed,
                theta_mb=args.theta_mb,
                theta_u=args.theta_u,
                alpha=args.alpha,
                beta=args.beta,
                verbose=args.verbose,
                nodebug=args.nodebug
            )
            
            pe, lat, stick = analyze_reversal(df, reversal_trial=args.changepoint)
            
            results[agent_name]['pe'].append(pe)
            results[agent_name]['lat'].append(lat)
            results[agent_name]['stick'].append(stick)
            
            # Прогресс
            if (i + 1) % 10 == 0 and not args.nodebug:
                print_always(f"  Прогресс {agent_name}: {i + 1}/{n_seeds}")
        
    # ВЫВОД РЕЗУЛЬТАТОВ
    print_always("=" * 70)
    print_always("РЕЗУЛЬТАТЫ (Медианы и Средние)")
    print_always("=" * 70)
    
    for agent_name in results.keys():
        pe = results[agent_name]['pe']
        lat = [l for l in results[agent_name]['lat'] if l != 999]
        stick = results[agent_name]['stick']
        
        print_always(f"\n{agent_name} Agent:")
        print_always(f"  Perseverative Errors:  {np.mean(pe):.1f} ± {np.std(pe):.1f} (Median: {np.median(pe):.1f})")
        
        if len(lat) > 0:
            print_always(f"  Latency to Explore:    {np.mean(lat):.1f} ± {np.std(lat):.1f} (Median: {np.median(lat):.1f})")
        else:
            print_always(f"  Latency to Explore:    НИКОГДА (999)")
        
        print_always(f"  Post-reversal Stickiness: {np.mean(stick):.1%}")
    
    # СТАТИСТИКА ДВОЙНОЙ ДИССОЦИАЦИИ
    print_always("\n" + "=" * 70)
    print_always("ДОКАЗАТЕЛЬСТВО ДВОЙНОЙ ДИССОЦИАЦИИ (Mann-Whitney U)")
    print_always("=" * 70)
    
    # 1. Влияние V_G на Латентность (Full vs NoVG)
    lat_full = [l for l in results['Full']['lat'] if l != 999]
    lat_novg = [l for l in results['NoVG']['lat'] if l != 999]
    
    if len(lat_full) > 0 and len(lat_novg) > 0:
        u_lat, p_lat = stats.mannwhitneyu(lat_full, lat_novg, alternative='greater')
        print_always(f"Ось 1 (Инерция контроля V_G): Full vs NoVG Latency -> p = {p_lat:.2e}")
    else:
        print_always("Ось 1 (Инерция контроля V_G): Недостаточно данных")
        p_lat = 1.0
    
    # 2. Влияние V_p на Персеверацию (Full vs NoVp)
    pe_full = results['Full']['pe']
    pe_novp = results['NoVp']['pe']
    u_pe, p_pe = stats.mannwhitneyu(pe_full, pe_novp, alternative='greater')
    print_always(f"Ось 2 (Инерция действия V_p): Full vs NoVp Perseveration -> p = {p_pe:.2e}")
    
    # Финальный вердикт
    print_always("\n" + "=" * 70)
    if p_lat < 0.05 and p_pe < 0.05:
        print_always("✓ УСПЕХ: Двойная диссоциация доказана.")
        print_always("V_G и V_p ортогонально управляют контролем и поведением.")
        print_always("=" * 70)
    else:
        print_always("✗ ПРОВАЛ: Диссоциация статистически не подтверждена.")
        print_always("=" * 70)


if __name__ == "__main__":
    main()