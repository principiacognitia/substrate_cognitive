"""
Stage 2B: Reversal Learning Task.
Демонстрирует Двойную Диссоциацию (V_G vs V_p) на метриках персеверации и латентности.

Запуск:
    python -m stage2.reversal.run_reversal --n-seeds 30 --n-trials 2000
    python -m stage2.reversal.run_reversal --no-log --nodebug

Выходные данные:
    logs/reversal/{experiment_id}_trials.csv — полные логи триалов
    logs/reversal/{experiment_id}_results.csv — агрегированные метрики
    logs/reversal/{experiment_id}_meta.json — метаданные эксперимента
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
from typing import Dict, List, Tuple

from stage2.reversal.env_reversal import ReversalEnv
from stage2.core.baselines import (
    RheologicalAgent,
    RheologicalAgent_NoVG,
    RheologicalAgent_NoVp
)
from stage2.core.args import parse_args, print_debug, print_always
from stage2.core.logger import TrialLogger, ExperimentLogger


def run_reversal_single(
    agent_class,
    n_trials: int = 2000,
    reversal_trial: int = 1000,
    seed: int = 42,
    theta_mb: float = 0.30,
    theta_u: float = 1.5,
    alpha: float = 0.35,
    beta: float = 4.0,
    volatility_threshold: float = 0.50,
    logger: TrialLogger = None,
    verbose: bool = False,
    nodebug: bool = False
) -> Tuple[pd.DataFrame, Dict]:
    """
    Запускает один эксперимент Reversal.
    
    Returns:
        DataFrame с данными триалов
        dict с метриками (perseverative_errors, latency_to_explore, stickiness)
    """
    env = ReversalEnv(
        n_trials=n_trials,
        reversal_trial=reversal_trial,
        seed=seed
    )
    
    agent = agent_class(
        theta_mb=theta_mb,
        theta_u=theta_u,
        alpha=alpha,
        beta=beta,
        volatility_threshold=volatility_threshold,
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
        
        # Логирование
        if logger:
            logger.log_trial(
                trial=trial,
                seed=seed,
                s1=s1, a1=a1, s2=s2, a2=a2,
                trans_type=trans_type,
                reward=reward,
                mode=mode,
                V_G=agent.V_G,
                V_p=agent.V_p,
                u_delta=u_t_prev[0],
                u_s=u_t_prev[1],
                u_v=u_t_prev[2],
                u_c=u_t_prev[3]
            )
        
        # Отладочный вывод
        if verbose and not nodebug:
            if trial in [reversal_trial-10, reversal_trial, reversal_trial+5, reversal_trial+20]:
                print(f"  Trial {trial:4d} | Mode: {mode:7s} | a1: {a1} | Reward: {reward}")
    
    df = pd.DataFrame(data)
    metrics = analyze_reversal(df, reversal_trial=reversal_trial)
    
    return df, metrics


def infer_old_action(df: pd.DataFrame, reversal_trial: int, window: int = 100) -> int:
    """
    Вычисляет реальное правило, которое выучил агент до реверсала.
    """
    pre = df[(df["trial"] >= reversal_trial - window) & (df["trial"] < reversal_trial)]
    if len(pre) == 0:
        return int(df[df["trial"] < reversal_trial]["a1"].mode().iloc[0])
    return int(pre["a1"].mode().iloc[0])


def analyze_reversal(df: pd.DataFrame, reversal_trial: int = 1000, window: int = 100) -> Dict:
    """
    Анализирует результаты Reversal.
    
    Returns:
        dict с метриками
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
    
    return {
        'perseverative_errors': perseverative_errors,
        'latency_to_explore': latency_to_explore,
        'stickiness': stickiness,
        'old_action': old_action
    }


def run_reversal_study(
    n_seeds: int = 30,
    n_trials: int = 2000,
    reversal_trial: int = 1000,
    theta_mb: float = 0.30,
    theta_u: float = 1.5,
    alpha: float = 0.35,
    beta: float = 4.0,
    volatility_threshold: float = 0.50,
    output_dir: str = 'logs/reversal/',
    log_trials: bool = True,
    experiment_id: str = None,
    verbose: bool = False,
    nodebug: bool = False
) -> Tuple[pd.DataFrame, ExperimentLogger]:
    """
    Запускает полную абляцию по всем seeds и агентам.
    """
    agents = [
        ('Full', RheologicalAgent),
        ('NoVG', RheologicalAgent_NoVG),
        ('NoVp', RheologicalAgent_NoVp)
    ]
    
    # Создаём логгер эксперимента
    if experiment_id is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_id = f'reversal_{timestamp}'
    
    exp_logger = ExperimentLogger(
        experiment_name=experiment_id,
        log_dir=output_dir,
        config={
            'n_seeds': n_seeds,
            'n_trials': n_trials,
            'reversal_trial': reversal_trial,
            'theta_mb': theta_mb,
            'theta_u': theta_u,
            'alpha': alpha,
            'beta': beta,
            'volatility_threshold': volatility_threshold
        },
        description='Reversal Learning Task — Double Dissociation Study'
    )
    
    all_results = []
    
    print_always("=" * 70)
    print_always("Stage 2B: Reversal Task — Double Dissociation")
    print_always("=" * 70)
    print_always(f"Seeds: {n_seeds}, Trials: {n_trials}, Reversal: {reversal_trial}")
    print_always(f"Theta_MB: {theta_mb}, Theta_U: {theta_u}")
    print_always(f"Alpha: {alpha}, Beta: {beta}, Volatility Threshold: {volatility_threshold}")
    print_always(f"Logging: {'enabled' if log_trials else 'disabled'}")
    print_always("--" * 35)
    
    for agent_name, agent_class in agents:
        print_always(f"\nЗапуск {agent_name} агента...")
        
        for i in range(n_seeds):
            seed = 42 + i
            
            # Создаём trial-логгер для этого агента
            trial_logger = None
            if log_trials:
                trial_logger = exp_logger.get_trial_logger(suffix=f'{agent_name}_seed{seed}')
            
            if verbose and not nodebug:
                print(f"  Seed {seed}...")
            
            df, metrics = run_reversal_single(
                agent_class=agent_class,
                n_trials=n_trials,
                reversal_trial=reversal_trial,
                seed=seed,
                theta_mb=theta_mb,
                theta_u=theta_u,
                alpha=alpha,
                beta=beta,
                volatility_threshold=volatility_threshold,
                logger=trial_logger,
                verbose=verbose,
                nodebug=nodebug
            )
            
            result = {
                'agent_class': agent_class.__name__,
                'seed': seed,
                **metrics
            }
            all_results.append(result)
            
            # Прогресс
            if (i + 1) % 10 == 0 and not nodebug:
                print_always(f"  Прогресс {agent_name}: {i + 1}/{n_seeds}")
    
    results_df = pd.DataFrame(all_results)
    
    # Сохраняем агрегированные результаты
    if log_trials:
        exp_logger.save_results(results_df, 'reversal_results.csv')
        exp_logger.finalize()
    
    return results_df, exp_logger


def analyze_results(df: pd.DataFrame, nodebug: bool = False) -> Dict:
    """
    Анализирует результаты Reversal и выводит статистику.
    """
    print_always("\n" + "=" * 70)
    print_always("РЕЗУЛЬТАТЫ (Медианы и Средние)")
    print_always("=" * 70)
    
    for agent_name, agent_class in [
        ('Full', 'RheologicalAgent'),
        ('NoVG', 'RheologicalAgent_NoVG'),
        ('NoVp', 'RheologicalAgent_NoVp')
    ]:
        agent_data = df[df['agent_class'] == agent_class]
        
        if len(agent_data) == 0:
            print_always(f"\n⚠️ {agent_name} Agent: НЕТ ДАННЫХ!")
            continue
        
        pe = agent_data['perseverative_errors'].values
        lat = [l for l in agent_data['latency_to_explore'].values if l != 999]
        stick = agent_data['stickiness'].values
        
        print_always(f"\n{agent_name} Agent ({len(agent_data)} seeds):")
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
    
    stats_results = {}
    
    # 1. Влияние V_G на Латентность (Full vs NoVG)
    lat_full = [l for l in df[df['agent_class'] == 'RheologicalAgent']['latency_to_explore'].values if l != 999]
    lat_novg = [l for l in df[df['agent_class'] == 'RheologicalAgent_NoVG']['latency_to_explore'].values if l != 999]
    
    if len(lat_full) > 0 and len(lat_novg) > 0:
        u_lat, p_lat = stats.mannwhitneyu(lat_full, lat_novg, alternative='greater')
        n1, n2 = len(lat_full), len(lat_novg)
        effect_size = 1 - (2 * u_lat) / (n1 * n2)
        stats_results['latency'] = {'U': u_lat, 'p': p_lat, 'effect_size': effect_size}
        print_always(f"Ось 1 (Инерция контроля V_G): Full vs NoVG Latency -> p = {p_lat:.2e}, effect = {effect_size:.3f}")
    else:
        print_always("Ось 1 (Инерция контроля V_G): Недостаточно данных")
        p_lat = 1.0
    
    # 2. Влияние V_p на Персеверацию (Full vs NoVp)
    pe_full = df[df['agent_class'] == 'RheologicalAgent']['perseverative_errors'].values
    pe_novp = df[df['agent_class'] == 'RheologicalAgent_NoVp']['perseverative_errors'].values
    u_pe, p_pe = stats.mannwhitneyu(pe_full, pe_novp, alternative='greater')
    n1, n2 = len(pe_full), len(pe_novp)
    effect_size = 1 - (2 * u_pe) / (n1 * n2)
    stats_results['perseveration'] = {'U': u_pe, 'p': p_pe, 'effect_size': effect_size}
    print_always(f"Ось 2 (Инерция действия V_p): Full vs NoVp Perseveration -> p = {p_pe:.2e}, effect = {effect_size:.3f}")
    
    # 3. Full vs NoVp по Latency (проверка "ортогональности")
    lat_novp = [l for l in df[df['agent_class'] == 'RheologicalAgent_NoVp']['latency_to_explore'].values if l != 999]
    if len(lat_full) > 0 and len(lat_novp) > 0:
        u_lat_novp, p_lat_novp = stats.mannwhitneyu(lat_full, lat_novp, alternative='two-sided')
        n1, n2 = len(lat_full), len(lat_novp)
        effect_size = 1 - (2 * u_lat_novp) / (n1 * n2)
        stats_results['full_vs_novp_latency'] = {'U': u_lat_novp, 'p': p_lat_novp, 'effect_size': effect_size}
        print_always(f"Full vs NoVp Latency (secondary effect): p = {p_lat_novp:.2e}, effect = {effect_size:.3f}")
    
    # Финальный вердикт
    print_always("\n" + "=" * 70)
    if p_lat < 0.05 and p_pe < 0.05:
        print_always("✓ УСПЕХ: Двойная диссоциация доказана.")
        print_always("V_G и V_p dissociably управляют контролем и поведением.")
        print_always("=" * 70)
        return {'success': True, 'stats': stats_results}
    else:
        print_always("✗ ПРОВАЛ: Диссоциация статистически не подтверждена.")
        print_always("=" * 70)
        return {'success': False, 'stats': stats_results}


def main():
    """Точка входа для скрипта Reversal."""
    args = parse_args(
        description="Stage 2B: Reversal Learning Task — Double Dissociation Study",
        default_output_dir='logs/reversal/'
    )
    
    results_df, exp_logger = run_reversal_study(
        n_seeds=args.n_seeds,
        n_trials=args.n_trials,
        reversal_trial=args.changepoint,
        theta_mb=args.theta_mb,
        theta_u=args.theta_u,
        alpha=args.alpha,
        beta=args.beta,
        volatility_threshold=args.volatility_threshold,
        output_dir=args.output_dir,
        log_trials=not args.no_log,
        experiment_id=args.experiment_id,
        verbose=args.verbose,
        nodebug=args.nodebug
    )
    
    analysis_results = analyze_results(results_df, nodebug=args.nodebug)
    
    # Сохраняем статистику в JSON
    if not args.no_log:
        stats_path = os.path.join(args.output_dir, f'{exp_logger.experiment_name}_stats.json')

        # ИСПРАВЛЕНИЕ: Конвертируем numpy типы в Python native types
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(i) for i in obj]
            elif isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(analysis_results), f, indent=2, ensure_ascii=False)
        print_always(f"\nСтатистика сохранена в {stats_path}")
    
    return analysis_results['success']


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)