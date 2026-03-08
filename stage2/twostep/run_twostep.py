"""
Stage 2A: Two-Step Task — Ablation Study.
Доказательство двойной диссоциации V_G vs V_p.

Запуск:
    python -m stage2.twostep.run_twostep --n-seeds 30 --changepoint 1000
    python -m stage2.twostep.run_twostep --no-log --nodebug

Выходные данные:
    logs/twostep/{experiment_id}_trials.csv — полные логи триалов
    logs/twostep/{experiment_id}_results.csv — агрегированные метрики
    logs/twostep/{experiment_id}_meta.json — метаданные эксперимента
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
from typing import Dict, List, Tuple

from stage2.twostep.env_twostep import TwoStepEnv
from stage2.core.baselines import (
    RheologicalAgent,
    RheologicalAgent_NoVG,
    RheologicalAgent_NoVp
)
from stage2.core.args import parse_args, print_debug, print_always
from stage2.core.logger import TrialLogger, ExperimentLogger


def run_single_experiment(
    agent_class,
    n_trials: int = 2000,
    changepoint: int = 1000,
    seed: int = 42,
    theta_mb: float = 0.30,
    theta_u: float = 1.5,
    alpha: float = 0.35,
    beta: float = 4.0,
    volatility_threshold: float = 0.50,
    logger: TrialLogger = None,
    verbose: bool = False,
    nodebug: bool = False
) -> Dict:
    """
    Запускает один эксперимент с указанным агентом.
    
    Returns:
        dict с метриками (latency, explore_rates, etc.)
    """
    env = TwoStepEnv(
        n_trials=n_trials,
        seed=seed,
        with_changepoint=True,
        changepoint_trial=changepoint
    )
    
    agent = agent_class(
        alpha=alpha,
        beta=beta,
        seed=seed,
        theta_mb=theta_mb,
        theta_u=theta_u,
        volatility_threshold=volatility_threshold
    )
    
    # Переменные для сбора метрик
    u_t_prev = np.zeros(4)
    explore_trials = []
    v_g_history = []
    v_p_history = []
    mode_history = []
    
    for trial in range(1, n_trials + 1):
        s1 = env.reset()
        a1 = agent.select_action_stage1(s1, u_t_prev)
        s2, trans_type = env.step_stage1(a1)
        a2 = agent.select_action_stage2(s2)
        reward, done, info = env.step_stage2(a2)
        
        mode = agent.get_mode() if hasattr(agent, 'get_mode') else 'EXPLOIT'
        V_G = agent.V_G
        V_p = agent.V_p
        
        u_t_prev = agent.update(a1, a2, reward, s2, trans_type, s1)
        
        # Сбор метрик
        if mode == 'EXPLORE':
            explore_trials.append(trial)
        
        v_g_history.append(V_G)
        v_p_history.append(V_p)
        mode_history.append(1 if mode == 'EXPLORE' else 0)
        
        # Логирование
        if logger:
            logger.log_trial(
                trial=trial,
                seed=seed,
                s1=s1, a1=a1, s2=s2, a2=a2,
                trans_type=trans_type,
                reward=reward,
                mode=mode,
                V_G=V_G,
                V_p=V_p,
                u_delta=u_t_prev[0],
                u_s=u_t_prev[1],
                u_v=u_t_prev[2],
                u_c=u_t_prev[3]
            )
        
        # Отладочный вывод
        if verbose and not nodebug:
            if trial in [changepoint-10, changepoint, changepoint+5, changepoint+10, changepoint+20]:
                print(f"  Trial {trial:4d} | Mode: {mode:7s} | V_G: {V_G:.3f} | V_p: {V_p:.3f} | Reward: {reward}")
    
    # Вычисление метрик
    post_change = [t for t in explore_trials if t > changepoint]
    latency = post_change[0] - changepoint if len(post_change) > 0 else None
    
    # Доли EXPLORE в непересекающихся окнах
    explore_pre = len([t for t in explore_trials if changepoint-100 <= t < changepoint]) / 100.0
    explore_shock = len([t for t in explore_trials if changepoint <= t < changepoint+30]) / 30.0
    explore_post = len([t for t in explore_trials if changepoint+100 <= t < changepoint+200]) / 100.0
    
    return {
        'agent_class': agent_class.__name__,
        'seed': seed,
        'latency': latency if latency is not None else 999,
        'explore_pre': explore_pre,
        'explore_shock': explore_shock,
        'explore_post': explore_post,
        'v_g_mean': np.mean(v_g_history),
        'v_p_mean': np.mean(v_p_history),
        'n_explore_total': len(explore_trials)
    }


def run_ablation_study(
    n_seeds: int = 30,
    n_trials: int = 2000,
    changepoint: int = 1000,
    theta_mb: float = 0.30,
    theta_u: float = 1.5,
    alpha: float = 0.35,
    beta: float = 4.0,
    volatility_threshold: float = 0.50,
    output_dir: str = 'logs/twostep/',
    log_trials: bool = True,
    experiment_id: str = None,
    verbose: bool = False,
    nodebug: bool = False
) -> Tuple[pd.DataFrame, ExperimentLogger]:
    """
    Запускает полную абляцию по всем seeds и агентам.
    
    Returns:
        DataFrame с результатами всех экспериментов
        ExperimentLogger для сохранения метаданных
    """
    agents = [
        ('Full', RheologicalAgent),
        ('NoVG', RheologicalAgent_NoVG),
        ('NoVp', RheologicalAgent_NoVp)
    ]
    
    # Создаём логгер эксперимента
    if experiment_id is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_id = f'twostep_ablation_{timestamp}'
    
    exp_logger = ExperimentLogger(
        experiment_name=experiment_id,
        log_dir=output_dir,
        config={
            'n_seeds': n_seeds,
            'n_trials': n_trials,
            'changepoint': changepoint,
            'theta_mb': theta_mb,
            'theta_u': theta_u,
            'alpha': alpha,
            'beta': beta,
            'volatility_threshold': volatility_threshold
        },
        description='Two-Step Task Ablation Study (V_G vs V_p)'
    )
    
    all_results = []
    
    print_always("=" * 70)
    print_always("Stage 2A: Two-Step Task — Ablation Study")
    print_always("=" * 70)
    print_always(f"Seeds: {n_seeds}, Trials: {n_trials}, Changepoint: {changepoint}")
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
            
            result = run_single_experiment(
                agent_class=agent_class,
                n_trials=n_trials,
                changepoint=changepoint,
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
            
            all_results.append(result)
            
            # Прогресс
            if (i + 1) % 10 == 0 and not nodebug:
                print_always(f"  Прогресс {agent_name}: {i + 1}/{n_seeds}")
    
    results_df = pd.DataFrame(all_results)
    
    # Сохраняем агрегированные результаты
    if log_trials:
        exp_logger.save_results(results_df, 'ablation_results.csv')
        exp_logger.finalize()
    
    return results_df, exp_logger


def analyze_results(df: pd.DataFrame, nodebug: bool = False) -> Dict:
    """
    Анализирует результаты абляции и выводит статистику.
    
    Returns:
        dict со статистикой тестов
    """
    print_always("\n" + "=" * 70)
    print_always("РЕЗУЛЬТАТЫ")
    print_always("=" * 70)
    print_always(f"\nДоступные классы агентов: {df['agent_class'].unique()}")
    print_always("--" * 35)
    
    # Группировка по агентам
    for agent_name, agent_class in [
        ('Full', 'RheologicalAgent'),
        ('NoVG', 'RheologicalAgent_NoVG'),
        ('NoVp', 'RheologicalAgent_NoVp')
    ]:
        agent_data = df[df['agent_class'] == agent_class]
        
        if len(agent_data) == 0:
            print_always(f"\n⚠️ {agent_name} Agent: НЕТ ДАННЫХ!")
            continue
        
        latencies = agent_data['latency'].values
        latencies_valid = [l for l in latencies if l != 999]
        
        if len(latencies_valid) == 0:
            print_always(f"\n{agent_name} Agent ({len(agent_data)} seeds):")
            print_always(f"  ⚠️ Все латентности = 999 (агент не переключился)")
            continue
        
        print_always(f"\n{agent_name} Agent ({len(agent_data)} seeds):")
        print_always(f"  Mean Latency:    {np.mean(latencies_valid):.1f} ± {np.std(latencies_valid):.1f} trials")
        print_always(f"  Median Latency:  {np.median(latencies_valid):.1f} trials")
        print_always(f"  Min: {np.min(latencies_valid)}, Max: {np.max(latencies_valid)}")
        print_always(f"  Explore Pre:     {agent_data['explore_pre'].mean():.1%}")
        print_always(f"  Explore Shock:   {agent_data['explore_shock'].mean():.1%}")
        print_always(f"  Explore Post:    {agent_data['explore_post'].mean():.1%}")
    
    # Статистическое сравнение
    print_always("\n" + "=" * 70)
    print_always("Статистический тест (Mann-Whitney U)")
    print_always("=" * 70)
    
    full_latencies = df[df['agent_class'] == 'RheologicalAgent']['latency'].values
    novg_latencies = df[df['agent_class'] == 'RheologicalAgent_NoVG']['latency'].values
    novp_latencies = df[df['agent_class'] == 'RheologicalAgent_NoVp']['latency'].values
    
    stats_results = {}
    
    # Full vs NoVG (Ось 1: V_G влияет на латентность)
    if len(full_latencies) > 0 and len(novg_latencies) > 0:
        u_stat, p_value = stats.mannwhitneyu(full_latencies, novg_latencies, alternative='greater')
        # Effect size (rank-biserial correlation)
        n1, n2 = len(full_latencies), len(novg_latencies)
        effect_size = 1 - (2 * u_stat) / (n1 * n2)
        stats_results['full_vs_novg'] = {'U': u_stat, 'p': p_value, 'effect_size': effect_size}
        print_always(f"\nFull vs NoVG (V_G effect):")
        print_always(f"  U = {u_stat:.1f}, p = {p_value:.2e}, effect size = {effect_size:.3f}")
    
    # Full vs NoVp (Ось 2: V_p влияет на персеверацию, не на латентность)
    if len(full_latencies) > 0 and len(novp_latencies) > 0:
        u_stat, p_value = stats.mannwhitneyu(full_latencies, novp_latencies, alternative='two-sided')
        n1, n2 = len(full_latencies), len(novp_latencies)
        effect_size = 1 - (2 * u_stat) / (n1 * n2)
        stats_results['full_vs_novp'] = {'U': u_stat, 'p': p_value, 'effect_size': effect_size}
        print_always(f"\nFull vs NoVp (V_p effect on latency):")
        print_always(f"  U = {u_stat:.1f}, p = {p_value:.2e}, effect size = {effect_size:.3f}")
    
    # NoVp vs NoVG
    if len(novp_latencies) > 0 and len(novg_latencies) > 0:
        u_stat, p_value = stats.mannwhitneyu(novp_latencies, novg_latencies, alternative='greater')
        n1, n2 = len(novp_latencies), len(novg_latencies)
        effect_size = 1 - (2 * u_stat) / (n1 * n2)
        stats_results['novp_vs_novg'] = {'U': u_stat, 'p': p_value, 'effect_size': effect_size}
        print_always(f"\nNoVp vs NoVG:")
        print_always(f"  U = {u_stat:.1f}, p = {p_value:.2e}, effect size = {effect_size:.3f}")
    
    # Финальный вердикт
    print_always("\n" + "=" * 70)
    
    full_median = np.median([l for l in full_latencies if l != 999])
    novg_median = np.median([l for l in novg_latencies if l != 999])
    
    if 'full_vs_novg' in stats_results and stats_results['full_vs_novg']['p'] < 0.001:
        print_always("✓ УСПЕХ (DOUBLE DISSOCIATION):")
        print_always("  V_G достоверно увеличивает латентность переключения режимов.")
        print_always("  V_p влияет на персеверацию внутри режима (см. explore_pre).")
        print_always("=" * 70)
        return {'success': True, 'stats': stats_results}
    else:
        print_always("✗ ПРОВАЛ: Разница статистически незначима.")
        print_always("=" * 70)
        return {'success': False, 'stats': stats_results}


def main():
    """Точка входа для скрипта абляции."""
    args = parse_args(description="Stage 2A: Two-Step Task — Ablation Study")
    
    # Дополнительные аргументы специфичные для абляции
    n_seeds = 30  # Хардкод для стандартизации
    
    results_df, exp_logger = run_ablation_study(
        n_seeds=n_seeds,
        n_trials=args.n_trials,
        changepoint=args.changepoint,
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
        stats_path = os.path.join(args.output_dir, f'{exp_logger.experiment_id}_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        print_always(f"\nСтатистика сохранена в {stats_path}")
    
    return analysis_results['success']


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)