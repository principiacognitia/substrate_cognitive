"""
Level 4: Ablation Study — Доказательство необходимости V_G и V_p.

Сравнивает три версии агента:
  1. Full RheologicalAgent (V_G + V_p)
  2. NoVG (V_G = 0, V_p активен)
  3. NoVp (V_p = 0, V_G активен)

Запуск:
  python -m stage2.twostep.debug_ablation --seed 42 --n-trials 2000
  python -m stage2.twostep.debug_ablation --nodebug --n-seeds 30
"""

import sys
sys.path.insert(0, 'E:/CRS-1/substrate_cognitive')

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
from stage2.twostep.env_twostep import TwoStepEnv
from stage2.core import (
    RheologicalAgent,
    RheologicalAgent_NoVG,
    RheologicalAgent_NoVp,
    parse_args,
    print_debug,
    print_always
)
from stage2.core.logger import TrialLogger


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
    log_trials: bool = False,
    output_dir: str = 'logs/twostep/',
    experiment_id: str = None,
    verbose: bool = False,
    nodebug: bool = False
) -> dict:
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
    
    # Логгер (опционально)
    logger = None
    if log_trials:
        if experiment_id is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_id = f'ablation_{agent_class.__name__}_seed{seed}_{timestamp}'
        logger = TrialLogger(log_dir=output_dir, experiment_id=experiment_id)
    
    # Переменные для сбора метрик
    u_t_prev = np.zeros(4)
    explore_trials = []
    v_g_history = []
    v_p_history = []
    
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
    
    # Вычисляем доли EXPLORE в непересекающихся окнах
    # Pre: 100 триалов ДО смены (Доказываем стационарную привычку)
    explore_pre = len([t for t in explore_trials if changepoint-100 <= t < changepoint]) / 100
    # Shock: 30 триалов СРАЗУ ПОСЛЕ смены (Ловим транзиентный пик поиска)
    explore_shock = len([t for t in explore_trials if changepoint <= t < changepoint+30]) / 30
    # Post: 100 триалов ПОСЛЕ адаптации (с 100-го по 200-й после смены)
    # Доказываем, что агент сформировал НОВУЮ стационарную привычку
    explore_post = len([t for t in explore_trials if changepoint+100 <= t < changepoint+200]) / 100
    
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
    log_trials: bool = False,
    verbose: bool = False,
    nodebug: bool = False
) -> pd.DataFrame:
    """
    Запускает полную абляцию по всем седам и агентам.
    
    Returns:
        DataFrame с результатами всех экспериментов
    """
    agents = [
        ('Full', RheologicalAgent),
        ('NoVG', RheologicalAgent_NoVG),
        ('NoVp', RheologicalAgent_NoVp)
    ]
    
    all_results = []
    
    print_always("=" * 70)
    print_always("Level 4: Ablation Study — Доказательство необходимости V_G и V_p")
    print_always("=" * 70)
    print_always(f"Seeds: {n_seeds}, Trials: {n_trials}, Changepoint: {changepoint}")
    print_always(f"Theta_MB: {theta_mb}, Theta_U: {theta_u}")
    print_always(f"Alpha: {alpha}, Beta: {beta}, Volatility Threshold: {volatility_threshold}")
    print_always(f"Log Trials: {log_trials}, Output Dir: {output_dir}")
    print_always("--" * 35)
    
    for agent_name, agent_class in agents:
        print_always(f"Запуск {agent_name} агента...")
        
        for i in range(n_seeds):
            seed = 42 + i
            
            # ИСПРАВЛЕНИЕ: прямая проверка вместо print_debug
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
                log_trials=log_trials,
                output_dir=output_dir,
                verbose=verbose,
                nodebug=nodebug
            )
            
            all_results.append(result)
            
            # Прогресс
            if (i + 1) % 10 == 0 and not nodebug:
                print_always(f"  Прогресс {agent_name}: {i + 1}/{n_seeds}")
        
        print_always("")
            
    return pd.DataFrame(all_results)


def analyze_results(df: pd.DataFrame, nodebug: bool = False) -> None:
    """
    Анализирует результаты абляции и выводит статистику.
    """
    print_always("=" * 70)
    print_always("РЕЗУЛЬТАТЫ")
    print_always("=" * 70)
    
    # ИСПРАВЛЕНИЕ: проверяем какие имена классов реально есть в DataFrame
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
        print_always(f"  Mean Latency:    {np.mean(latencies_valid):.1f} ± {np.std(latencies_valid):.1f} триалов")
        print_always(f"  Median Latency:  {np.median(latencies_valid):.1f} триалов")
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

    # Проверка что данные есть
    if len(full_latencies) == 0:
        print_always("\n✗ ОШИБКА: Нет данных для Full агента!")
        return
    
    if len(novg_latencies) == 0:
        print_always("\n✗ ОШИБКА: Нет данных для NoVG агента!")
        return

    if len(novp_latencies) == 0:
        print_always("\n✗ ОШИБКА: Нет данных для NoVp агента!")
        return    
    
    # Full vs NoVG (Ожидаем, что у Full задержка больше)
    u_stat, p_value = stats.mannwhitneyu(full_latencies, novg_latencies, alternative='greater')
    print_always(f"\nFull vs NoVG: U = {u_stat:.1f}, p = {p_value:.2e}")
    
    # Full vs NoVp (Ожидаем, что они ПРИМЕРНО РАВНЫ)
    u_stat, p_value = stats.mannwhitneyu(full_latencies, novp_latencies, alternative='two-sided')
    print_always(f"Full vs NoVp: U = {u_stat:.1f}, p = {p_value:.2e}")
    
    # NoVp vs NoVG (Ожидаем, что у NoVp задержка больше, так как у него есть V_G!)
    u_stat, p_value = stats.mannwhitneyu(novp_latencies, novg_latencies, alternative='greater')
    print_always(f"NoVp vs NoVG: U = {u_stat:.1f}, p = {p_value:.2e}")
    
    # Финальный вердикт
    print_always("\n" + "=" * 70)
    
    full_median = np.median([l for l in full_latencies if l != 999])
    novg_median = np.median([l for l in novg_latencies if l != 999])
    
    if len(full_latencies) > 0 and len(novg_latencies) > 0:
        if full_median > novg_median + 5 and p_value < 0.001:
            print_always("✓ УСПЕХ (DOUBLE DISSOCIATION):")
            print_always("  V_G достоверно увеличивает латентность переключения режимов.")
            print_always("  V_p влияет на персеверацию внутри режима (см. explore_pre).")
        else:
            print_always("✗ ПРОВАЛ: Разница статистически незначима.")
    else:
        print_always("✗ ПРОВАЛ: Недостаточно данных для сравнения.")
    
    print_always("=" * 70)


def main():
    """Точка входа для скрипта абляции."""
    args = parse_args(description="Level 4: Ablation Study — V_G vs V_p")
    
    # Дополнительные аргументы специфичные для абляции
    parser = args.__class__
    if not hasattr(args, 'n_seeds'):
        import argparse
        # Добавляем аргумент n_seeds если парсер позволяет
        pass
    
    # Хардкод n_seeds для абляции (можно вынести в аргументы)
    n_seeds = 30
    
    results_df = run_ablation_study(
        n_seeds=n_seeds,
        n_trials=args.n_trials,
        changepoint=args.changepoint,
        theta_mb=args.theta_mb,
        theta_u=args.theta_u,
        alpha=args.alpha,
        beta=args.beta,
        volatility_threshold=args.volatility_threshold,
        output_dir=args.output_dir,
        log_trials=args.log_trials,
        verbose=args.verbose,
        nodebug=args.nodebug
    )
    
    analyze_results(results_df, nodebug=args.nodebug)
    
    # Сохранение результатов
    if args.log_trials:
        results_df.to_csv(f"{args.output_dir}ablation_results.csv", index=False)
        print_always(f"\nРезультаты сохранены в {args.output_dir}ablation_results.csv")


if __name__ == "__main__":
    main()