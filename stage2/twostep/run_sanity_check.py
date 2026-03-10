"""
Stage 2A: Two-Step Task — Sanity Check (MB/MF Signatures).
Валидация MB/MF сигнатур на seed-level (не trial-level!).

Запуск:
    python -m stage2.twostep.run_sanity_check --n-seeds 30 --n-trials 2000
    python -m stage2.twostep.run_sanity_check --no-log --nodebug

Выходные данные:
    logs/twostep/{experiment_id}_trials.csv — полные логи триалов
    logs/twostep/{experiment_id}_sanity_results.csv — коэффициенты регрессии по seeds
    logs/twostep/{experiment_id}_meta.json — метаданные эксперимента
"""

import os
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from stage2.twostep.env_twostep import TwoStepEnv
from stage2.core.baselines import MFAgent, MBAgent
from stage2.core.args import parse_args, print_debug, print_always
from stage2.core.logger import TrialLogger, ExperimentLogger


def run_agent(env: TwoStepEnv, agent, n_trials: int = 2000, 
              logger: TrialLogger = None, verbose: bool = False, 
              nodebug: bool = False) -> pd.DataFrame:
    """
    Запускает агента и собирает данные для регрессии.
    
    КРИТИЧНО: Данные собираются с лагом (t-1 → t), как в Daw et al. (2011).
    """
    data = []
    prev_a1 = None
    prev_reward = None
    prev_trans_factor = None
    
    for trial in range(n_trials):
        s1 = env.reset()
        a1 = agent.select_action_stage1(s1)
        s2, trans_type = env.step_stage1(a1)
        a2 = agent.select_action_stage2(s2)
        reward, done, info = env.step_stage2(a2)
        agent.update(a1, a2, reward, s2, trans_type, s1)
        
        trans_factor = 1 if trans_type == 'common' else -1
        
        # Записываем данные только начиная со второго триала (есть предыдущий)
        if prev_a1 is not None:
            stay = 1 if (a1 == prev_a1) else 0
            data.append({
                'trial': trial,
                'reward': prev_reward,  # Награда из t-1
                'trans_factor': prev_trans_factor,  # Переход из t-1
                'stay': stay  # Выбор в t совпал с выбором в t-1
            })
            
            # Логирование
            if logger:
                logger.log_trial(
                    trial=trial,
                    seed=env.rng.integers(0, 1000000),  # Approximate
                    s1=s1, a1=a1, s2=s2, a2=a2,
                    trans_type=trans_type,
                    reward=reward,
                    stay=stay
                )
        
        # Сохраняем для следующего триала
        prev_a1 = a1
        prev_reward = reward
        prev_trans_factor = trans_factor
        
        # Отладочный вывод
        if verbose and not nodebug:
            if trial % 500 == 0:
                print(f"  Trial {trial}...")
    
    df = pd.DataFrame(data)
    df['interaction'] = df['reward'] * df['trans_factor']
    return df


def logistic_regression(df: pd.DataFrame):
    """
    Stay ~ reward + trans_factor + interaction
    """
    if len(df) < 10:
        return None
    
    X = sm.add_constant(df[['reward', 'trans_factor', 'interaction']])
    model = sm.Logit(df['stay'], X)
    try:
        result = model.fit(disp=0, maxiter=100)
    except:
        return None
    return result


def run_sanity_study(
    n_seeds: int = 30,
    n_trials: int = 2000,
    alpha: float = 0.35,
    beta: float = 4.0,
    output_dir: str = 'logs/twostep/',
    log_trials: bool = True,
    experiment_id: str = None,
    verbose: bool = False,
    nodebug: bool = False
) -> Tuple[pd.DataFrame, ExperimentLogger]:
    """
    Запускает sanity check по всем seeds для MF и MB агентов.
    
    Returns:
        DataFrame с коэффициентами регрессии по seeds
        ExperimentLogger для сохранения метаданных
    """
    # Создаём логгер эксперимента
    if experiment_id is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_id = f'twostep_sanity_{timestamp}'
    
    exp_logger = ExperimentLogger(
        experiment_name=experiment_id,
        log_dir=output_dir,
        config={
            'n_seeds': n_seeds,
            'n_trials': n_trials,
            'alpha': alpha,
            'beta': beta
        },
        description='Two-Step Task Sanity Check (MB/MF Signatures)'
    )
    
    all_results = []
    
    print_always("=" * 70)
    print_always("Stage 2A: Two-Step Task — Sanity Check (MB/MF Signatures)")
    print_always("=" * 70)
    print_always(f"Seeds: {n_seeds}, Trials: {n_trials}")
    print_always(f"Alpha: {alpha}, Beta: {beta}")
    print_always(f"Logging: {'enabled' if log_trials else 'disabled'}")
    print_always("--" * 35)
    
    for agent_name, agent_class in [('MF', MFAgent), ('MB', MBAgent)]:
        print_always(f"\nЗапуск {agent_name}-only агента...")
        
        for i in range(n_seeds):
            seed = 42 + i
            
            # Создаём trial-логгер для этого seed
            trial_logger = None
            if log_trials and agent_name == 'MF':  # Логируем только MF для экономии места
                trial_logger = exp_logger.get_trial_logger(suffix=f'{agent_name}_seed{seed}')
            
            if verbose and not nodebug:
                print(f"  Seed {seed}...")
            
            env = TwoStepEnv(seed=seed, n_trials=n_trials, with_changepoint=False)
            agent = agent_class(alpha=alpha, beta=beta, seed=seed)
            
            df = run_agent(env, agent, n_trials=n_trials, 
                          logger=trial_logger, verbose=verbose, nodebug=nodebug)
            
            result = logistic_regression(df)
            
            if result is not None:
                all_results.append({
                    'agent_type': agent_name,
                    'seed': seed,
                    'coef_reward': result.params['reward'],
                    'p_reward': result.pvalues['reward'],
                    'coef_trans': result.params['trans_factor'],
                    'p_trans': result.pvalues['trans_factor'],
                    'coef_interaction': result.params['interaction'],
                    'p_interaction': result.pvalues['interaction'],
                    'n_trials': len(df)
                })
            
            # Прогресс
            if (i + 1) % 10 == 0 and not nodebug:
                print_always(f"  Прогресс {agent_name}: {i + 1}/{n_seeds}")
    
    results_df = pd.DataFrame(all_results)
    
    # Сохраняем агрегированные результаты
    if log_trials:
        exp_logger.save_results(results_df, 'sanity_results.csv')
        exp_logger.finalize()
    
    return results_df, exp_logger


def analyze_results(df: pd.DataFrame, nodebug: bool = False) -> Dict:
    """
    Анализирует результаты sanity check.
    """
    print_always("\n" + "=" * 70)
    print_always("РЕЗУЛЬТАТЫ (Seed-level регрессия)")
    print_always("=" * 70)
    
    stats_results = {}
    
    for agent_name in ['MF', 'MB']:
        agent_data = df[df['agent_type'] == agent_name]
        
        if len(agent_data) == 0:
            print_always(f"\n⚠️ {agent_name} Agent: НЕТ ДАННЫХ!")
            continue
        
        print_always(f"\n{agent_name} Agent ({len(agent_data)} seeds):")
        print_always(f"  Reward coef:     {agent_data['coef_reward'].mean():.4f} ± {agent_data['coef_reward'].std():.4f}")
        print_always(f"  Interaction coef: {agent_data['coef_interaction'].mean():.4f} ± {agent_data['coef_interaction'].std():.4f}")
        print_always(f"  Interaction p:   {agent_data['p_interaction'].median():.4f} (median)")
        
        # Валидация критериев
        if agent_name == 'MF':
            passed_reward = agent_data['coef_reward'].mean() > 0.1
            passed_interaction = agent_data['p_interaction'].median() > 0.10
            stats_results['mf'] = {
                'passed': passed_reward and passed_interaction,
                'coef_reward': agent_data['coef_reward'].mean(),
                'coef_interaction': agent_data['coef_interaction'].mean(),
                'p_interaction': agent_data['p_interaction'].median()
            }
        else:  # MB
            passed_interaction = agent_data['p_interaction'].median() < 0.01
            passed_coef = agent_data['coef_interaction'].mean() > 0.2
            stats_results['mb'] = {
                'passed': passed_interaction and passed_coef,
                'coef_reward': agent_data['coef_reward'].mean(),
                'coef_interaction': agent_data['coef_interaction'].mean(),
                'p_interaction': agent_data['p_interaction'].median()
            }
    
    # Финальный вердикт
    print_always("\n" + "=" * 70)
    print_always("Валидация критериев")
    print_always("=" * 70)
    
    mf_ok = stats_results.get('mf', {}).get('passed', False)
    mb_ok = stats_results.get('mb', {}).get('passed', False)
    
    if mf_ok:
        print_always("✓ MF: reward coef > 0.1, interaction p > 0.10")
    else:
        print_always("✗ MF: критерии не выполнены")
    
    if mb_ok:
        print_always("✓ MB: interaction p < 0.01, coef > 0.2")
    else:
        print_always("✗ MB: критерии не выполнены")
    
    print_always("=" * 70)
    
    if mf_ok and mb_ok:
        print_always("✓ УСПЕХ: MB/MF сигнатуры различимы")
        print_always("=" * 70)
        return {'success': True, 'stats': stats_results}
    else:
        print_always("✗ ПРОВАЛ: MB/MF сигнатуры не различимы")
        print_always("=" * 70)
        return {'success': False, 'stats': stats_results}


def main():
    """Точка входа для скрипта sanity check."""
    args = parse_args(
        description="Stage 2A: Two-Step Task — Sanity Check",
        default_output_dir='logs/twostep/'
    )
    
    results_df, exp_logger = run_sanity_study(
        n_seeds=args.n_seeds,
        n_trials=args.n_trials,
        alpha=args.alpha,
        beta=args.beta,
        output_dir=args.output_dir,
        log_trials=not args.no_log,
        experiment_id=args.experiment_id,
        verbose=args.verbose,
        nodebug=args.nodebug
    )
    
    analysis_results = analyze_results(results_df, nodebug=args.nodebug)
    
    # Сохраняем статистику в JSON
    if not args.no_log:
        stats_path = os.path.join(args.output_dir, f'{exp_logger.experiment_name}_sanity_stats.json')

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