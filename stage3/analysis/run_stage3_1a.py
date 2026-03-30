"""
Stage 3.1A: Run Full Experiments.

Запускает 30 seeds с полным логированием.
Структура как в Stage 2: одна папка на запуск, внутри все seeds.

Usage:
    python -m stage3.analysis.run_stage3_1a --n-seeds 30 --n-trials 100
"""

import argparse
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import sys

# Добавляем корень проекта в path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stage3.core.agent_stage3 import AgentStage3
from stage3.envs.open_covered_choice_env import OpenCoveredChoiceEnv
from stage3.configs.config_stage3_1a import CONFIG_3_1A


def run_experiment(seed: int, n_trials: int, output_dir: str, ablation: str = 'full'):
    """
    Запускает один seed с корректным RNG seeding.
    
    Args:
        seed: Random seed (42-71)
        n_trials: Количество триалов
        output_dir: Директория для логов (общая для всех seeds)
        ablation: 'full', 'novg', 'novp', 'nox', 'one_shot_off'
    """
    # =====================================================================
    # КРИТИЧЕСКИ ВАЖНО: Сбрасываем RNG ПЕРЕД созданием env и agent
    # =====================================================================
    np.random.seed(seed)
    
    # Создаём среду с seed
    env_config = CONFIG_3_1A['env'].copy()
    env = OpenCoveredChoiceEnv(env_config, seed=seed)
    
    # Создаём агента
    agent_config = CONFIG_3_1A['agent'].copy()
    agent_config['compatibility_mode'] = False
    agent_config['log_level'] = 0  # Отключаем логирование в agent (env ведёт логи)
    
    # Применяем абляцию
    if ablation != 'full':
        ablation_config = CONFIG_3_1A['ablation'].get(ablation, {})
        modifications = ablation_config.get('modifications', {})
        
        if 'agent' in modifications:
            agent_updates = modifications['agent']
            
            # NoVG: нулевые temporal traces
            if ablation == 'novg' and 'temporal_state' in agent_updates:
                agent_config['temporal_state']['h_risk'] = 0.0
                agent_config['temporal_state']['h_opp'] = 0.0
            
            # NoVp: отключаем viscosity
            if ablation == 'novp' and 'viscosity' in agent_updates:
                agent_config['viscosity']['k_use'] = 0.0
                agent_config['viscosity']['k_melt'] = 0.0
            
            # NoX: нулевые exposure aggregates
            if ablation == 'nox' and 'exposure_field' in agent_updates:
                agent_config['exposure_field']['zero_output'] = True
            
            # One-shot off: высокий порог
            if ablation == 'one_shot_off' and 'temporal_state' in agent_updates:
                agent_config['temporal_state']['one_shot_threshold'] = 999.0
                agent_config['temporal_state']['one_shot_boost'] = 0.0
    
    agent = AgentStage3(agent_config)
    
    # Запускаем триалы
    for trial in range(n_trials):
        observation = env.reset(trial=trial)
        done = False
        
        while not done:
            action, metadata = agent.step(
                observation=observation,
                reward=0.0,
                action=0,
                salience=None
            )
            
            observation, reward, done, info = env.step(
                action=action,
                mode=metadata['mode'],
                gate_trigger=metadata['gate_constraint']
            )
    
    # Сохраняем логи в ОБЩУЮ папку (все seeds в одной директории)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"stage3_1a_seed{seed}_trials.csv"
    
    env.save_logs(output_dir, log_filename)  # ← Два аргумента!
    
    return len(env.trial_summaries)


def main():
    parser = argparse.ArgumentParser(description='Run Stage 3.1A Experiments')
    parser.add_argument('--n-seeds', type=int, default=30, help='Number of seeds')
    parser.add_argument('--n-trials', type=int, default=100, help='Trials per seed')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (auto-generated if None)')
    parser.add_argument('--ablation', type=str, default='full', help='Ablation type')
    
    args = parser.parse_args()
    
    # =====================================================================
    # СТРУКТУРА КАК В STAGE 2: Одна папка на запуск с timestamp
    # =====================================================================
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"logs/stage3/stage3_1a/run_{timestamp}/"
    else:
        output_dir = args.output_dir
    
    # Создаём директорию
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем метаданные
    metadata = {
        'stage': '3.1A',
        'n_seeds': args.n_seeds,
        'n_trials': args.n_trials,
        'ablation': args.ablation,
        'config': 'config_stage3_1a',
        'start_time': datetime.now().isoformat(),
        'seed_range': [42, 42 + args.n_seeds - 1],
        'output_structure': 'One folder per run, all seeds inside'
    }
    
    with open(output_path / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Запускаем эксперименты
    print(f"Starting Stage 3.1A: {args.n_seeds} seeds × {args.n_trials} trials")
    print(f"Output: {output_path}")
    print(f"Ablation: {args.ablation}")
    print("=" * 70)
    
    total_trials = 0
    for seed in range(42, 42 + args.n_seeds):
        # Сбрасываем RNG для каждого seed
        np.random.seed(seed)
        
        n_trials_completed = run_experiment(
            seed=seed,
            n_trials=args.n_trials,
            output_dir=str(output_path),
            ablation=args.ablation
        )
        
        total_trials += n_trials_completed
        print(f"  Seed {seed}: {n_trials_completed} trials completed")
    
    # Завершаем метаданные
    metadata['end_time'] = datetime.now().isoformat()
    metadata['total_trials'] = total_trials
    metadata['files_created'] = len(list(output_path.glob('*.csv')))
    
    with open(output_path / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print("=" * 70)
    print(f"✓ Stage 3.1A completed: {args.n_seeds} seeds, {total_trials} total trials")
    print(f"Logs saved to: {output_path}")
    print(f"Files created: {metadata['files_created']}")


if __name__ == "__main__":
    main()