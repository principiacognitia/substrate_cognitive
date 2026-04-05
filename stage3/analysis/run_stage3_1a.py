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
import pandas as pd
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
    
    agent = AgentStage3(agent_config, seed=seed)  # ← Передать seed
    
    # Запускаем триалы
    for trial in range(n_trials):
        observation = env.reset(trial=trial)
        done = False
        step_count = 0  # ← Защитный лимит для отладки
        
        while not done and step_count < 100:
            # === PRE ===
            if env.debug:
                print(f"[PRE]  t={env.state.tick} node={env.state.current_node} "
                      f"state={env.state.deliberation_state.value} "
                      f"q={observation.get('q_values', [0.0])}")

            action, metadata = agent.step(
                observation=observation,
                reward=0.0,
                action=0,
                salience=None
            )

            # === AGENT ===
            if env.debug:
                print(f"[AGENT] a={action} mode={metadata['mode']} "
                      f"probs={metadata.get('action_probs', [])}")

            observation, reward, done, info = env.step(
                action=action,
                mode=metadata['mode'],
                gate_trigger=metadata.get('gate_constraint', 'default'),  # ← Безопасный fallback
                action_probs=metadata.get('action_probs', [0.5, 0.5])
            )

            # === POST ===
            if env.debug:
                print(f"[POST] node={env.state.current_node} "
                      f"state={env.state.deliberation_state.value} "
                      f"committed={env.state.committed_path}\n")

            step_count += 1
    
    # Сохраняем логи в ОБЩУЮ папку (все seeds в одной директории)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"stage3_1a_seed{seed}_trials.csv"
    
    env.save_logs(output_dir, log_filename)  # ← Два аргумента!
    
    return len(env.trial_summaries)

def summarize_run(output_dir: str) -> None:
    """
    Собирает все seed CSV из output_dir и сохраняет:
    - all_trials_combined.csv
    - seed_summary.csv
    - run_summary.json
    """
    output_path = Path(output_dir)
    csv_files = sorted(output_path.glob("stage3_1a_seed*_trials.csv"))

    if not csv_files:
        print("  ⚠ No trial CSV files found for summary.")
        return

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df["source_file"] = f.name
        dfs.append(df)

    all_trials = pd.concat(dfs, ignore_index=True)
    all_trials.to_csv(output_path / "all_trials_combined.csv", index=False)

    # --- Per-seed summary ---
    def _count_value(series, value):
        return int((series == value).sum())

    seed_summary = (
        all_trials
        .groupby("seed", dropna=False)
        .apply(lambda g: pd.Series({
            "n_trials": len(g),
            "n_open": _count_value(g["path_choice"], "open"),
            "n_covered": _count_value(g["path_choice"], "covered"),
            "p_open": float((g["path_choice"] == "open").mean()),
            "p_covered": float((g["path_choice"] == "covered").mean()),
            "mean_reward_total": float(g["reward_total"].mean()),
            "mean_commit_latency": float(g["commit_latency"].mean()),
            "mean_junction_pause_duration": float(g["junction_pause_duration"].mean()),
            "mean_reorientation_count": float(g["reorientation_count"].mean()),
            "p_explore_at_junction": float((g["mode_at_junction"] == "explore").mean()),
            "n_commit_bound": _count_value(g["commit_reason"], "bound"),
            "n_commit_timeout": _count_value(g["commit_reason"], "timeout"),
            "p_commit_bound": float((g["commit_reason"] == "bound").mean()),
            "p_commit_timeout": float((g["commit_reason"] == "timeout").mean()),
        }))
        .reset_index()
        .sort_values("seed")
    )

    seed_summary.to_csv(output_path / "seed_summary.csv", index=False)

    # --- Overall run summary ---
    run_summary = {
        "n_seeds": int(seed_summary["seed"].nunique()),
        "n_trials_total": int(len(all_trials)),
        "n_open_total": int((all_trials["path_choice"] == "open").sum()),
        "n_covered_total": int((all_trials["path_choice"] == "covered").sum()),
        "p_open_total": float((all_trials["path_choice"] == "open").mean()),
        "p_covered_total": float((all_trials["path_choice"] == "covered").mean()),
        "mean_reward_total": float(all_trials["reward_total"].mean()),
        "mean_commit_latency": float(all_trials["commit_latency"].mean()),
        "mean_junction_pause_duration": float(all_trials["junction_pause_duration"].mean()),
        "mean_reorientation_count": float(all_trials["reorientation_count"].mean()),
        "p_explore_at_junction": float((all_trials["mode_at_junction"] == "explore").mean()),
        "n_commit_bound_total": int((all_trials["commit_reason"] == "bound").sum()),
        "n_commit_timeout_total": int((all_trials["commit_reason"] == "timeout").sum()),
        "p_commit_bound_total": float((all_trials["commit_reason"] == "bound").mean()),
        "p_commit_timeout_total": float((all_trials["commit_reason"] == "timeout").mean()),
        "covered_rate_mean_across_seeds": float(seed_summary["p_covered"].mean()),
        "covered_rate_std_across_seeds": float(seed_summary["p_covered"].std(ddof=0)),
        "commit_latency_mean_across_seeds": float(seed_summary["mean_commit_latency"].mean()),
    }

    with open(output_path / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)

    print("  Summary saved:")
    print(f"    - {output_path / 'all_trials_combined.csv'}")
    print(f"    - {output_path / 'seed_summary.csv'}")
    print(f"    - {output_path / 'run_summary.json'}")

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
    
    summarize_run(str(output_path))
    
    print("=" * 70)
    print(f"✓ Stage 3.1A completed: {args.n_seeds} seeds, {total_trials} total trials")
    print(f"Logs saved to: {output_path}")
    print(f"Files created: {metadata['files_created']}")


if __name__ == "__main__":
    main()