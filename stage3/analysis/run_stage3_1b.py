"""
Stage 3.1B Runner: Threat / Reward Conflict.

Запускает эксперименты Stage 3.1B с поддержкой:
- Одиночных условий (canonical conditions)
- Полной 3x3 матрицы
- One-shot protocol

CLI Usage:
    python -m stage3.analysis.run_stage3_1b --n-seeds 50 --n-trials 100 --condition balanced
    python -m stage3.analysis.run_stage3_1b --grid 3x3 --n-seeds 50 --n-trials 100
    python -m stage3.analysis.run_stage3_1b --one-shot --n-seeds 50

Author: Alex Snow (Aleksey L. Snigirov)
License: MIT
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stage3.configs.config_stage3_1b import (
    CONFIG_3_1B,
    AGENT_CONFIG_3_1B,
    LOGGING_CONFIG_3_1B,
    ONE_SHOT_DISABLED,
    get_condition_grid,
    get_canonical_conditions,
    get_one_shot_protocol,
    build_env_config_for_condition,
)
from stage3.envs.open_covered_choice_env import OpenCoveredChoiceEnv
from stage3.core.agent_stage3 import AgentStage3


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Run Stage 3.1B experiments')
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--condition', type=str,
        choices=['balanced', 'balanced_conflict', 'reward_dominant', 'threat_dominant'],
        help='Run single canonical condition'
    )
    mode_group.add_argument('--grid', type=str, 
                           choices=['3x3'],
                           help='Run full 3x3 matrix')
    mode_group.add_argument('--one-shot', action='store_true',
                           help='Run one-shot protocol')
    
    # Common parameters
    parser.add_argument('--n-seeds', type=int, default=50,
                       help='Number of seeds (default: 50)')
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of trials per seed (default: 100)')
    
    # Output directory
    parser.add_argument('--output-dir', type=str, default='logs/stage3/stage3_1b',
        help='Base output directory; run-specific timestamped subdir will be created inside it')
    
    parser.add_argument('--ablation', type=str, default='full',
                       choices=['full', 'novg', 'novp', 'nox', 'one_shot_off'],
                       help='Ablation condition (default: full)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    return parser.parse_args()

def normalize_condition_name(condition_name: str) -> str:
    """
    Нормализует CLI aliases для canonical conditions.
    """
    aliases = {
        'balanced': 'balanced_conflict',
        'balanced_conflict': 'balanced_conflict',
        'reward_dominant': 'reward_dominant',
        'threat_dominant': 'threat_dominant',
    }

    if condition_name not in aliases:
        raise ValueError(f"Unknown condition name: {condition_name}")

    return aliases[condition_name]

def build_timestamped_output_dir(base_output_dir: str, run_label: str) -> str:
    """
    Creates timestamped output directory inside base_output_dir.

    Example:
        base_output_dir = logs/stage3/stage3_1b
        run_label = balanced_conflict_full
        -> logs/stage3/stage3_1b/balanced_conflict_full_20260408_123456
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_output_dir) / f"{run_label}_{timestamp}"
    return str(output_dir)

def apply_ablation(agent_config: Dict[str, Any], ablation_name: str) -> Dict[str, Any]:
    """Applies ablation modifications to agent config."""
    from stage3.configs.config_stage3_1b import ABLATION_CONFIG_3_1B
    
    if ablation_name == 'full':
        return agent_config
    
    ablation = ABLATION_CONFIG_3_1B.get(ablation_name, {})
    modifications = ablation.get('modifications', {})
    
    # Deep copy config
    modified_config = json.loads(json.dumps(agent_config))
    
    # Apply modifications
    if 'agent' in modifications:
        agent_mods = modifications['agent']
        
        # NoVG: zero out temporal state
        if 'temporal_state' in agent_mods:
            ts_mods = agent_mods['temporal_state']
            if 'h_risk' in ts_mods:
                modified_config['temporal_state']['h_risk'] = ts_mods['h_risk']
            if 'h_opp' in ts_mods:
                modified_config['temporal_state']['h_opp'] = ts_mods['h_opp']
        
        # NoVp: zero out viscosity
        if 'viscosity' in agent_mods:
            vis_mods = agent_mods['viscosity']
            if 'k_use' in vis_mods:
                modified_config.setdefault('viscosity', {})['k_use'] = vis_mods['k_use']
            if 'k_melt' in vis_mods:
                modified_config.setdefault('viscosity', {})['k_melt'] = vis_mods['k_melt']
        
        # NoX: zero out exposure field output
        if 'exposure_field' in agent_mods:
            ef_mods = agent_mods['exposure_field']
            if 'zero_output' in ef_mods:
                modified_config.setdefault('exposure_field', {})['zero_output'] = ef_mods['zero_output']
        
        # One-shot off
        if 'temporal_state' in agent_mods:
            ts_mods = agent_mods['temporal_state']
            if 'one_shot_threshold' in ts_mods:
                modified_config['temporal_state']['one_shot_threshold'] = ts_mods['one_shot_threshold']
            if 'one_shot_boost' in ts_mods:
                modified_config['temporal_state']['one_shot_boost'] = ts_mods['one_shot_boost']
    
    return modified_config


def run_condition(
    seed: int,
    condition_id: str,
    n_trials: int,
    ablation: str = 'full',
    one_shot_override: Optional[Dict] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Runs a single condition for one seed.

    ВАЖНО:
    - AgentStage3 НЕ reset'ится между trial, чтобы сохранить TemporalState persistence
    - one-shot выключен по умолчанию и включается только через one_shot_override
    """
    # Build environment config
    env_config = build_env_config_for_condition(
        condition_id=condition_id,
        one_shot_override=one_shot_override
    )
    env_config['n_trials'] = n_trials

    # Ablation-specific handling for one-shot
    if ablation == 'one_shot_off':
        env_config['one_shot'] = json.loads(json.dumps(ONE_SHOT_DISABLED))

    # Apply ablation to agent config
    agent_config = apply_ablation(json.loads(json.dumps(AGENT_CONFIG_3_1B)), ablation)

    # Create env and agent
    env = OpenCoveredChoiceEnv(env_config, seed=seed)
    agent = AgentStage3(agent_config, seed=seed)

    trial_summaries = []

    # IMPORTANT: reward feedback from previous step
    prev_reward = 0.0

    for trial in range(1, n_trials + 1):
        obs = env.reset(trial=trial)

        # НЕ вызываем agent.reset() здесь:
        # Stage 3.1B должен позволять persistence через TemporalState

        done = False
        tick = 0
        max_ticks = 100

        while not done and tick < max_ticks:
            tick += 1

            action, metadata = agent.step(
                observation=obs,
                reward=prev_reward
            )

            mode = metadata.get('mode', 'EXPLOIT')
            gate_trigger = metadata.get('gate_constraint', 'default')
            action_probs = metadata.get('action_probs', [0.5, 0.5])

            obs, reward, done, info = env.step(
                action=action,
                mode=mode,
                gate_trigger=gate_trigger,
                action_probs=action_probs
            )

            prev_reward = reward

        if not done:
            raise RuntimeError(
                f"Trial did not terminate within {max_ticks} ticks: "
                f"seed={seed}, trial={trial}, condition={condition_id}, "
                f"node={env.state.current_node}, state={env.state.deliberation_state.value}"
            )

        summaries = env.get_trial_summaries()
        if summaries:
            trial_summaries.append(summaries[-1])

    trial_dicts = [s.to_dict() for s in trial_summaries]

    path_choices = [t['path_choice'] for t in trial_dicts]
    p_open = sum(1 for p in path_choices if p == 'open') / len(path_choices) if path_choices else 0.0
    p_covered = sum(1 for p in path_choices if p == 'covered') / len(path_choices) if path_choices else 0.0

    result = {
        'seed': seed,
        'condition_id': condition_id,
        'ablation': ablation,
        'n_trials': n_trials,
        'trial_summaries': trial_dicts,
        'p_open': p_open,
        'p_covered': p_covered,
        'mean_junction_pause_duration': np.mean([t['junction_pause_duration'] for t in trial_dicts]) if trial_dicts else 0.0,
        'mean_commit_latency': np.mean([t['commit_latency'] for t in trial_dicts]) if trial_dicts else 0.0,
        'mean_reorientation_count': np.mean([t['reorientation_count'] for t in trial_dicts]) if trial_dicts else 0.0,
        'mean_junction_deliberation_proxy': np.mean([t['junction_deliberation_proxy'] for t in trial_dicts]) if trial_dicts else 0.0,
        'p_commit_bound': np.mean([t['commit_reason'] == 'bound' for t in trial_dicts]) if trial_dicts else 0.0,
        'p_commit_timeout': np.mean([t['commit_reason'] == 'timeout' for t in trial_dicts]) if trial_dicts else 0.0,
    }

    if verbose:
        print(
            f"  Seed {seed}: P(open)={p_open:.3f}, P(covered)={p_covered:.3f}, "
            f"pause={result['mean_junction_pause_duration']:.2f}, "
            f"latency={result['mean_commit_latency']:.2f}, "
            f"timeout={result['p_commit_timeout']:.2f}"
        )

    return result


def run_single_condition(
    condition_name: str,
    n_seeds: int,
    n_trials: int,
    ablation: str,
    output_dir: str,
    verbose: bool
) -> Dict[str, Any]:
    """Runs a single canonical condition."""
    canonical = get_canonical_conditions()
    condition_name = normalize_condition_name(condition_name)

    if condition_name not in canonical:
        raise ValueError(f"Unknown condition: {condition_name}")

    condition_id = canonical[condition_name]['condition_id']
    
    print(f"\n{'='*60}")
    print(f"Stage 3.1B: {condition_name} ({condition_id})")
    print(f"Seeds: {n_seeds}, Trials: {n_trials}, Ablation: {ablation}")
    print(f"{'='*60}")
    
    all_results = []
    
    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx
        result = run_condition(
            seed=seed,
            condition_id=condition_id,
            n_trials=n_trials,
            ablation=ablation,
            verbose=verbose
        )
        all_results.append(result)
    
    # Aggregate results
    return aggregate_and_save(all_results, output_dir, f"{condition_name}_{ablation}")


def run_grid(
    grid_type: str,
    n_seeds: int,
    n_trials: int,
    ablation: str,
    output_dir: str,
    verbose: bool
) -> Dict[str, Any]:
    """Runs full 3x3 matrix."""
    condition_grid = get_condition_grid()
    
    print(f"\n{'='*60}")
    print(f"Stage 3.1B: Full 3x3 Matrix")
    print(f"Conditions: {len(condition_grid)}, Seeds: {n_seeds}, Trials: {n_trials}")
    print(f"{'='*60}")
    
    all_results = {}
    
    for condition_id in condition_grid.keys():
        print(f"\n--- Condition: {condition_id} ---")
        condition_results = []
        
        for seed_idx in range(n_seeds):
            seed = 42 + seed_idx
            result = run_condition(
                seed=seed,
                condition_id=condition_id,
                n_trials=n_trials,
                ablation=ablation,
                verbose=verbose
            )
            condition_results.append(result)
        
        all_results[condition_id] = condition_results
    
    # Aggregate and save
    return aggregate_grid_and_save(all_results, output_dir, f"grid_{grid_type}_{ablation}")


def run_one_shot_protocol(
    n_seeds: int,
    n_trials: int,
    ablation: str,
    output_dir: str,
    verbose: bool
) -> Dict[str, Any]:
    """
    Runs one-shot protocol with pre/post blocks.
    """
    protocol = get_one_shot_protocol()

    print(f"\n{'='*60}")
    print(f"Stage 3.1B: One-Shot Protocol")
    print(f"Pre-block: {protocol['pre_block_trials']} trials")
    print(f"Shock trial: {protocol['shock_trial']}")
    print(f"Post-block: {protocol['post_block_trials']} trials")
    print(f"Seeds: {n_seeds}, Ablation: {ablation}")
    print(f"{'='*60}")

    # Always use protocol total_trials, not CLI n_trials
    total_trials = protocol['total_trials']
    condition_id = protocol['condition_id']

    all_results = []

    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx
        result = run_condition(
            seed=seed,
            condition_id=condition_id,
            n_trials=total_trials,
            ablation=ablation,
            one_shot_override=protocol['one_shot'],
            verbose=verbose
        )
        all_results.append(result)

    return aggregate_and_save(all_results, output_dir, f"one_shot_{ablation}")


def aggregate_and_save(
    results: List[Dict],
    output_dir: str,
    prefix: str
) -> Dict[str, Any]:
    """
    Aggregates results and saves to CSV/JSON.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_trials = []
    seed_summaries = []

    for result in results:
        seed_summaries.append({
            'seed': result['seed'],
            'condition_id': result['condition_id'],
            'ablation': result['ablation'],
            'n_trials': result['n_trials'],
            'p_open': result['p_open'],
            'p_covered': result['p_covered'],
            'mean_junction_pause_duration': result['mean_junction_pause_duration'],
            'mean_commit_latency': result['mean_commit_latency'],
            'mean_reorientation_count': result['mean_reorientation_count'],
            'mean_junction_deliberation_proxy': result['mean_junction_deliberation_proxy'],
            'p_commit_bound': result['p_commit_bound'],
            'p_commit_timeout': result['p_commit_timeout'],
        })

        for trial in result['trial_summaries']:
            all_trials.append(trial)

    if not all_trials:
        raise RuntimeError("No trials collected in aggregate_and_save()")

    df_trials = pd.DataFrame(all_trials)
    df_seeds = pd.DataFrame(seed_summaries)

    trials_file = output_path / f"{prefix}_all_trials.csv"
    seeds_file = output_path / f"{prefix}_seed_summary.csv"
    condition_file = output_path / f"{prefix}_condition_summary.csv"
    summary_file = output_path / f"{prefix}_run_summary.json"

    df_trials.to_csv(trials_file, index=False)
    df_seeds.to_csv(seeds_file, index=False)

    mode_counts = (
        df_trials['mode_at_junction']
        .value_counts(normalize=True)
        .to_dict()
        if 'mode_at_junction' in df_trials.columns else {}
    )

    condition_summary = {
        'condition_id': df_trials['condition_id'].iloc[0] if 'condition_id' in df_trials.columns else '',
        'ablation': df_trials['ablation_name'].iloc[0] if 'ablation_name' in df_trials.columns else '',
        'n_seeds': int(df_seeds['seed'].nunique()),
        'n_trials_total': int(len(df_trials)),
        'p_open': float((df_trials['path_choice'] == 'open').mean()),
        'p_covered': float((df_trials['path_choice'] == 'covered').mean()),
        'mean_junction_pause_duration': float(df_trials['junction_pause_duration'].mean()),
        'mean_commit_latency': float(df_trials['commit_latency'].mean()),
        'mean_reorientation_count': float(df_trials['reorientation_count'].mean()),
        'mean_junction_deliberation_proxy': float(df_trials['junction_deliberation_proxy'].mean()),
        'p_commit_bound': float((df_trials['commit_reason'] == 'bound').mean()),
        'p_commit_timeout': float((df_trials['commit_reason'] == 'timeout').mean()),
        'mode_at_junction_distribution': mode_counts,
    }

    pd.DataFrame([condition_summary]).to_csv(condition_file, index=False)

    run_summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_seeds': int(df_seeds['seed'].nunique()),
            'n_trials_per_seed': int(df_seeds['n_trials'].iloc[0]) if 'n_trials' in df_seeds.columns else 0,
            'ablation': condition_summary['ablation'],
        },
        'condition_summary': condition_summary,
    }

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(df_trials)} trials to {trials_file}")
    print(f"Saved {len(df_seeds)} seed summaries to {seeds_file}")
    print(f"Saved condition summary to {condition_file}")
    print(f"Saved run summary to {summary_file}")

    print(f"\n{'='*60}")
    print(f"SUMMARY: {condition_summary['condition_id']} ({condition_summary['ablation']})")
    print(f"  P(open) = {condition_summary['p_open']:.3f}")
    print(f"  P(covered) = {condition_summary['p_covered']:.3f}")
    print(f"  Junction pause = {condition_summary['mean_junction_pause_duration']:.2f}")
    print(f"  Commit latency = {condition_summary['mean_commit_latency']:.2f}")
    print(f"  Reorientation = {condition_summary['mean_reorientation_count']:.2f}")
    print(f"  Deliberation proxy = {condition_summary['mean_junction_deliberation_proxy']:.2f}")
    print(f"  P(commit timeout) = {condition_summary['p_commit_timeout']:.3f}")
    print(f"{'='*60}")

    return run_summary


def aggregate_grid_and_save(
    results: Dict[str, List[Dict]],
    output_dir: str,
    prefix: str
) -> Dict[str, Any]:
    """
    Aggregates grid results and saves to CSV/JSON.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_trials = []
    seed_summaries = []
    condition_summaries = []

    for condition_id, condition_results in results.items():
        condition_trials = []

        for result in condition_results:
            seed_summaries.append({
                'seed': result['seed'],
                'condition_id': result['condition_id'],
                'ablation': result['ablation'],
                'p_open': result['p_open'],
                'p_covered': result['p_covered'],
                'mean_junction_pause_duration': result['mean_junction_pause_duration'],
                'mean_commit_latency': result['mean_commit_latency'],
                'mean_reorientation_count': result['mean_reorientation_count'],
                'mean_junction_deliberation_proxy': result['mean_junction_deliberation_proxy'],
                'p_commit_bound': result['p_commit_bound'],
                'p_commit_timeout': result['p_commit_timeout'],
            })

            for trial in result['trial_summaries']:
                all_trials.append(trial)
                condition_trials.append(trial)

        df_condition = pd.DataFrame(condition_trials)

        mode_counts = (
            df_condition['mode_at_junction'].value_counts(normalize=True).to_dict()
            if 'mode_at_junction' in df_condition.columns else {}
        )

        condition_summaries.append({
            'condition_id': condition_id,
            'n_seeds': len(condition_results),
            'n_trials_total': int(len(df_condition)),
            'p_open': float((df_condition['path_choice'] == 'open').mean()),
            'p_covered': float((df_condition['path_choice'] == 'covered').mean()),
            'mean_junction_pause_duration': float(df_condition['junction_pause_duration'].mean()),
            'mean_commit_latency': float(df_condition['commit_latency'].mean()),
            'mean_reorientation_count': float(df_condition['reorientation_count'].mean()),
            'mean_junction_deliberation_proxy': float(df_condition['junction_deliberation_proxy'].mean()),
            'p_commit_bound': float((df_condition['commit_reason'] == 'bound').mean()),
            'p_commit_timeout': float((df_condition['commit_reason'] == 'timeout').mean()),
            'mode_at_junction_distribution': mode_counts,
        })

    df_trials = pd.DataFrame(all_trials)
    df_seeds = pd.DataFrame(seed_summaries)
    df_conditions = pd.DataFrame(condition_summaries)

    trials_file = output_path / f"{prefix}_all_trials.csv"
    seeds_file = output_path / f"{prefix}_seed_summary.csv"
    conditions_file = output_path / f"{prefix}_condition_summary.csv"
    summary_file = output_path / f"{prefix}_run_summary.json"

    df_trials.to_csv(trials_file, index=False)
    df_seeds.to_csv(seeds_file, index=False)
    df_conditions.to_csv(conditions_file, index=False)

    run_summary = {
        'timestamp': datetime.now().isoformat(),
        'n_conditions': len(results),
        'condition_summaries': condition_summaries,
    }

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(df_trials)} trials to {trials_file}")
    print(f"Saved {len(df_seeds)} seed summaries to {seeds_file}")
    print(f"Saved {len(df_conditions)} condition summaries to {conditions_file}")
    print(f"Saved grid run summary to {summary_file}")

    print(f"\n{'='*60}")
    print("CONDITION MATRIX SUMMARY")
    print(f"{'='*60}")
    for condition in sorted(condition_summaries, key=lambda x: x['condition_id']):
        print(
            f"  {condition['condition_id']}: "
            f"P(open)={condition['p_open']:.3f}, "
            f"pause={condition['mean_junction_pause_duration']:.2f}, "
            f"latency={condition['mean_commit_latency']:.2f}, "
            f"timeout={condition['p_commit_timeout']:.2f}"
        )
    print(f"{'='*60}")

    return run_summary


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.condition:
        condition_name = normalize_condition_name(args.condition)
        run_label = f"{condition_name}_{args.ablation}"
        output_dir = build_timestamped_output_dir(args.output_dir, run_label)

        print(f"\nOutput directory: {output_dir}")

        run_single_condition(
            condition_name=condition_name,
            n_seeds=args.n_seeds,
            n_trials=args.n_trials,
            ablation=args.ablation,
            output_dir=output_dir,
            verbose=args.verbose
        )

    elif args.grid:
        run_label = f"grid_{args.grid}_{args.ablation}"
        output_dir = build_timestamped_output_dir(args.output_dir, run_label)

        print(f"\nOutput directory: {output_dir}")

        run_grid(
            grid_type=args.grid,
            n_seeds=args.n_seeds,
            n_trials=args.n_trials,
            ablation=args.ablation,
            output_dir=output_dir,
            verbose=args.verbose
        )

    elif args.one_shot:
        run_label = f"one_shot_{args.ablation}"
        output_dir = build_timestamped_output_dir(args.output_dir, run_label)

        print(f"\nOutput directory: {output_dir}")

        run_one_shot_protocol(
            n_seeds=args.n_seeds,
            n_trials=args.n_trials,
            ablation=args.ablation,
            output_dir=output_dir,
            verbose=args.verbose
        )


if __name__ == '__main__':
    main()
