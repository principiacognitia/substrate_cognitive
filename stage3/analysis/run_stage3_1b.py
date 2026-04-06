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
    mode_group.add_argument('--condition', type=str, 
                           choices=['balanced_conflict', 'reward_dominant', 'threat_dominant'],
                           help='Run single canonical condition')
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
    parser.add_argument('--output-dir', type=str, default='logs/stage3/stage3_1b/',
                       help='Output directory')
    parser.add_argument('--ablation', type=str, default='full',
                       choices=['full', 'novg', 'novp', 'nox', 'one_shot_off'],
                       help='Ablation condition (default: full)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    return parser.parse_args()


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
    
    Args:
        seed: Random seed
        condition_id: Condition ID (e.g., 'R1_T2')
        n_trials: Number of trials
        ablation: Ablation name
        one_shot_override: Optional one-shot config override
        verbose: Verbose output
    
    Returns:
        result: Dict with trial summaries and metadata
    """
    # Build environment config
    env_config = build_env_config_for_condition(condition_id)
    env_config['n_trials'] = n_trials
    
    # Override one-shot if specified
    if one_shot_override:
        env_config['one_shot'] = one_shot_override
    
    # Apply ablation
    agent_config = apply_ablation(json.loads(json.dumps(AGENT_CONFIG_3_1B)), ablation)
    
    # Create env and agent
    env = OpenCoveredChoiceEnv(env_config, seed=seed)
    agent = AgentStage3(agent_config, seed=seed)
    
    # Run trials
    trial_summaries = []
    
    for trial in range(1, n_trials + 1):
        obs = env.reset(trial=trial)
        
        # Reset agent for new trial
        agent.reset()
        
        done = False
        tick = 0
        max_ticks = 100  # Safety limit
        
        while not done and tick < max_ticks:
            tick += 1
            
            # Get agent action using step() method
            action, metadata = agent.step(obs)
            mode = metadata.get('mode', 'EXPLOIT')
            gate_trigger = metadata.get('gate_constraint', 'default')
            action_probs = metadata.get('action_probs', [0.5, 0.5])
            
            # Environment step
            obs, reward, done, info = env.step(
                action=action,
                mode=mode,
                gate_trigger=gate_trigger,
                action_probs=action_probs
            )
            
            # Agent update (already done in step())
        
        # Collect trial summary
        summaries = env.get_trial_summaries()
        if summaries:
            trial_summaries.append(summaries[-1])
    
    # Convert summaries to dicts
    trial_dicts = [s.to_dict() for s in trial_summaries]
    
    # Compute seed-level statistics
    path_choices = [t['path_choice'] for t in trial_dicts]
    p_open = sum(1 for p in path_choices if p == 'open') / len(path_choices)
    
    result = {
        'seed': seed,
        'condition_id': condition_id,
        'ablation': ablation,
        'n_trials': n_trials,
        'trial_summaries': trial_dicts,
        'p_open': p_open,
        'mean_junction_pause_duration': np.mean([t['junction_pause_duration'] for t in trial_dicts]),
        'mean_commit_latency': np.mean([t['commit_latency'] for t in trial_dicts]),
        'mean_reorientation_count': np.mean([t['reorientation_count'] for t in trial_dicts]),
        'mean_junction_deliberation_proxy': np.mean([t['junction_deliberation_proxy'] for t in trial_dicts]),
    }
    
    if verbose:
        print(f"  Seed {seed}: P(open)={p_open:.3f}, "
              f"pause={result['mean_junction_pause_duration']:.2f}, "
              f"latency={result['mean_commit_latency']:.2f}")
    
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
    """Runs one-shot protocol with pre/post blocks."""
    protocol = get_one_shot_protocol()
    
    print(f"\n{'='*60}")
    print(f"Stage 3.1B: One-Shot Protocol")
    print(f"Pre-block: {protocol['pre_block_trials']} trials")
    print(f"Shock trial: {protocol['shock_trial']}")
    print(f"Post-block: {protocol['post_block_trials']} trials")
    print(f"Seeds: {n_seeds}, Ablation: {ablation}")
    print(f"{'='*60}")
    
    # Use balanced conflict condition
    condition_id = 'R1_T2'
    
    all_results = []
    
    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx
        result = run_condition(
            seed=seed,
            condition_id=condition_id,
            n_trials=n_trials,
            ablation=ablation,
            one_shot_override=protocol['one_shot'],
            verbose=verbose
        )
        all_results.append(result)
    
    # Aggregate results
    return aggregate_and_save(all_results, output_dir, f"one_shot_{ablation}")


def aggregate_and_save(
    results: List[Dict],
    output_dir: str,
    prefix: str
) -> Dict[str, Any]:
    """Aggregates results and saves to CSV/JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Flatten trial summaries
    all_trials = []
    seed_summaries = []
    
    for result in results:
        # Add seed-level summary
        seed_summaries.append({
            'seed': result['seed'],
            'condition_id': result['condition_id'],
            'ablation': result['ablation'],
            'n_trials': result['n_trials'],
            'p_open': result['p_open'],
            'mean_junction_pause_duration': result['mean_junction_pause_duration'],
            'mean_commit_latency': result['mean_commit_latency'],
            'mean_reorientation_count': result['mean_reorientation_count'],
            'mean_junction_deliberation_proxy': result['mean_junction_deliberation_proxy'],
        })
        
        # Add all trials
        for trial in result['trial_summaries']:
            all_trials.append(trial)
    
    # Save trial-level data
    if all_trials:
        df_trials = pd.DataFrame(all_trials)
        trials_file = output_path / f"{prefix}_all_trials.csv"
        df_trials.to_csv(trials_file, index=False)
        print(f"\nSaved {len(df_trials)} trials to {trials_file}")
    
    # Save seed-level summary
    df_seeds = pd.DataFrame(seed_summaries)
    seeds_file = output_path / f"{prefix}_seed_summary.csv"
    df_seeds.to_csv(seeds_file, index=False)
    print(f"Saved {len(df_seeds)} seed summaries to {seeds_file}")
    
    # Compute condition-level statistics
    condition_summary = {
        'condition_id': results[0]['condition_id'] if results else '',
        'ablation': results[0]['ablation'] if results else '',
        'n_seeds': len(results),
        'n_trials_per_seed': results[0]['n_trials'] if results else 0,
        'p_open_mean': np.mean([r['p_open'] for r in results]),
        'p_open_std': np.std([r['p_open'] for r in results]),
        'mean_junction_pause_duration': np.mean([r['mean_junction_pause_duration'] for r in results]),
        'mean_commit_latency': np.mean([r['mean_commit_latency'] for r in results]),
        'mean_reorientation_count': np.mean([r['mean_reorientation_count'] for r in results]),
        'mean_junction_deliberation_proxy': np.mean([r['mean_junction_deliberation_proxy'] for r in results]),
    }
    
    # Save run summary
    run_summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_seeds': len(results),
            'n_trials': results[0]['n_trials'] if results else 0,
            'ablation': results[0]['ablation'] if results else '',
        },
        'condition_summary': condition_summary,
    }
    
    summary_file = output_path / f"{prefix}_run_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(run_summary, f, indent=2)
    print(f"Saved run summary to {summary_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {condition_summary['condition_id']} ({condition_summary['ablation']})")
    print(f"  P(open) = {condition_summary['p_open_mean']:.3f} ± {condition_summary['p_open_std']:.3f}")
    print(f"  Junction pause = {condition_summary['mean_junction_pause_duration']:.2f}")
    print(f"  Commit latency = {condition_summary['mean_commit_latency']:.2f}")
    print(f"  Deliberation proxy = {condition_summary['mean_junction_deliberation_proxy']:.2f}")
    print(f"{'='*60}")
    
    return run_summary


def aggregate_grid_and_save(
    results: Dict[str, List[Dict]],
    output_dir: str,
    prefix: str
) -> Dict[str, Any]:
    """Aggregates grid results and saves to CSV/JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Flatten all trials
    all_trials = []
    seed_summaries = []
    condition_summaries = []
    
    for condition_id, condition_results in results.items():
        for result in condition_results:
            # Seed summary
            seed_summaries.append({
                'seed': result['seed'],
                'condition_id': result['condition_id'],
                'ablation': result['ablation'],
                'p_open': result['p_open'],
                'mean_junction_pause_duration': result['mean_junction_pause_duration'],
                'mean_commit_latency': result['mean_commit_latency'],
                'mean_reorientation_count': result['mean_reorientation_count'],
                'mean_junction_deliberation_proxy': result['mean_junction_deliberation_proxy'],
            })
            
            # Trials
            for trial in result['trial_summaries']:
                all_trials.append(trial)
        
        # Condition-level summary
        condition_summaries.append({
            'condition_id': condition_id,
            'n_seeds': len(condition_results),
            'p_open_mean': np.mean([r['p_open'] for r in condition_results]),
            'p_open_std': np.std([r['p_open'] for r in condition_results]),
            'mean_junction_pause_duration': np.mean([r['mean_junction_pause_duration'] for r in condition_results]),
            'mean_commit_latency': np.mean([r['mean_commit_latency'] for r in condition_results]),
            'mean_reorientation_count': np.mean([r['mean_reorientation_count'] for r in condition_results]),
            'mean_junction_deliberation_proxy': np.mean([r['mean_junction_deliberation_proxy'] for r in condition_results]),
        })
    
    # Save all trials
    if all_trials:
        df_trials = pd.DataFrame(all_trials)
        trials_file = output_path / f"{prefix}_all_trials.csv"
        df_trials.to_csv(trials_file, index=False)
        print(f"\nSaved {len(df_trials)} trials to {trials_file}")
    
    # Save seed summaries
    df_seeds = pd.DataFrame(seed_summaries)
    seeds_file = output_path / f"{prefix}_seed_summary.csv"
    df_seeds.to_csv(seeds_file, index=False)
    print(f"Saved {len(df_seeds)} seed summaries to {seeds_file}")
    
    # Save condition summaries
    df_conditions = pd.DataFrame(condition_summaries)
    conditions_file = output_path / f"{prefix}_condition_summary.csv"
    df_conditions.to_csv(conditions_file, index=False)
    print(f"Saved {len(df_conditions)} condition summaries to {conditions_file}")
    
    # Print heatmap-style summary
    print(f"\n{'='*60}")
    print("CONDITION MATRIX SUMMARY (P(open))")
    print(f"{'='*60}")
    
    # Extract reward and threat levels
    for condition in sorted(condition_summaries, key=lambda x: x['condition_id']):
        cid = condition['condition_id']
        p_open = condition['p_open_mean']
        pause = condition['mean_junction_pause_duration']
        print(f"  {cid}: P(open)={p_open:.3f}, pause={pause:.2f}")
    
    print(f"{'='*60}")
    
    return {
        'timestamp': datetime.now().isoformat(),
        'n_conditions': len(results),
        'condition_summaries': condition_summaries,
    }


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.condition:
        run_single_condition(
            condition_name=args.condition,
            n_seeds=args.n_seeds,
            n_trials=args.n_trials,
            ablation=args.ablation,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
    
    elif args.grid:
        run_grid(
            grid_type=args.grid,
            n_seeds=args.n_seeds,
            n_trials=args.n_trials,
            ablation=args.ablation,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
    
    elif args.one_shot:
        run_one_shot_protocol(
            n_seeds=args.n_seeds,
            n_trials=args.n_trials,
            ablation=args.ablation,
            output_dir=args.output_dir,
            verbose=args.verbose
        )


if __name__ == '__main__':
    main()
