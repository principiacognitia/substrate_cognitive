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
    parser.add_argument(
        '--diagnostic-forced-shock',
        action="store_true",
        help="Diagnostic mode: on one-shot trial, force action toward configured one_shot_path at junction"
    )
    parser.add_argument(
        '--forced-shock-path', type=str, default=None,
        choices=['open', 'covered', None],
        help="Optional override for forced event path in diagnostic mode; default uses env.one_shot_path"
    )
    parser.add_argument(
        '--diagnostic-forced-treat',
        action="store_true",
        help="Diagnostic mode: force action toward configured treat path at junction"
    )
    parser.add_argument(
        '--one-shot-kind',
        type=str,
        default='shock',
        choices=['shock', 'treat'],
        help='One-shot protocol kind (default: shock)'
    )
    parser.add_argument(
        '--one-shot-path-override',
        type=str,
        default=None,
        choices=['open', 'covered', None],
        help='Optional override for one-shot path'
    )
    parser.add_argument(
        '--one-shot-reward-override',
        type=float,
        default=None,
        help='Optional override for one-shot reward magnitude'
    )
    parser.add_argument(
        '--one-shot-salience-override',
        type=float,
        default=None,
        help='Optional override for one-shot salience'
    )
    parser.add_argument(
        '--one-shot-stakes-override',
        type=float,
        default=None,
        help='Optional override for one-shot stakes'
    )
    
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
    parser.add_argument(
        '--w-qneg-input-override',
        type=float,
        default=None,
        help='Optional override for temporal_state.w_qneg_input'
    )
    parser.add_argument(
        '--w-qpos-input-override',
        type=float,
        default=None,
        help='Optional override for temporal_state.w_qpos_input'
    )
    parser.add_argument(
        '--k-pos-override',
        type=float,
        default=None,
        help='Optional override for temporal_state.k_pos'
    )
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    # Diagnostic trace options
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable diagnostic trace collection'
    )
    parser.add_argument(
        '--debug-console',
        action='store_true',
        help='Print compact diagnostic trace to console'
    )
    parser.add_argument(
        '--debug-console-start',
        type=int,
        default=None,
        help='First trial for console debug printing (inclusive)'
    )
    parser.add_argument(
        '--debug-console-end',
        type=int,
        default=None,
        help='Last trial for console debug printing (inclusive)'
    )
    parser.add_argument(
        '--debug-junction-only',
        action='store_true',
        help='Store/print only junction-related diagnostic rows'
    )
    parser.add_argument(
        '--debug-trial-window',
        type=int,
        default=2,
        help='Store all rows within +/- N trials around shock trial (default: 2)'
)
    
    args = parser.parse_args()

    if args.diagnostic_forced_shock and args.diagnostic_forced_treat:
        parser.error('--diagnostic-forced-shock and --diagnostic-forced-treat are mutually exclusive')

    if (args.debug_console_start is None) ^ (args.debug_console_end is None):
        parser.error('--debug-console-start and --debug-console-end must be provided together')

    if (
        args.debug_console_start is not None and
        args.debug_console_end is not None and
        args.debug_console_start > args.debug_console_end
    ):
        parser.error('--debug-console-start must be <= --debug-console-end')

    return args



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

def fmt3(x):
    """Safe formatter for debug console."""
    if x is None:
        return "nan"
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return "nan"

def format_param_tag(prefix: str, x: Optional[float]) -> str:
    """Short suffix for sweep parameter in run labels."""
    if x is None:
        return ""
    return f"_{prefix}_{int(round(float(x) * 100)):03d}"

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
    verbose: bool = False,
    diagnostic_forced_shock=False,
    forced_shock_path=None,
    debug=False,
    debug_console=False,
    debug_console_start=None,
    debug_console_end=None,
    debug_junction_only=False,
    debug_trial_window=2,
    w_qneg_input_override=None,
    w_qpos_input_override=None,
    k_pos_override=None,
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
    env_config['debug'] = False
    env_config['n_trials'] = n_trials

    # Ablation-specific handling for one-shot
    if ablation == 'one_shot_off':
        env_config['one_shot'] = json.loads(json.dumps(ONE_SHOT_DISABLED))

    # Apply ablation to agent config
    agent_config = apply_ablation(json.loads(json.dumps(AGENT_CONFIG_3_1B)), ablation)

    if w_qneg_input_override is not None:
        agent_config.setdefault('temporal_state', {})['w_qneg_input'] = float(w_qneg_input_override)

    if w_qpos_input_override is not None:
        agent_config.setdefault('temporal_state', {})['w_qpos_input'] = float(w_qpos_input_override)

    if k_pos_override is not None:
        agent_config.setdefault('temporal_state', {})['k_pos'] = float(k_pos_override)

    w_qneg_input_effective = (
        agent_config.get('temporal_state', {}).get('w_qneg_input', np.nan)
    )
    w_qpos_input_effective = (
        agent_config.get('temporal_state', {}).get('w_qpos_input', np.nan)
    )
    k_pos_effective = (
        agent_config.get('temporal_state', {}).get('k_pos', np.nan)
    )

    # Create env and agent
    env = OpenCoveredChoiceEnv(env_config, seed=seed)
    agent = AgentStage3(agent_config, seed=seed)

    trial_summaries = []
    step_rows = []
    debug_rows = []

    # IMPORTANT: reward feedback from previous step
    prev_reward = 0.0

    forced_action_applied_count = 0

    # Temporal state
    pending_one_shot_salience = None
    pending_one_shot_stakes = None
    pending_one_shot_source_trial = None
    pending_one_shot_source_tick = None
    pending_one_shot_source_X_risk = None
    pending_one_shot_source_X_opp = None
    pending_one_shot_source_reward = None

    for trial in range(1, n_trials + 1):
        obs = env.reset(trial=trial)

        console_window_active = False
        if debug_console:
            if debug_console_start is None or debug_console_end is None:
                console_window_active = True
            else:
                console_window_active = (debug_console_start <= trial <= debug_console_end)

        env.debug = bool(console_window_active)

        # НЕ вызываем agent.reset() здесь:
        # Stage 3.1B должен позволять persistence через TemporalState

        done = False
        tick = 0
        max_ticks = 100

        while not done and tick < max_ticks:
            tick += 1

            step_salience = None
            step_stakes = None
            step_one_shot_from_pending = False
            step_one_shot_source_trial = None
            step_one_shot_source_tick = None
            step_one_shot_source_X_risk = None
            step_one_shot_source_X_opp = None
            step_one_shot_source_reward = None

            if pending_one_shot_salience is not None:
                step_salience = float(pending_one_shot_salience)
                step_stakes = float(pending_one_shot_stakes if pending_one_shot_stakes is not None else 1.0)
                step_one_shot_from_pending = True

                step_one_shot_source_trial = pending_one_shot_source_trial
                step_one_shot_source_tick = pending_one_shot_source_tick
                step_one_shot_source_X_risk = pending_one_shot_source_X_risk
                step_one_shot_source_X_opp = pending_one_shot_source_X_opp
                step_one_shot_source_reward = pending_one_shot_source_reward

                pending_one_shot_salience = None
                pending_one_shot_stakes = None
                pending_one_shot_source_trial = None
                pending_one_shot_source_tick = None
                pending_one_shot_source_X_risk = None
                pending_one_shot_source_X_opp = None
                pending_one_shot_source_reward = None

            elif float(obs.get('one_shot_fired', 0.0)) > 0.0:
                step_salience = float(obs.get('one_shot_salience', 0.0))
                step_stakes = float(obs.get('one_shot_stakes', 1.0))

            obs_one_shot_fired_pre = obs.get('one_shot_fired', np.nan)
            obs_one_shot_salience_pre = obs.get('one_shot_salience', np.nan)
            obs_one_shot_stakes_pre = obs.get('one_shot_stakes', np.nan)

            obs_pre = dict(obs)

            if step_one_shot_from_pending:
                obs_pre['one_shot_source_X_risk'] = (
                    float(step_one_shot_source_X_risk)
                    if step_one_shot_source_X_risk is not None else 0.0
                )
                obs_pre['one_shot_source_X_opp'] = (
                    float(step_one_shot_source_X_opp)
                    if step_one_shot_source_X_opp is not None else 0.0
                )
                obs_pre['one_shot_source_reward'] = (
                    float(step_one_shot_source_reward)
                    if step_one_shot_source_reward is not None else 0.0
                )
            else:
                obs_pre['one_shot_source_X_risk'] = obs_pre.get('one_shot_source_X_risk', 0.0)
                obs_pre['one_shot_source_X_opp'] = obs_pre.get('one_shot_source_X_opp', 0.0)
                obs_pre['one_shot_source_reward'] = obs_pre.get('one_shot_source_reward', 0.0)

            action, metadata = agent.step(
                observation=obs_pre,
                reward=prev_reward,
                salience=step_salience,
                stakes=step_stakes
            )

            mode = metadata.get('mode', 'EXPLOIT')
            gate_trigger = metadata.get('gate_constraint', 'default')
            action_probs = metadata.get('action_probs', [0.5, 0.5])

            forced_action_applied = False

            # -----------------------------------------------------------------
            # Diagnostic forced-shock mode
            # Purpose:
            #   Guarantee that on shock trial the agent commits to one_shot_path,
            #   so that we can test causal post-shock effects without dilution.
            # This is a diagnostic intervention only, not a final experiment.
            # -----------------------------------------------------------------
            if diagnostic_forced_shock:
                shock_trial = getattr(env, "one_shot_trial", -1)
                configured_shock_path = forced_shock_path or getattr(env, "one_shot_path", "open")

                at_junction = float(obs.get("at_junction", 0.0)) > 0.5
                already_committed = float(obs.get("committed_path_encoded", 0.0)) != 0.0

                if trial == shock_trial and at_junction and not already_committed:
                    if configured_shock_path == "open":
                        action = 0
                    elif configured_shock_path == "covered":
                        action = 1

                    forced_action_applied = True
                    forced_action_applied_count += 1

                    event_reward = float(env_config.get("one_shot", {}).get("one_shot_reward", 0.0))
                    force_tag = "forced_treat" if event_reward > 0.0 else "forced_shock"
                    gate_trigger = f"{gate_trigger}|{force_tag}"

            obs_post, reward, done, info = env.step(
                action=action,
                mode=mode,
                gate_trigger=gate_trigger,
                action_probs=action_probs
            )

            obs = obs_post

            post_step_one_shot_fired = (
                float(obs_post.get('one_shot_fired', 0.0)) > 0.0
                or bool(info.get('one_shot_fired', False))
            )

            if post_step_one_shot_fired:
                pending_one_shot_salience = float(
                    obs_post.get('one_shot_salience', info.get('one_shot_salience', 0.0))
                )
                pending_one_shot_stakes = float(
                    obs_post.get('one_shot_stakes', info.get('one_shot_stakes', 1.0))
                )
                pending_one_shot_source_trial = trial
                pending_one_shot_source_tick = info.get('tick', tick)

                pending_one_shot_source_X_risk = float(
                    obs_post.get('one_shot_source_X_risk', info.get('one_shot_source_X_risk', 0.0))
                )
                pending_one_shot_source_X_opp = float(
                    obs_post.get('one_shot_source_X_opp', info.get('one_shot_source_X_opp', 0.0))
                )
                pending_one_shot_source_reward = float(
                    obs_post.get('one_shot_source_reward', info.get('one_shot_source_reward', reward))
                )

                source_override_mode = env_config.get('one_shot', {}).get('source_override_mode', 'none')

                if source_override_mode == 'path_negative':
                    pending_one_shot_source_X_opp = 0.0

                elif source_override_mode == 'positive_reward':
                    pending_one_shot_source_X_risk = 0.0
                    pending_one_shot_source_X_opp = 1.0

            temporal_state = metadata.get('temporal_state', {})
            node_exposure = metadata.get('node_exposure', {})

            step_rows.append({
                'seed': seed,
                'condition_id': condition_id,
                'ablation': ablation,
                'trial': trial,
                'tick': info.get('tick', tick),

                'node_id': info.get('node_id', ''),
                'at_junction': info.get('at_junction', False),
                'deliberation_state': info.get('deliberation_state', ''),
                'candidate_path': info.get('candidate_path', ''),
                'committed_path': info.get('committed_path', ''),

                'mode': mode,
                'gate_trigger': gate_trigger,
                'action': action,
                'reward': reward,

                'X_risk': node_exposure.get('X_risk', np.nan),
                'X_opp': node_exposure.get('X_opp', np.nan),
                'D_est': node_exposure.get('D_est', np.nan),

                'h_risk': temporal_state.get('h_risk', np.nan),
                'h_opp': temporal_state.get('h_opp', np.nan),
                'h_time': temporal_state.get('h_time', np.nan),
                'q_neg': temporal_state.get('q_neg', np.nan),
                'q_pos': temporal_state.get('q_pos', np.nan),

                'one_shot_fired': temporal_state.get('one_shot_pending', False),
                'one_shot_type': metadata.get('one_shot_type', 'none'),

                'open_reward_prob': info.get('open_reward_prob', np.nan),
                'covered_reward_prob': info.get('covered_reward_prob', np.nan),
                'open_X_risk': info.get('open_X_risk', np.nan),
                'covered_X_risk': info.get('covered_X_risk', np.nan),
                'reward_gap': info.get('reward_gap', np.nan),
                'risk_gap': info.get('risk_gap', np.nan),
                'threat_gap': info.get('threat_gap', np.nan),
                'conflict_condition': info.get('conflict_condition', ''),

                'salience': metadata.get('salience_used', np.nan),
                'stakes': metadata.get('stakes_used', np.nan),
                'one_shot_amplitude': metadata.get('one_shot_amplitude', np.nan),
                'one_shot_pending': metadata.get('one_shot_pending', False),

                'u_delta': metadata.get('instant_diagnostics', {}).get('u_delta', np.nan),
                'u_entropy': metadata.get('instant_diagnostics', {}).get('u_entropy', np.nan),
                'u_volatility': metadata.get('instant_diagnostics', {}).get('u_volatility', np.nan),

                'one_shot_active': info.get('one_shot_active', False),
                'one_shot_trial': info.get('one_shot_trial', -1),
                'one_shot_path': info.get('one_shot_path', ''),
                'one_shot_kind': env_config.get('one_shot', {}).get('one_shot_kind', 'off'),
                'source_override_mode': env_config.get('one_shot', {}).get('source_override_mode', 'none'),

                'diagnostic_forced_shock': diagnostic_forced_shock,
                'forced_shock_path': forced_shock_path or getattr(env, "one_shot_path", ""),
                'forced_action_applied': forced_action_applied,
                'one_shot_kind': env_config.get('one_shot', {}).get('one_shot_kind', 'off'),
                'source_override_mode': env_config.get('one_shot', {}).get('source_override_mode', 'none'),

                # Temporal state for next step (if one-shot fired)
                'step_one_shot_from_pending': step_one_shot_from_pending,
                'pending_one_shot_salience_used': step_salience,
                'pending_one_shot_stakes_used': step_stakes,
                'post_step_one_shot_fired': post_step_one_shot_fired,
                'pending_one_shot_source_trial': step_one_shot_source_trial,
                'pending_one_shot_source_tick': step_one_shot_source_tick,
                'one_shot_source_X_risk': step_one_shot_source_X_risk,
                'one_shot_source_X_opp': step_one_shot_source_X_opp,
                'one_shot_source_reward': step_one_shot_source_reward,

                'obs_one_shot_fired_pre': obs_one_shot_fired_pre,
                'obs_one_shot_salience_pre': obs_one_shot_salience_pre,
                'obs_one_shot_stakes_pre': obs_one_shot_stakes_pre,
            })

            prev_reward = reward

            # Determine if we should store debug row for this step
            shock_trial = getattr(env, 'one_shot_trial', -1)
            in_shock_window = abs(trial - shock_trial) <= debug_trial_window if shock_trial >= 0 else False
            at_junction = bool(info.get('at_junction', False))

            pre_at_junction = float(obs_pre.get('at_junction', 0.0)) > 0.5
            post_at_junction = bool(info.get('at_junction', False))

            policy_q_values = metadata.get('q_values', [])
            real_junction_choice_row = bool(pre_at_junction and isinstance(policy_q_values, list) and len(policy_q_values) == 2)

            should_store_debug = debug and (
                (not debug_junction_only) or
                pre_at_junction or
                post_at_junction or
                in_shock_window or
                bool(info.get('one_shot_fired', False)) or
                forced_action_applied
            )

            # Store debug row if any of the following is true:
            debug_row = {
                'seed': seed,
                'condition_id': condition_id,
                'ablation': ablation,
                'trial': trial,
                'tick': info.get('tick', tick),

                'pre_node_id': obs_pre.get('node_id', ''),
                'pre_at_junction': pre_at_junction,
                'pre_deliberation_state': obs_pre.get('deliberation_state', ''),
                'pre_X_risk': obs_pre.get('X_risk', np.nan),
                'pre_X_opp': obs_pre.get('X_opp', np.nan),
                'pre_D_est': obs_pre.get('D_est', np.nan),

                'pre_q_values': obs_pre.get('q_values', []),
                'pre_risk_values': obs_pre.get('risk_values', []),

                'pre_option_reward_values': obs_pre.get('option_reward_values', []),
                'pre_option_risk_values': obs_pre.get('option_risk_values', []),
                'pre_option_visibility_values': obs_pre.get('option_visibility_values', []),
                'pre_option_expected_threat_values': obs_pre.get('option_expected_threat_values', []),

                'pre_one_shot_fired': obs_pre.get('one_shot_fired', np.nan),
                'pre_one_shot_salience': obs_pre.get('one_shot_salience', np.nan),
                'pre_one_shot_stakes': obs_pre.get('one_shot_stakes', np.nan),

                'pre_one_shot_source_X_risk': obs_pre.get('one_shot_source_X_risk', np.nan),
                'pre_one_shot_source_X_opp': obs_pre.get('one_shot_source_X_opp', np.nan),
                'pre_one_shot_source_reward': obs_pre.get('one_shot_source_reward', np.nan),

                'mode': mode,
                'gate_trigger': gate_trigger,
                'action': action,

                'u_delta': metadata.get('instant_diagnostics', {}).get('u_delta', np.nan),
                'u_entropy': metadata.get('instant_diagnostics', {}).get('u_entropy', np.nan),
                'u_volatility': metadata.get('instant_diagnostics', {}).get('u_volatility', np.nan),

                'h_risk': metadata.get('temporal_state', {}).get('h_risk', np.nan),
                'h_opp': metadata.get('temporal_state', {}).get('h_opp', np.nan),
                'h_time': metadata.get('temporal_state', {}).get('h_time', np.nan),

                'q_neg': metadata.get('temporal_state', {}).get('q_neg', np.nan),
                'q_pos': metadata.get('temporal_state', {}).get('q_pos', np.nan),
                'one_shot_type': metadata.get('one_shot_type', 'none'),

                'safe_drive': metadata.get('safe_drive', np.nan),
                'uncertainty_signal': metadata.get('uncertainty_signal', np.nan),
                'v_g_approx': metadata.get('v_g_approx', np.nan),
                'explore_gate_output': metadata.get('explore_gate_output', np.nan),

                'node_X_risk': metadata.get('node_exposure', {}).get('X_risk', np.nan),
                'node_X_opp': metadata.get('node_exposure', {}).get('X_opp', np.nan),
                'node_D_est': metadata.get('node_exposure', {}).get('D_est', np.nan),

                'gate_X_risk': metadata.get('gate_exposure', {}).get('X_risk', np.nan),
                'gate_X_opp': metadata.get('gate_exposure', {}).get('X_opp', np.nan),
                'gate_D_est': metadata.get('gate_exposure', {}).get('D_est', np.nan),
                'gate_exposure_source': metadata.get('gate_exposure_source', ''),

                'policy_q_values': metadata.get('q_values', []),
                'policy_risk_values': metadata.get('risk_values', []),
                'action_probs': metadata.get('action_probs', []),

                'gate_state_snapshot': metadata.get('gate_state_snapshot', {}),
                'action_policy_debug': metadata.get('action_policy_debug', {}),

                'post_node_id': info.get('node_id', ''),
                'post_at_junction': post_at_junction,
                'post_deliberation_state': info.get('deliberation_state', ''),
                'candidate_path': info.get('candidate_path', ''),
                'committed_path': info.get('committed_path', ''),

                'open_X_risk': info.get('open_X_risk', np.nan),
                'covered_X_risk': info.get('covered_X_risk', np.nan),
                'open_reward_prob': info.get('open_reward_prob', np.nan),
                'covered_reward_prob': info.get('covered_reward_prob', np.nan),

                'reward': reward,
                'post_one_shot_fired': bool(info.get('one_shot_fired', False)),
                'one_shot_pending': metadata.get('one_shot_pending', False),
                'one_shot_amplitude': metadata.get('one_shot_amplitude', np.nan),
                'salience_used': metadata.get('salience_used', np.nan),
                'stakes_used': metadata.get('stakes_used', np.nan),

                'diagnostic_forced_shock': diagnostic_forced_shock,
                'forced_shock_path': forced_shock_path or getattr(env, "one_shot_path", ""),
                'forced_action_applied': forced_action_applied,

                'step_one_shot_from_pending': step_one_shot_from_pending,
                'pending_one_shot_salience_used': step_salience,
                'pending_one_shot_stakes_used': step_stakes,

                'pending_one_shot_source_trial': step_one_shot_source_trial,
                'pending_one_shot_source_tick': step_one_shot_source_tick,
                'one_shot_source_X_risk': step_one_shot_source_X_risk,
                'one_shot_source_X_opp': step_one_shot_source_X_opp,
                'one_shot_source_reward': step_one_shot_source_reward,

                'real_junction_choice_row': real_junction_choice_row,
            }

            if should_store_debug:
                debug_rows.append(debug_row)

            if console_window_active and should_store_debug:
                probs = debug_row.get('action_probs', [])
                if isinstance(probs, list) and len(probs) == 2:
                    probs_str = f"[{fmt3(probs[0])},{fmt3(probs[1])}]"
                else:
                    probs_str = "[]"

                print(
                    "[DBG] "
                    f"s={debug_row['seed']} t={debug_row['trial']} k={debug_row['tick']} "
                    f"node={debug_row['post_node_id']} st={debug_row['post_deliberation_state']} "
                    f"mode={debug_row['mode']} gate={debug_row['gate_trigger']} "
                    f"a={debug_row['action']} r={fmt3(debug_row['reward'])} "
                    f"Hr={fmt3(debug_row.get('h_risk'))} "
                    f"Qn={fmt3(debug_row.get('q_neg'))} "
                    f"Qp={fmt3(debug_row.get('q_pos'))} "
                    f"safe={fmt3(debug_row.get('safe_drive'))} "
                    f"unc={fmt3(debug_row.get('uncertainty_signal'))} "
                    f"p={probs_str} "
                    f"path={debug_row.get('candidate_path')}->{debug_row.get('committed_path')} "
                    f"shot={debug_row.get('step_one_shot_from_pending')}/{debug_row.get('post_one_shot_fired')}",
                    flush=True
                )
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
        'step_rows': step_rows,
        'p_open': p_open,
        'p_covered': p_covered,
        'mean_junction_pause_duration': np.mean([t['junction_pause_duration'] for t in trial_dicts]) if trial_dicts else 0.0,
        'mean_commit_latency': np.mean([t['commit_latency'] for t in trial_dicts]) if trial_dicts else 0.0,
        'mean_reorientation_count': np.mean([t['reorientation_count'] for t in trial_dicts]) if trial_dicts else 0.0,
        'mean_junction_deliberation_proxy': np.mean([t['junction_deliberation_proxy'] for t in trial_dicts]) if trial_dicts else 0.0,
        'p_commit_bound': np.mean([t['commit_reason'] == 'bound' for t in trial_dicts]) if trial_dicts else 0.0,
        'p_commit_timeout': np.mean([t['commit_reason'] == 'timeout' for t in trial_dicts]) if trial_dicts else 0.0,
        'diagnostic_forced_shock': diagnostic_forced_shock,
        'forced_shock_path': forced_shock_path or getattr(env, "one_shot_path", ""),
        'forced_action_applied_count': forced_action_applied_count,
        'w_qneg_input_override': w_qneg_input_override,
        'w_qneg_input_effective': w_qneg_input_effective,
        'w_qpos_input_override': w_qpos_input_override,
        'w_qpos_input_effective': w_qpos_input_effective,
        'k_pos_override': k_pos_override,
        'k_pos_effective': k_pos_effective,
        'debug_rows': debug_rows,
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
    verbose: bool,
    diagnostic_forced_shock=False,
    forced_shock_path=None,
    debug=False,
    debug_console=False,
    debug_console_start=None,
    debug_console_end=None,
    debug_junction_only=False,
    debug_trial_window=2,
    w_qneg_input_override=None,
    w_qpos_input_override=None,
    k_pos_override=None,
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
            diagnostic_forced_shock=diagnostic_forced_shock,
            forced_shock_path=forced_shock_path,
            verbose=verbose,
            debug=debug,
            debug_console=debug_console,
            debug_console_start=debug_console_start,
            debug_console_end=debug_console_end,
            debug_junction_only=debug_junction_only,
            debug_trial_window=debug_trial_window,
            w_qneg_input_override=w_qneg_input_override,
            w_qpos_input_override=w_qpos_input_override,
            k_pos_override=k_pos_override,
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
    verbose: bool,
    diagnostic_forced_shock=False,
    forced_shock_path=None,
    debug=False,
    debug_console=False,
    debug_console_start=None,
    debug_console_end=None,
    debug_junction_only=False,
    debug_trial_window=2,
    w_qneg_input_override=None,
    w_qpos_input_override=None,
    k_pos_override=None,
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
                diagnostic_forced_shock=diagnostic_forced_shock,
                forced_shock_path=forced_shock_path,
                verbose=verbose,
                debug=debug,
                debug_console=debug_console,
                debug_console_start=debug_console_start,
                debug_console_end=debug_console_end,
                debug_junction_only=debug_junction_only,
                debug_trial_window=debug_trial_window,
                w_qneg_input_override=w_qneg_input_override,
                w_qpos_input_override=w_qpos_input_override,
                k_pos_override=k_pos_override,
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
    verbose: bool,
    diagnostic_forced_shock=False,
    forced_shock_path=None,
    debug=False,
    debug_console=False,
    debug_console_start=None,
    debug_console_end=None,
    debug_junction_only=False,
    debug_trial_window=2,
    w_qneg_input_override=None,
    w_qpos_input_override=None,
    k_pos_override=None,
    one_shot_protocol=None
) -> Dict[str, Any]:
    """
    Runs one-shot protocol with pre/post blocks.
    """
    protocol = one_shot_protocol if one_shot_protocol is not None else get_one_shot_protocol()

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
            diagnostic_forced_shock=diagnostic_forced_shock,
            forced_shock_path=forced_shock_path,
            verbose=verbose,
            debug=debug,
            debug_console=debug_console,
            debug_console_start=debug_console_start,
            debug_console_end=debug_console_end,
            debug_junction_only=debug_junction_only,
            debug_trial_window=debug_trial_window,
            w_qneg_input_override=w_qneg_input_override,
            w_qpos_input_override=w_qpos_input_override,
            k_pos_override=k_pos_override,
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
    all_steps = []
    seed_summaries = []
    all_debug_rows = []

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
            'w_qneg_input_effective': result.get('w_qneg_input_effective', np.nan),
            'w_qpos_input_effective': result.get('w_qpos_input_effective', np.nan),
            'k_pos_effective': result.get('k_pos_effective', np.nan),
        })
        for trial in result['trial_summaries']:
            all_trials.append(trial)
        for step in result.get('step_rows', []):
            all_steps.append(step)
        for row in result.get('debug_rows', []):
            all_debug_rows.append(row)
    if not all_trials:
        raise RuntimeError("No trials collected in aggregate_and_save()")

    df_trials = pd.DataFrame(all_trials)
    df_seeds = pd.DataFrame(seed_summaries)

    trials_file = output_path / f"{prefix}_all_trials.csv"
    steps_file = output_path / f"{prefix}_all_steps.csv"
    seeds_file = output_path / f"{prefix}_seed_summary.csv"
    condition_file = output_path / f"{prefix}_condition_summary.csv"
    summary_file = output_path / f"{prefix}_run_summary.json"
    debug_file = output_path / f"{prefix}_debug_trace.jsonl"

    df_trials.to_csv(trials_file, index=False)
    if all_steps:
        df_steps = pd.DataFrame(all_steps)
        df_steps.to_csv(steps_file, index=False)
        print(f"Saved {len(df_steps)} steps to {steps_file}")
    df_seeds.to_csv(seeds_file, index=False)
    
    if all_debug_rows:
        with open(debug_file, 'w', encoding='utf-8') as f:
            for row in all_debug_rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f"Saved {len(all_debug_rows)} debug rows to {debug_file}")

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
        'w_qneg_input_effective': float(df_seeds['w_qneg_input_effective'].iloc[0]) if 'w_qneg_input_effective' in df_seeds.columns else np.nan,
        'w_qpos_input_effective': float(df_seeds['w_qpos_input_effective'].iloc[0]) if 'w_qpos_input_effective' in df_seeds.columns else np.nan,
        'k_pos_effective': float(df_seeds['k_pos_effective'].iloc[0]) if 'k_pos_effective' in df_seeds.columns else np.nan,
    }

    pd.DataFrame([condition_summary]).to_csv(condition_file, index=False)

    run_summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_seeds': int(df_seeds['seed'].nunique()),
            'n_trials_per_seed': int(df_seeds['n_trials'].iloc[0]) if 'n_trials' in df_seeds.columns else 0,
            'ablation': condition_summary['ablation'],
            'w_qneg_input_override': condition_summary.get('w_qneg_input_effective', np.nan),
            'w_qpos_input_override': condition_summary.get('w_qpos_input_effective', np.nan),
            'k_pos_override': condition_summary.get('k_pos_effective', np.nan),
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
    all_steps = []
    seed_summaries = []
    condition_summaries = []
    all_debug_rows = []

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
                'w_qneg_input_effective': result.get('w_qneg_input_effective', np.nan),
                'w_qpos_input_effective': result.get('w_qpos_input_effective', np.nan),
                'k_pos_effective': result.get('k_pos_effective', np.nan),
            })

            for trial in result['trial_summaries']:
                all_trials.append(trial)
                condition_trials.append(trial)

            for step in result.get('step_rows', []):
                all_steps.append(step)

            for row in result.get('debug_rows', []):
                all_debug_rows.append(row)    

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
    steps_file = output_path / f"{prefix}_all_steps.csv"
    seeds_file = output_path / f"{prefix}_seed_summary.csv"
    conditions_file = output_path / f"{prefix}_condition_summary.csv"
    summary_file = output_path / f"{prefix}_run_summary.json"
    debug_file = output_path / f"grid_{prefix}_debug_trace.jsonl"

    df_trials.to_csv(trials_file, index=False)
    if all_steps:
        df_steps = pd.DataFrame(all_steps)
        df_steps.to_csv(steps_file, index=False)
        print(f"Saved {len(df_steps)} steps to {steps_file}")
    df_seeds.to_csv(seeds_file, index=False)
    df_conditions.to_csv(conditions_file, index=False)

    if all_debug_rows:
        with open(debug_file, 'w', encoding='utf-8') as f:
            for row in all_debug_rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f"Saved {len(all_debug_rows)} debug rows to {debug_file}")

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
        diagnostic_suffix = "_forced" if args.diagnostic_forced_shock else ""
        param_suffix = (
            format_param_tag("wqni", args.w_qneg_input_override)
            + format_param_tag("wqpi", args.w_qpos_input_override)
            + format_param_tag("kpos", args.k_pos_override)
        )
        run_label = f"{condition_name}_{args.ablation}{diagnostic_suffix}{param_suffix}"
        output_dir = build_timestamped_output_dir(args.output_dir, run_label)

        print(f"\nOutput directory: {output_dir}")

        run_single_condition(
            condition_name=condition_name,
            n_seeds=args.n_seeds,
            n_trials=args.n_trials,
            ablation=args.ablation,
            output_dir=output_dir,
            verbose=args.verbose,
            diagnostic_forced_shock=args.diagnostic_forced_shock,
            forced_shock_path=args.forced_shock_path,
            debug=args.debug,
            debug_console=args.debug_console,
            debug_console_start=args.debug_console_start,
            debug_console_end=args.debug_console_end,
            debug_junction_only=args.debug_junction_only,
            debug_trial_window=args.debug_trial_window,
            w_qneg_input_override=args.w_qneg_input_override,
            w_qpos_input_override=args.w_qpos_input_override,
            k_pos_override=args.k_pos_override,
        )

    elif args.grid:
        diagnostic_suffix = "_forced" if args.diagnostic_forced_shock else ""
        param_suffix = (
            format_param_tag("wqni", args.w_qneg_input_override)
            + format_param_tag("wqpi", args.w_qpos_input_override)
            + format_param_tag("kpos", args.k_pos_override)
        )
        run_label = f"grid_{args.ablation}{diagnostic_suffix}{param_suffix}"
        output_dir = build_timestamped_output_dir(args.output_dir, run_label)

        print(f"\nOutput directory: {output_dir}")

        run_grid(
            grid_type=args.grid,
            n_seeds=args.n_seeds,
            n_trials=args.n_trials,
            ablation=args.ablation,
            output_dir=output_dir,
            verbose=args.verbose,
            diagnostic_forced_shock=args.diagnostic_forced_shock,
            forced_shock_path=args.forced_shock_path,
            debug=args.debug,
            debug_console=args.debug_console,
            debug_console_start=args.debug_console_start,
            debug_console_end=args.debug_console_end,
            debug_junction_only=args.debug_junction_only,
            debug_trial_window=args.debug_trial_window,
            w_qneg_input_override=args.w_qneg_input_override,
            w_qpos_input_override=args.w_qpos_input_override,
            k_pos_override=args.k_pos_override,
        )

    elif args.one_shot:
        one_shot_protocol = get_one_shot_protocol(
            kind=args.one_shot_kind,
            path=args.one_shot_path_override or 'open',
            reward=args.one_shot_reward_override,
            salience=args.one_shot_salience_override if args.one_shot_salience_override is not None else 0.9,
            stakes=args.one_shot_stakes_override if args.one_shot_stakes_override is not None else 10.0,
        )

        force_event = args.diagnostic_forced_shock or args.diagnostic_forced_treat
        forced_path = args.forced_shock_path
        if forced_path is None and args.diagnostic_forced_treat:
            forced_path = args.one_shot_path_override or 'open'

        diagnostic_suffix = "_forced" if force_event else ""
        param_suffix = (
            format_param_tag("wqni", args.w_qneg_input_override)
            + format_param_tag("wqpi", args.w_qpos_input_override)
            + format_param_tag("kpos", args.k_pos_override)
        )
        run_label = f"one_shot_{args.one_shot_kind}_{args.ablation}{diagnostic_suffix}{param_suffix}"
        output_dir = build_timestamped_output_dir(args.output_dir, run_label)

        print(f"\nOutput directory: {output_dir}")

        run_one_shot_protocol(
            n_seeds=args.n_seeds,
            n_trials=args.n_trials,
            ablation=args.ablation,
            output_dir=output_dir,
            verbose=args.verbose,
            diagnostic_forced_shock=force_event,
            forced_shock_path=forced_path,
            debug=args.debug,
            debug_console=args.debug_console,
            debug_console_start=args.debug_console_start,
            debug_console_end=args.debug_console_end,
            debug_junction_only=args.debug_junction_only,
            debug_trial_window=args.debug_trial_window,
            w_qneg_input_override=args.w_qneg_input_override,
            w_qpos_input_override=args.w_qpos_input_override,
            k_pos_override=args.k_pos_override,
            one_shot_protocol=one_shot_protocol
        )


if __name__ == '__main__':
    main()
