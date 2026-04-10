"""
Stage 3.1B: Threat / Reward Conflict Configuration.

Расширение Stage 3.1A с добавлением:
- Unequal reward + unequal threat/exposure
- 3x3 matrix условий (reward_level × threat_level)
- One-shot protocol с pre/post shock блоками
- Conflict-aware logging и анализ

Design Principles:
- Использовать frozen baseline из 3.1A (exposure_q_bias = 0.35, eps = 0.05)
- Не переписывать среду, а расширять конфигом
- Сохранить logging contract из 3.1A + добавить conflict variables
- One-shot persistence через TemporalState (не отдельный trauma module)

Usage:
    from stage3.configs.config_stage3_1b import CONFIG_3_1B
    from stage3.configs.config_stage3_1b import get_condition_grid, get_canonical_conditions
    agent = AgentStage3(CONFIG_3_1B['agent'], seed=42)
    env = OpenCoveredChoiceEnv(CONFIG_3_1B['env'], seed=42)

Author: Alex Snow (Aleksey L. Snigirov)
License: MIT
"""

from typing import Dict, Any, List, Tuple
import copy


# =============================================================================
# BASELINE CONFIG FROM STAGE 3.1A (FROZEN)
# =============================================================================

ENV_CONFIG_BASELINE: Dict[str, Any] = {
    # === Basic Parameters ===
    'maze_type': 'open_covered_choice',
    'n_trials': 100,
    'n_sessions': 30,
    'seed_range': (42, 71),
    
    # === Maze Topology ===
    'topology': {
        'n_nodes': 5,
        'node_ids': ['start', 'junction', 'open_mid', 'covered_mid', 'goal'],
        'start_node': 'start',
        'goal_node': 'goal',
        'junction_node': 'junction',
    },
    
    # === Zone Parameters (from 3.1A) ===
    'zones': {
        'start': {
            'exposure': {'X_risk': 0.1, 'X_opp': 0.0, 'D_est': 0.5},
            'valence': {'nu': 0.0, 'stakes': 1.0}
        },
        'junction': {
            'exposure': {'X_risk': 0.3, 'X_opp': 0.3, 'D_est': 0.8},
            'valence': {'nu': 0.0, 'stakes': 1.0},
            'is_choice_point': True
        },
        'goal': {
            'exposure': {'X_risk': 0.1, 'X_opp': 1.0, 'D_est': 0.5},
            'valence': {'nu': 1.0, 'stakes': 1.0}
        }
    },
    
    # === Agent Starting Conditions ===
    'agent_start': {
        'initial_node': 'start',
        'initial_heading': 0.0,
        'initial_mode': 'EXPLOIT'
    },
    
    # === Temporal Parameters ===
    'temporal': {
        'tick_duration': 1.0,
        'junction_pause_min': 1,
        'junction_pause_max': 10,
        'inter_trial_interval': 5
    },
    
    # === Task 3: World Stochasticity ===
    'observability_noise_std': 0.0,
    
    # === Task 4: Deliberation Parameters (FROZEN from 3.1A) ===
    'debug': False,
    'deliberation': {
        'max_deliberation_ticks': 6,
        'junction_q_values': [0.5, 0.5],
        'exposure_q_bias': 0.35,  # FROZEN
        'eps': 0.05,              # FROZEN
        
        # Evidence accumulation
        'evidence_bound_base': 0.35,
        'evidence_bound_min': 0.20,
        'urgency_slope': 0.04,
        'min_evidence_step': 0.10,

        # Stage 3.1B reward/threat-conditioned Q shaping
        'reward_q_scale': 1.50,
        'threat_q_scale': 0.50,
        'risk_q_scale': 0.25,
    },
}

# =============================================================================
# PATH PARAMETERS: COVERED (FIXED BASELINE)
# =============================================================================

COVERED_PATH_FIXED: Dict[str, Any] = {
    'length': 3,
    'base_reward': 1.0,
    'reward_prob': 0.70,      # Fixed baseline
    'reward_bonus': 0.0,
    'threat_penalty': 0.0,
    'threat_prob': 0.0,
    'exposure_profile': {
        'X_risk': 0.20,       # Low threat
        'X_opp': 0.50,
        'D_est': 0.30         # Low visibility
    }
}


# =============================================================================
# PATH PARAMETERS: OPEN (VARIABLE BY CONDITION)
# =============================================================================

# Reward levels (3 levels)
OPEN_REWARD_LEVELS = {
    'R0': {'reward_prob': 0.70, 'reward_bonus': 0.0},    # No premium
    'R1': {'reward_prob': 0.80, 'reward_bonus': 0.2},    # Moderate premium
    'R2': {'reward_prob': 0.90, 'reward_bonus': 0.4},    # High premium
}

# Threat/risk levels (3 levels)
OPEN_THREAT_LEVELS = {
    'T1': {'X_risk': 0.45, 'threat_penalty': 0.0, 'threat_prob': 0.0},   # Mild threat
    'T2': {'X_risk': 0.60, 'threat_penalty': 0.5, 'threat_prob': 0.3},   # Balanced threat
    'T3': {'X_risk': 0.75, 'threat_penalty': 1.0, 'threat_prob': 0.5},   # Strong threat
}

# Fixed open visibility/opportunity
OPEN_FIXED: Dict[str, Any] = {
    'length': 3,
    'base_reward': 1.0,
    'X_opp': 0.50,
    'D_est': 0.90
}


# =============================================================================
# MATRIX CONFIG: 3x3 GRID
# =============================================================================

def get_condition_grid() -> Dict[str, Dict[str, Any]]:
    """
    Генерирует полную 3x3 матрицу условий.
    
    Returns:
        condition_grid: Dict[condition_id, path_config]
    """
    condition_grid = {}
    
    for reward_key, reward_params in OPEN_REWARD_LEVELS.items():
        for threat_key, threat_params in OPEN_THREAT_LEVELS.items():
            condition_id = f"{reward_key}_{threat_key}"
            
            open_path = {
                'length': OPEN_FIXED['length'],
                'base_reward': OPEN_FIXED['base_reward'],
                'reward_prob': reward_params['reward_prob'],
                'reward_bonus': reward_params['reward_bonus'],
                'threat_penalty': threat_params['threat_penalty'],
                'threat_prob': threat_params['threat_prob'],
                'exposure_profile': {
                    'X_risk': threat_params['X_risk'],
                    'X_opp': OPEN_FIXED['X_opp'],
                    'D_est': OPEN_FIXED['D_est']
                }
            }
            
            condition_grid[condition_id] = {
                'condition_id': condition_id,
                'reward_level': reward_key,
                'threat_level': threat_key,
                'paths': {
                    'open': open_path,
                    'covered': copy.deepcopy(COVERED_PATH_FIXED)
                }
            }
    
    return condition_grid


def get_canonical_conditions() -> Dict[str, Dict[str, Any]]:
    """
    Возвращает три канонические условия Stage 3.1B.

    Эти condition IDs фиксированы ТЗ и используются:
    - для smoke checks,
    - для ablations,
    - для acceptance protocol.
    """
    return {
        'reward_dominant': {
            'condition_id': 'R2_T1',
            'conflict_condition': 'reward_dominant',
            'description': 'High reward premium on open + mild threat',
            'open_reward_prob': 0.90,
            'open_X_risk': 0.45
        },
        'balanced_conflict': {
            'condition_id': 'R1_T2',
            'conflict_condition': 'balanced_conflict',
            'description': 'Moderate reward premium on open + balanced threat',
            'open_reward_prob': 0.80,
            'open_X_risk': 0.60
        },
        'threat_dominant': {
            'condition_id': 'R0_T3',
            'conflict_condition': 'threat_dominant',
            'description': 'No reward premium on open + strong threat',
            'open_reward_prob': 0.70,
            'open_X_risk': 0.75
        }
    }


# =============================================================================
# ONE-SHOT PROTOCOL CONFIG
# =============================================================================

# ВАЖНО:
# - one-shot НЕ должен быть активен в обычных condition/grid runs
# - он включается только в отдельном one-shot protocol

ONE_SHOT_DISABLED: Dict[str, Any] = {
    "one_shot_enabled": False,
    "one_shot_trial": -1,
    "one_shot_path": "",
    "one_shot_reward": 0.0,
    "one_shot_salience": 0.0,
    "one_shot_stakes": 0.0,
}

ONE_SHOT_CONFIG: Dict[str, Any] = {
    "one_shot_enabled": True,
    "one_shot_trial": 30,
    "one_shot_path": "open",
    "one_shot_reward": -5.0,
    "one_shot_salience": 0.9,
    "one_shot_stakes": 10.0,
}


def get_one_shot_protocol() -> Dict[str, Any]:
    """
    Возвращает one-shot protocol structure.

    ВАЖНО:
    protocol сам по себе включает one-shot,
    но обычные build_env_config_for_condition() должны оставаться one-shot OFF.
    """
    return {
        "pre_block_trials": 30,
        "shock_trial": 30,
        "post_block_trials": 69,
        "total_trials": 100,
        "condition_name": "balanced_conflict",
        "condition_id": "R1_T2",
        "one_shot": copy.deepcopy(ONE_SHOT_CONFIG),
    }


# =============================================================================
# ENVIRONMENT CONFIGURATION (3.1B)
# =============================================================================

ENV_CONFIG_3_1B: Dict[str, Any] = {
    **ENV_CONFIG_BASELINE,
    
    # === Path Parameters (будут overridden by condition) ===
    'paths': {
        'open': {
            **OPEN_FIXED,
            'reward_prob': 0.80,  # Default: balanced conflict
            'reward_bonus': 0.2,
            'threat_penalty': 0.5,
            'threat_prob': 0.3,
            'exposure_profile': {
                'X_risk': 0.60,  # Default: balanced threat
                'X_opp': 0.50,
                'D_est': 0.90
            }
        },
        'covered': COVERED_PATH_FIXED
    },
    
    # === One-Shot Protocol ===
    # По умолчанию выключен. Включается только отдельным one-shot runner mode.
    'one_shot': copy.deepcopy(ONE_SHOT_DISABLED),
    
    # === Condition Metadata ===
    'condition_id': 'R1_T2',  # Default: balanced conflict
    'reward_level': 'R1',
    'threat_level': 'T2',
}


# =============================================================================
# AGENT CONFIGURATION (SAME AS 3.1A)
# =============================================================================

AGENT_CONFIG_3_1B: Dict[str, Any] = {
    # === Backward Compatibility ===
    'compatibility_mode': False,
    'log_level': 2,
    
    # === Stage 2 Legacy ===
    'stage2_legacy': {
        'alpha': 0.35,
        'beta': 4.0,
        'k_use': 0.08,
        'k_melt': 0.20,
        'lambda_decay': 0.01,
        'tau_vol': 0.50,
    },
    
    # === Stage 3 Action Policy ===
    'action_policy': {
        'beta_exploit': 4.0,
        'beta_explore': 1.0,
        'beta_safe': 5.0,
        'lambda_risk': 2.0,
        'epsilon_explore': 0.0,
        'commit_confidence': 0.7,
        'max_deliberation_ticks': 10,
    },
    
    # === Stage 3 Core: Gate Thresholds ===
    'gate_thresholds': {
        'critical_risk_threshold': 0.42,
        'suspicion_threshold': 0.5,
        'visibility_threshold': 0.3,
        'safe_window_threshold': 50,
        'theta_mb': 0.30,
        'theta_u': 1.5,

        # Patch B: current + accumulated threat
        'safe_drive_weight_current': 0.6,
        'safe_drive_weight_temporal': 0.4,

        'w_volatility': 1.0,
        'w_entropy': 1.0,
        'v_g_weight_hrisk': 0.7,
        'v_g_weight_xrisk': 0.3,
    },
    
    # === Stage 3 Core: Temporal State ===
    'temporal_state': {
        'lambda_risk': 0.10,   # legacy / fallback
        'lambda_opp': 0.90,

        # Patch A: asymmetric risk trace
        'lambda_risk_in': 0.10,
        'lambda_risk_out': 0.98,
        'one_shot_decay_override': 0.995,
        'one_shot_persistence_window': 10,
        'one_shot_floor': 0.15,

        'salience_threshold': 0.5,
        'one_shot_threshold': 5.0,
        'one_shot_boost': 2.0,
    },
    
    # === Stage 3 Core: Exposure Field ===
    'exposure_field': {
        'valence_scale': 1.0,
        'observability_scale': 1.0,
        'risk_threshold': 0.5,
        'opportunity_threshold': 0.5,
    },
}


# =============================================================================
# LOGGING CONFIGURATION (EXTENDED FOR 3.1B)
# =============================================================================

LOGGING_CONFIG_3_1B: Dict[str, Any] = {
    # === Output Paths ===
    'output_dir': 'logs/stage3/stage3_1b/',
    'log_format': 'csv',
    'save_frequency': 1,
    
    # === Step Log Fields (3.1A mandatory + 3.1B extensions) ===
    'step_log_fields': [
        'seed',
        'trial',
        'tick',
        'node_id',
        'edge_id',
        'at_junction',
        'deliberation_state',
        'candidate_path',
        'committed_path',
        'mode_before',
        'mode_after',
        'gate_trigger',
        'action',
        'reward',
        'salience',
        'stakes',
        'u_delta',
        'u_entropy',
        'u_volatility',
        'X_risk',
        'X_opp',
        'D_est',
        'h_risk',
        'h_opp',
        'h_time',
        'one_shot_fired',
        'q_values',
        'risk_values',
        'action_probs',
        'sampled_action',
        # === NEW: 3.1B Conflict Variables ===
        'condition_id',
        'reward_level',
        'threat_level',
        'reward_gap',
        'risk_gap',
        'threat_gap',
        'conflict_condition',
        'open_reward_prob',
        'covered_reward_prob',
        'open_X_risk',
        'covered_X_risk',
        'open_threat_penalty',
        'covered_threat_penalty',
        'one_shot_active',
        'one_shot_trial',
        'one_shot_path',
    ],
    
    # === Trial Summary Log Fields (3.1A mandatory + 3.1B extensions) ===
    'trial_log_fields': [
        'seed',
        'trial',
        'path_choice',
        'reward_total',
        'junction_pause_duration',
        'reorientation_count',
        'retreat_return_count',
        'commit_latency',
        'commit_reason',
        'junction_deliberation_proxy',
        'mode_at_junction',
        'final_mode',
        'one_shot_fired',
        'config_name',
        'ablation_name',
        # === NEW: 3.1B Conflict Variables ===
        'condition_id',
        'reward_level',
        'threat_level',
        'reward_gap',
        'risk_gap',
        'threat_gap',
        'conflict_condition',
        'one_shot_active',
        'one_shot_trial',
        'path_choice_preferred_by_reward',
        'path_choice_preferred_by_threat',
    ],
    
    # === Optional Pseudo-Kinematic Log ===
    'kinematic_log_fields': [
        'heading_state',
        'heading_change',
        'candidate_heading_switches',
        'zone_entry_tick',
        'zone_exit_tick'
    ],
    
    # === Downward Compatibility ===
    'stage2_compat': {
        'include_stage2_fields': True,
        'field_mapping': {
            'trial': 'trial',
            'action': 'a1',
            'reward': 'reward',
            'mode_after': 'mode'
        }
    }
}


# =============================================================================
# ABLATION CONFIGURATION (SAME AS 3.1A)
# =============================================================================

ABLATION_CONFIG_3_1B: Dict[str, Dict[str, Any]] = {
    'full': {
        'name': 'Full',
        'description': 'Все компоненты активны',
        'modifications': {}
    },
    'novg': {
        'name': 'NoVG',
        'description': 'V_G ≡ 0 (control-mode inertia removed)',
        'modifications': {
            'agent': {
                'temporal_state': {
                    'h_risk': 0.0,
                    'h_opp': 0.0
                }
            }
        }
    },
    'novp': {
        'name': 'NoVp',
        'description': 'V_p ≡ 0 (action-level inertia removed)',
        'modifications': {
            'agent': {
                'viscosity': {
                    'k_use': 0.0,
                    'k_melt': 0.0
                }
            }
        }
    },
    'nox': {
        'name': 'NoX-to-Gate',
        'description': 'ExposureAggregates.zeros() (exposure field отключен)',
        'modifications': {
            'agent': {
                'exposure_field': {
                    'zero_output': True
                }
            }
        }
    },
    'one_shot_off': {
        'name': 'One-Shot-Off',
        'description': 'One-shot update отключен (amplitude cap)',
        'modifications': {
            'agent': {
                'temporal_state': {
                    'one_shot_threshold': 999.0,
                    'one_shot_boost': 0.0,
                    'one_shot_persistence_window': 0,
                    'one_shot_floor': 0.0
                }
            }
        }
    }
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def classify_conflict_condition(condition_id: str) -> str:
    """
    Возвращает semantic label для canonical conditions.
    Для остальных условий возвращает 'mixed'.
    """
    canonical = get_canonical_conditions()

    for _, data in canonical.items():
        if data['condition_id'] == condition_id:
            return data['conflict_condition']

    return 'mixed'

def compute_conflict_variables(open_path: Dict, covered_path: Dict) -> Dict[str, Any]:
    """
    Вычисляет производные конфликтные переменные.

    Здесь НЕ классифицируем canonical condition эвристикой.
    Canonical label задаётся отдельно по condition_id.
    """

    # Expected positive reward values
    # Используем reward_prob * (base_reward + reward_bonus)
    # как рабочее приближение expected positive reward.
    e_r_open = (
        open_path.get('base_reward', 1.0) + open_path.get('reward_bonus', 0.0)
    ) * open_path.get('reward_prob', 1.0)

    e_r_covered = (
        covered_path.get('base_reward', 1.0) + covered_path.get('reward_bonus', 0.0)
    ) * covered_path.get('reward_prob', 1.0)

    # Expected threat cost
    e_t_open = open_path.get('threat_penalty', 0.0) * open_path.get('threat_prob', 0.0)
    e_t_covered = covered_path.get('threat_penalty', 0.0) * covered_path.get('threat_prob', 0.0)

    # Exposure-derived risk gap
    x_risk_open = open_path.get('exposure_profile', {}).get('X_risk', 0.0)
    x_risk_covered = covered_path.get('exposure_profile', {}).get('X_risk', 0.0)

    reward_gap = e_r_open - e_r_covered
    risk_gap = x_risk_open - x_risk_covered
    threat_gap = e_t_open - e_t_covered

    preferred_by_reward = 'open' if reward_gap > 0 else 'covered'
    preferred_by_threat = 'covered' if risk_gap > 0 else 'open'

    return {
        'reward_gap': reward_gap,
        'risk_gap': risk_gap,
        'threat_gap': threat_gap,
        'preferred_by_reward': preferred_by_reward,
        'preferred_by_threat': preferred_by_threat,
    }

def build_env_config_for_condition(
    condition_id: str,
    one_shot_override: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Строит полный env config для конкретного условия.

    ВАЖНО:
    - обычный condition build -> one-shot OFF
    - отдельный one-shot protocol -> one_shot_override
    """
    condition_grid = get_condition_grid()

    if condition_id not in condition_grid:
        raise ValueError(f"Unknown condition_id: {condition_id}")

    condition_data = condition_grid[condition_id]
    paths = condition_data['paths']

    conflict_vars = compute_conflict_variables(paths['open'], paths['covered'])
    conflict_condition = classify_conflict_condition(condition_id)

    env_config = {
        **copy.deepcopy(ENV_CONFIG_BASELINE),
        'paths': copy.deepcopy(paths),
        'condition_id': condition_id,
        'reward_level': condition_data['reward_level'],
        'threat_level': condition_data['threat_level'],
        'conflict_vars': {
            **conflict_vars,
            'conflict_condition': conflict_condition,
        },
        'one_shot': copy.deepcopy(ONE_SHOT_DISABLED),
    }

    if one_shot_override is not None:
        env_config['one_shot'] = copy.deepcopy(one_shot_override)

    return env_config


# =============================================================================
# MAIN CONFIG DICT
# =============================================================================

CONFIG_3_1B: Dict[str, Any] = {
    'env': copy.deepcopy(ENV_CONFIG_3_1B),
    'agent': AGENT_CONFIG_3_1B,
    'logging': LOGGING_CONFIG_3_1B,
    'ablation': ABLATION_CONFIG_3_1B,
    'matrix': get_condition_grid(),
    'canonical': get_canonical_conditions(),
    'one_shot_protocol': get_one_shot_protocol(),
}


# =============================================================================
# ACCEPTANCE CRITERIA
# =============================================================================

ACCEPTANCE_CRITERIA: Dict[str, Dict[str, Any]] = {
    '16.1_path_tradeoff': {
        'description': 'P(open) и P(covered) систематически меняются по матрице reward × threat',
        'metric': 'path_choice_variance_across_conditions',
        'threshold': 0.1,  # Минимальная дисперсия
        'test': 'assert_gt'
    },
    '16.2_balanced_conflict_max_deliberation': {
        'description': 'В balanced-conflict deliberation proxies максимальны',
        'metric': 'junction_deliberation_proxy_balanced_vs_others',
        'threshold': 1.0,  # Должен быть выше
        'test': 'assert_gt'
    },
    '16.3_gate_not_constant': {
        'description': 'mode_at_junction не является константой',
        'metric': 'mode_at_junction_entropy',
        'threshold': 0.5,  # Минимальная энтропия
        'test': 'assert_gt'
    },
    '16.4_one_shot_persistence': {
        'description': 'Post-shock shift в P(open) и h_risk',
        'metric': 'path_choice_pre_post_shift',
        'threshold': 0.15,  # Минимальный сдвиг
        'test': 'assert_gt'
    },
    '16.5_no_architecture_regression': {
        'description': 'Сохранены: one Gate, no argmax routing, deterministic execution',
        'metric': 'architecture_invariants',
        'threshold': 1.0,  # Все инварианты должны выполняться
        'test': 'assert_eq'
    }
}
