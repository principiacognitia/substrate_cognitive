"""
Stage 3.1A: Open/Covered Choice Maze Configuration.

Базовая версия пространственной задачи с центральной развилкой.
Без threat/reward conflict (только exposure difference).

Design Principles:
- Graph-based topology (5 nodes minimum)
- Exposure field как отдельный слой (не categorical labels)
- VTE proxies логируются для future IdPhi wrapper
- Downward compatibility со Stage 2 logging format
- NoX-to-Gate абляция = нулевые ExposureAggregates

Usage:
    from stage3.configs.config_stage3_1a import CONFIG_3_1A
    agent = AgentStage3(CONFIG_3_1A['agent'])
    env = OpenCoveredChoiceEnv(CONFIG_3_1A['env'])

Author: Alex Snow (Aleksey L. Snigirov)
License: MIT
"""

from typing import Dict, Any, List


# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

ENV_CONFIG: Dict[str, Any] = {
    # === Basic Parameters ===
    'maze_type': 'open_covered_choice',  # Тип лабиринта
    'n_trials': 100,                     # Количество триалов на сессию
    'n_sessions': 30,                    # Количество сессий (seeds)
    'seed_range': (42, 71),              # Диапазон seeds (30 seeds)
    
    # === Maze Topology ===
    'topology': {
        'n_nodes': 5,                    # Минимальная конфигурация
        'node_ids': ['start', 'junction', 'open_mid', 'covered_mid', 'goal'],
        'start_node': 'start',
        'goal_node': 'goal',
        'junction_node': 'junction',
    },
    
    # === Path Parameters ===
    'paths': {
        'open': {
            'length': 3,                 # Длина пути (в тиках)
            'base_reward': 1.0,          # Базовая награда
            'exposure_profile': {        # Exposure profile для open path
                'X_risk': 0.7,           # Высокая экспозиция
                'X_opp': 0.5,
                'D_est': 0.9             # Высокая видимость
            }
        },
        'covered': {
            'length': 3,                 # Длина пути (одинаковая с open)
            'base_reward': 1.0,          # Одинаковая награда (3.1A)
            'exposure_profile': {        # Exposure profile для covered path
                'X_risk': 0.2,           # Низкая экспозиция
                'X_opp': 0.5,
                'D_est': 0.3             # Низкая видимость
            }
        }
    },
    
    # === Zone Parameters ===
    'zones': {
        'start': {
            'exposure': {'X_risk': 0.1, 'X_opp': 0.0, 'D_est': 0.5},
            'valence': {'nu': 0.0, 'stakes': 1.0}
        },
        'junction': {
            'exposure': {'X_risk': 0.3, 'X_opp': 0.3, 'D_est': 0.8},
            'valence': {'nu': 0.0, 'stakes': 1.0},
            'is_choice_point': True      # Флаг для VTE logging
        },
        'goal': {
            'exposure': {'X_risk': 0.1, 'X_opp': 1.0, 'D_est': 0.5},
            'valence': {'nu': 1.0, 'stakes': 1.0}
        }
    },
    
    # === Agent Starting Conditions ===
    'agent_start': {
        'initial_node': 'start',
        'initial_heading': 0.0,          # Начальное направление (для VTE)
        'initial_mode': 'EXPLOIT'        # Начальный режим Gate
    },
    
    # === Temporal Parameters ===
    'temporal': {
        'tick_duration': 1.0,            # Длительность одного тика (сек)
        'junction_pause_min': 1,         # Минимальная пауза на junction
        'junction_pause_max': 10,        # Максимальная пауза на junction
        'inter_trial_interval': 5        # Интервал между триалами (тики)
    }
}


# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

AGENT_CONFIG: Dict[str, Any] = {
    # === Gate Thresholds (Stage 3.0) ===
    'gate_thresholds': {
        'critical_risk_threshold': 0.7,    # Порог для EXPLOIT_SAFE
        'suspicion_threshold': 0.5,        # Порог для ABSENCE_CHECK
        'visibility_threshold': 0.3,       # Максимальная D_est для ABSENCE_CHECK
        'safe_window_threshold': 50,       # Минимальный h_time для ABSENCE_CHECK
        'theta_mb': 0.30,                  # Mode switch threshold (Stage 2)
        'theta_u': 1.5,                    # Uncertainty baseline (Stage 2)
    },
    
    # === Temporal State Config ===
    'temporal_state': {
        'lambda_risk': 0.9,                # Decay rate для h_risk
        'lambda_opp': 0.9,                 # Decay rate для h_opp
        'salience_threshold': 0.5,         # Порог для сброса h_time
        'one_shot_threshold': 5.0,         # Порог амплитуды для one-shot
        'one_shot_boost': 2.0,             # Множитель для one-shot update
    },
    
    # === Exposure Field Config ===
    'exposure_field': {
        'valence_scale': 1.0,              # Масштаб для валентности
        'observability_scale': 1.0,        # Масштаб для наблюдаемости
        'risk_threshold': 0.5,             # Порог для X_risk агрегации
        'opportunity_threshold': 0.5,      # Порог для X_opp агрегации
    },
    
    # === Stage 2 Compatibility ===
    'compatibility_mode': False,           # False = Stage 3 режим
    'log_level': 2,                        # 0=none, 1=summary, 2=full
    
    # === Viscosity Parameters (Stage 2, для совместимости) ===
    'viscosity': {
        'alpha': 0.35,                     # Learning rate
        'beta': 4.0,                       # Inverse softmax temperature
        'k_use': 0.08,                     # Hardening rate
        'k_melt': 0.20,                    # Melting rate
        'lambda_decay': 0.01,              # Decay rate
        'tau_vol': 0.50,                   # Volatility threshold
    }
}


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG: Dict[str, Any] = {
    # === Output Paths ===
    'output_dir': 'logs/stage3/stage3_1a/',
    'log_format': 'csv',                   # Формат логов
    'save_frequency': 1,                   # Сохранять каждый триал
    
    # === Step Log Fields (mandatory) ===
    'step_log_fields': [
        'seed',
        'trial',
        'tick',
        'node_id',
        'edge_id',
        'at_junction',
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
        'one_shot_fired'
    ],
    
    # === Trial Summary Log Fields (mandatory) ===
    'trial_log_fields': [
        'seed',
        'trial',
        'path_choice',                      # 'open' или 'covered'
        'reward_total',
        'junction_pause_duration',          # VTE proxy 1
        'reorientation_count',              # VTE proxy 2
        'retreat_return_count',             # VTE proxy 3
        'commit_latency',                   # VTE proxy 4
        'junction_deliberation_proxy',      # Optional composite
        'mode_at_junction',
        'final_mode',
        'one_shot_fired',
        'config_name',
        'ablation_name'
    ],
    
    # === Optional Pseudo-Kinematic Log (для future IdPhi wrapper) ===
    'kinematic_log_fields': [
        'heading_state',
        'heading_change',
        'candidate_heading_switches',
        'zone_entry_tick',
        'zone_exit_tick'
    ],
    
    # === Downward Compatibility with Stage 2 ===
    'stage2_compat': {
        'include_stage2_fields': True,     # Включить поля Stage 2
        'field_mapping': {                 # Mapping имён полей
            'trial': 'trial',
            'action': 'a1',
            'reward': 'reward',
            'mode_after': 'mode'
        }
    }
}


# =============================================================================
# ABLATION CONFIGURATION
# =============================================================================

ABLATION_CONFIG: Dict[str, Dict[str, Any]] = {
    # === Full Agent (все включено) ===
    'full': {
        'name': 'Full',
        'description': 'Все компоненты активны',
        'modifications': {}
    },
    
    # === NoVG (V_G = 0) ===
    'novg': {
        'name': 'NoVG',
        'description': 'V_G ≡ 0 (control-mode inertia removed)',
        'modifications': {
            'agent': {
                'temporal_state': {
                    'h_risk': 0.0,         # Принудительно нулевой
                    'h_opp': 0.0
                }
            }
        }
    },
    
    # === NoVp (V_p = 0) ===
    'novp': {
        'name': 'NoVp',
        'description': 'V_p ≡ 0 (action-level inertia removed)',
        'modifications': {
            'agent': {
                'viscosity': {
                    'k_use': 0.0,          # Отключить hardening
                    'k_melt': 0.0          # Отключить melting
                }
            }
        }
    },
    
    # === NoX-to-Gate (ExposureAggregates = zeros) ===
    'nox': {
        'name': 'NoX-to-Gate',
        'description': 'ExposureAggregates.zeros() (exposure field отключен)',
        'modifications': {
            'agent': {
                'exposure_field': {
                    'zero_output': True    # Возвращать нулевые aggregates
                }
            }
        }
    },
    
    # === One-Shot Off (amplitude cap) ===
    'one_shot_off': {
        'name': 'One-Shot-Off',
        'description': 'One-shot update отключен (amplitude cap)',
        'modifications': {
            'agent': {
                'temporal_state': {
                    'one_shot_threshold': 999.0,  # Никогда не срабатывает
                    'one_shot_boost': 0.0
                }
            }
        }
    }
}


# =============================================================================
# VTE PROXY CONFIGURATION
# =============================================================================

VTE_CONFIG: Dict[str, Any] = {
    # === Primary VTE Metrics (для статистики) ===
    'primary_metrics': [
        'junction_pause_duration',        # Тиков на junction до выбора
        'reorientation_count',            # Смен candidate_path на junction
        'retreat_return_count',           # Возвратов в start после junction
        'commit_latency'                  # Тиков до окончательного commit
    ],
    
    # === Secondary Composite (только для графиков) ===
    'composite_metric': {
        'name': 'junction_deliberation_proxy',
        'method': 'zscore_mean',         # Z-scored mean всех primary metrics
        'use_for_stats': False,          # Не использовать для статистики
        'use_for_plots': True            # Использовать для visualizations
    },
    
    # === Junction Detection ===
    'junction_detection': {
        'node_id': 'junction',
        'entry_threshold': 1,            # Тик входа в junction
        'exit_threshold': 1,             # Тик выхода из junction
        'pause_min': 1,                  # Минимальная пауза для VTE
        'pause_max': 100                 # Максимальная пауза (outlier cap)
    },
    
    # === Future IdPhi Wrapper (для Stage 3.2) ===
    'idphi_stub': {
        'enabled': False,                # Отключено в 3.1A
        'sampling_rate': 10.0,           # Hz (для future continuous tracking)
        'heading_smoothing': 0.9         # EMA для heading trace
    }
}


# =============================================================================
# STATISTICAL ANALYSIS CONFIGURATION
# =============================================================================

STATS_CONFIG: Dict[str, Any] = {
    # === Primary Comparisons ===
    'comparisons': [
        {
            'name': 'Full vs NoVG (path choice)',
            'group1': 'full',
            'group2': 'novg',
            'metric': 'P(open_path)',
            'test': 'chi_square',
            'alpha': 0.017               # Bonferroni-corrected
        },
        {
            'name': 'Full vs NoX (VTE proxy)',
            'group1': 'full',
            'group2': 'nox',
            'metric': 'junction_pause_duration',
            'test': 'mann_whitney_u',
            'alpha': 0.017
        },
        {
            'name': 'Full vs One-Shot-Off (persistence)',
            'group1': 'full',
            'group2': 'one_shot_off',
            'metric': 'path_choice_post_aversive',
            'test': 'chi_square',
            'alpha': 0.017
        }
    ],
    
    # === Multiple Comparison Correction ===
    'correction': {
        'method': 'bonferroni',
        'n_comparisons': 3,
        'alpha_uncorrected': 0.05,
        'alpha_corrected': 0.017
    },
    
    # === Effect Size Metrics ===
    'effect_sizes': [
        'rank_biserial',                # Для Mann-Whitney U
        'cohens_d',                     # Для t-tests
        'cramers_v'                     # Для chi-square
    ]
}


# =============================================================================
# ACCEPTANCE CRITERIA
# =============================================================================

ACCEPTANCE_CRITERIA: Dict[str, Dict[str, Any]] = {
    '10.1_spatialization_works': {
        'description': 'Агент стабильно проходит spatial env без логических поломок',
        'metric': 'completion_rate',
        'threshold': 0.95,               # 95% триалов завершаются успешно
        'test': 'assert_ge'
    },
    '10.2_exposure_sensitive_choice': {
        'description': 'При повышенном X_risk open path агент смещается к covered path',
        'metric': 'P(covered_path) - P(open_path)',
        'threshold': 0.2,                # Минимум 20% смещение
        'test': 'assert_ge'
    },
    '10.3_vte_proxy_at_junction': {
        'description': 'В условиях exposure conflict растёт junction_pause_duration',
        'metric': 'junction_pause_duration (high conflict) - junction_pause_duration (low conflict)',
        'threshold': 2.0,                # Минимум 2 тика разница
        'test': 'assert_ge'
    },
    '10.4_reward_exposure_dissociation': {
        'description': 'При низкой reward uncertainty, но высокой exposure conflict VTE сохраняется',
        'metric': 'junction_pause_duration (exposure conflict only)',
        'threshold': 3.0,                # Минимум 3 тика пауза
        'test': 'assert_ge'
    },
    '10.5_one_shot_persistence': {
        'description': 'Один high-amplitude aversive event вызывает устойчивый сдвиг path choice',
        'metric': 'P(path_change) post one-shot',
        'threshold': 0.6,                # 60% агентов меняют путь
        'test': 'assert_ge'
    },
    '10.6_no_argmax_regression': {
        'description': 'Gate routing остается каскадным, не scoring-based',
        'metric': 'gate_trigger field contains threshold name',
        'threshold': 1.0,                # 100% триалов имеют winning_constraint
        'test': 'assert_ge'
    },
    '10.7_stage2_compatibility': {
        'description': 'При нулевой экспозиции Stage 2-compatible logic не ломается',
        'metric': 'Stage 2 metrics deviation',
        'threshold': 0.05,               # Максимум 5% отклонение
        'test': 'assert_le'
    }
}


# =============================================================================
# COMBINED CONFIG (для удобного импорта)
# =============================================================================

CONFIG_3_1A: Dict[str, Dict[str, Any]] = {
    'env': ENV_CONFIG,
    'agent': AGENT_CONFIG,
    'logging': LOGGING_CONFIG,
    'ablation': ABLATION_CONFIG,
    'vte': VTE_CONFIG,
    'stats': STATS_CONFIG,
    'acceptance': ACCEPTANCE_CRITERIA
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_ablation_config(ablation_name: str) -> Dict[str, Any]:
    """
    Возвращает конфигурацию для конкретной абляции.
    
    Args:
        ablation_name: Название абляции ('full', 'novg', 'novp', 'nox', 'one_shot_off')
    
    Returns:
        Конфигурация абляции
    """
    if ablation_name not in ABLATION_CONFIG:
        raise ValueError(f"Unknown ablation: {ablation_name}. Available: {list(ABLATION_CONFIG.keys())}")
    
    return ABLATION_CONFIG[ablation_name]


def get_vte_proxy_names() -> List[str]:
    """
    Возвращает список VTE proxy метрик.
    
    Returns:
        List названий метрик
    """
    return VTE_CONFIG['primary_metrics']


def get_step_log_fields() -> List[str]:
    """
    Возвращает список полей для step log.
    
    Returns:
        List названий полей
    """
    return LOGGING_CONFIG['step_log_fields']


def get_trial_log_fields() -> List[str]:
    """
    Возвращает список полей для trial summary log.
    
    Returns:
        List названий полей
    """
    return LOGGING_CONFIG['trial_log_fields']


# =============================================================================
# TESTS
# =============================================================================

def test_config_structure():
    """
    Test: Config structure validation.
    """
    # Проверка что все ключи есть
    assert 'env' in CONFIG_3_1A
    assert 'agent' in CONFIG_3_1A
    assert 'logging' in CONFIG_3_1A
    assert 'ablation' in CONFIG_3_1A
    assert 'vte' in CONFIG_3_1A
    assert 'stats' in CONFIG_3_1A
    assert 'acceptance' in CONFIG_3_1A
    
    # Проверка что абляции определены
    assert 'full' in ABLATION_CONFIG
    assert 'novg' in ABLATION_CONFIG
    assert 'novp' in ABLATION_CONFIG
    assert 'nox' in ABLATION_CONFIG
    assert 'one_shot_off' in ABLATION_CONFIG
    
    # Проверка что VTE метрики определены
    assert len(VTE_CONFIG['primary_metrics']) == 4
    
    # Проверка что acceptance criteria определены
    assert len(ACCEPTANCE_CRITERIA) == 7
    
    print("✓ PASS: Config structure validation")
    return True


def test_helper_functions():
    """
    Test: Helper functions.
    """
    # Test get_ablation_config
    novg_config = get_ablation_config('novg')
    assert novg_config['name'] == 'NoVG'
    
    # Test get_vte_proxy_names
    vte_metrics = get_vte_proxy_names()
    assert len(vte_metrics) == 4
    assert 'junction_pause_duration' in vte_metrics
    
    # Test get_step_log_fields
    step_fields = get_step_log_fields()
    assert len(step_fields) > 20  # Минимум 20 полей
    
    # Test get_trial_log_fields
    trial_fields = get_trial_log_fields()
    assert 'path_choice' in trial_fields
    assert 'junction_pause_duration' in trial_fields
    
    print("✓ PASS: Helper functions")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("Stage 3.1A: Config Validation")
    print("=" * 70)
    
    test_config_structure()
    test_helper_functions()
    
    print("=" * 70)
    print("All config tests passed!")
    print("=" * 70)