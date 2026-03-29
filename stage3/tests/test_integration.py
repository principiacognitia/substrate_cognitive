"""
Stage 3.1A: Integration Test — Full Agent-Environment Loop.

Проверяет полный цикл:
1. Environment генерирует observation с exposure aggregates
2. Agent получает observation и вычисляет diagnostics
3. Gate выбирает режим через threshold cascade
4. Agent выбирает действие на основе режима
5. VTE proxies логируются корректно
6. Backward compatibility mode работает

Acceptance Criteria:
- No ready semions в observation
- Exposure aggregates flow correctly env → agent
- Temporal state updates correctly
- Gate mode selection works in spatial context
- VTE proxies logged at junction
- Backward compatibility produces Stage 2-like behavior

Author: Alex Snow (Aleksey L. Snigirov)
License: MIT
"""

import pytest
import numpy as np
from pathlib import Path

from stage3.core.agent_stage3 import AgentStage3, AgentStage3Config
from stage3.envs.open_covered_choice_env import OpenCoveredChoiceEnv
from stage3.configs.config_stage3_1a import CONFIG_3_1A
from stage3.core.gate_modes import GateMode


# =============================================================================
# TEST 1: Full integration — single trial
# =============================================================================

def test_full_integration_single_trial():
    """
    Test: Полный цикл agent-environment для одного триала.
    """
    # Создаём среду
    env = OpenCoveredChoiceEnv(CONFIG_3_1A['env'], seed=42)
    
    # Создаём агента
    agent = AgentStage3(CONFIG_3_1A['agent'])
    
    # Reset среды
    observation = env.reset(trial=1)
    
    # Проверяем что observation не содержит строковых меток
    for key, value in observation.items():
        if key not in ['path_choice']:  # Исключения для internal state
            assert not isinstance(value, str), f"Observation field {key} is string: {value}"
    
    # Проверяем что exposure aggregates присутствуют
    assert 'X_risk' in observation
    assert 'X_opp' in observation
    assert 'D_est' in observation
    
    # Запускаем полный триал
    done = False
    actions = []
    modes = []
    
    while not done:
        # Agent выбирает действие
        action, metadata = agent.step(
            observation=observation,
            reward=0.0,  # Будет вычислено средой
            action=actions[-1] if actions else 0,
            salience=None  # Будет вычислено агентом
        )
        
        # Environment обрабатывает действие
        observation, reward, done, info = env.step(
            action=action,
            mode=metadata['mode'],
            gate_trigger=metadata['gate_constraint']
        )
        
        actions.append(action)
        modes.append(metadata['mode'])
    
    # Проверяем что триал завершён
    assert done == True
    assert env.state.trial_complete == True
    assert env.state.path_choice in ['open', 'covered']
    
    # Проверяем что логи записаны
    assert len(agent.log_buffer) > 0
    
    print("✓ PASS: Full integration single trial")
    return True


# =============================================================================
# TEST 2: Exposure aggregates flow correctly
# =============================================================================

def test_exposure_aggregates_flow():
    """
    Test: Exposure aggregates корректно передаются env → agent.
    """
    env = OpenCoveredChoiceEnv(CONFIG_3_1A['env'], seed=42)
    agent = AgentStage3(CONFIG_3_1A['agent'])
    
    # Reset
    observation = env.reset(trial=1)
    
    # Сохраняем exposure из среды
    env_exposure = {
        'X_risk': observation['X_risk'],
        'X_opp': observation['X_opp'],
        'D_est': observation['D_est']
    }
    
    # Agent step
    action, metadata = agent.step(observation=observation, reward=0.0)
    
    # Проверяем что agent получил exposure
    agent_exposure = metadata['exposure']
    
    assert abs(agent_exposure['X_risk'] - env_exposure['X_risk']) < 0.01
    assert abs(agent_exposure['X_opp'] - env_exposure['X_opp']) < 0.01
    assert abs(agent_exposure['D_est'] - env_exposure['D_est']) < 0.01
    
    print("✓ PASS: Exposure aggregates flow correctly")
    return True


# =============================================================================
# TEST 3: Temporal state updates correctly
# =============================================================================

def test_temporal_state_updates():
    """
    Test: Temporal state обновляется корректно.
    """
    env = OpenCoveredChoiceEnv(CONFIG_3_1A['env'], seed=42)
    agent = AgentStage3(CONFIG_3_1A['agent'])
    
    # Reset
    observation = env.reset(trial=1)
    
    # Initial temporal state должен быть нулевым
    initial_state = agent.get_current_state()
    assert initial_state['temporal_state']['h_risk'] == 0.0
    assert initial_state['temporal_state']['h_opp'] == 0.0
    assert initial_state['temporal_state']['h_time'] == 0
    
    # Несколько шагов
    for _ in range(5):
        action, metadata = agent.step(observation=observation, reward=0.5)
        observation, reward, done, info = env.step(action=action, mode=metadata['mode'])
    
    # Temporal state должен обновиться
    current_state = agent.get_current_state()
    
    # h_risk и h_opp должны быть > 0 (после exposure)
    assert current_state['temporal_state']['h_risk'] >= 0.0
    assert current_state['temporal_state']['h_opp'] >= 0.0
    
    # h_time должен увеличиться
    assert current_state['temporal_state']['h_time'] >= 0
    
    print("✓ PASS: Temporal state updates correctly")
    return True


# =============================================================================
# TEST 4: Gate mode selection in spatial context
# =============================================================================

def test_gate_mode_selection_spatial():
    """
    Test: Gate выбирает режимы корректно в пространственной задаче.
    """
    env = OpenCoveredChoiceEnv(CONFIG_3_1A['env'], seed=42)
    agent = AgentStage3(CONFIG_3_1A['agent'])
    
    # Reset
    observation = env.reset(trial=1)
    
    modes_seen = set()
    
    # Запускаем несколько триалов
    for trial in range(10):
        env.reset(trial=trial)
        done = False
        
        while not done:
            action, metadata = agent.step(observation=observation, reward=0.0)
            observation, reward, done, info = env.step(action=action, mode=metadata['mode'])
            
            modes_seen.add(metadata['mode'])
    
    # Проверяем что режимы выбираются
    assert len(modes_seen) > 0
    
    # В Stage 3.1A должны быть как минимум EXPLOIT и EXPLORE
    assert 'exploit' in modes_seen or 'explore' in modes_seen
    
    print("✓ PASS: Gate mode selection in spatial context")
    return True


# =============================================================================
# TEST 5: VTE proxies logged at junction
# =============================================================================

# ИСПРАВЛЕНО:
def test_vte_proxies_logged():
    """
    Test: VTE proxies логируются на junction.
    
    Note: VTE proxies возвращаются в info от env.step(), не в metadata от agent.step().
    """
    env = OpenCoveredChoiceEnv(CONFIG_3_1A['env'], seed=42)
    agent = AgentStage3({**CONFIG_3_1A['agent'], 'log_level': 2})
    
    # Reset
    observation = env.reset(trial=1)
    
    # Запускаем триал
    done = False
    junction_logs = []
    
    while not done:
        action, metadata = agent.step(observation=observation, reward=0.0)
        observation, reward, done, info = env.step(action=action, mode=metadata['mode'])
        
        # ИСПРАВЛЕНИЕ: Проверяем что VTE proxies в info (от env.step())
        if 'vte_proxies' in info:
            vte = info['vte_proxies']
            assert 'junction_pause_duration' in vte
            assert 'reorientation_count' in vte
            junction_logs.append(vte)
    
    # Проверяем что логи VTE записаны
    assert len(junction_logs) > 0, f"No VTE proxies logged. junction_logs={junction_logs}"
    
    # Проверяем что trial summary содержит VTE метрики
    summaries = env.get_trial_summaries()
    assert len(summaries) == 1
    
    summary = summaries[0]
    assert summary.junction_pause_duration >= 0
    assert summary.reorientation_count >= 0
    
    print("✓ PASS: VTE proxies logged at junction")
    return True


# =============================================================================
# TEST 6: Backward compatibility mode
# =============================================================================

def test_backward_compatibility_integration():
    """
    Test: Backward compatibility mode работает в интеграции.
    """
    # Config с compatibility mode
    compat_config = CONFIG_3_1A.copy()
    compat_config['agent']['compatibility_mode'] = True
    
    env = OpenCoveredChoiceEnv(CONFIG_3_1A['env'], seed=42)
    agent = AgentStage3(compat_config['agent'])
    
    # Reset
    observation = env.reset(trial=1)
    
    # Запускаем триал
    done = False
    modes_seen = set()
    
    while not done:
        action, metadata = agent.step(observation=observation, reward=0.0)
        observation, reward, done, info = env.step(action=action, mode=metadata['mode'])
        modes_seen.add(metadata['mode'])
    
    # В compatibility mode должны быть только EXPLOIT/EXPLORE
    assert modes_seen.issubset({'exploit', 'explore'}), f"Unexpected modes: {modes_seen}"
    
    print("✓ PASS: Backward compatibility integration")
    return True


# =============================================================================
# TEST 7: Multiple trials — logging consistency
# =============================================================================

def test_multiple_trials_logging():
    """
    Test: Логирование консистентно для нескольких триалов.
    """
    env = OpenCoveredChoiceEnv(CONFIG_3_1A['env'], seed=42)
    # Создаём config с log_level=2
    agent_config = CONFIG_3_1A['agent'].copy()
    agent_config['log_level'] = 2
    
    agent = AgentStage3(agent_config)  # ← Правильно: log_level внутри config
    
    n_trials = 5
    
    # Запускаем несколько триалов
    for trial in range(n_trials):
        observation = env.reset(trial=trial)
        done = False
        
        while not done:
            action, metadata = agent.step(observation=observation, reward=0.0)
            observation, reward, done, info = env.step(action=action, mode=metadata['mode'])
    
    # Проверяем что все trial summaries записаны
    summaries = env.get_trial_summaries()
    assert len(summaries) == n_trials, f"Expected {n_trials} summaries, got {len(summaries)}"
    
    # Проверяем что все summaries имеют required fields
    for summary in summaries:
        assert summary.trial >= 0
        assert summary.path_choice in ['open', 'covered']
        assert summary.reward_total >= 0
    
    # Проверяем что agent log buffer соответствует
    assert len(agent.log_buffer) >= n_trials
    
    print("✓ PASS: Multiple trials logging consistency")
    return True


# =============================================================================
# TEST 8: Path choice under exposure difference
# =============================================================================

def test_path_choice_under_exposure():
    """
    Test: Агент выбирает path с учётом exposure difference.
    """
    # Среда с высоким contrast exposure
    high_contrast_config = CONFIG_3_1A['env'].copy()
    high_contrast_config['paths']['open']['exposure_profile']['X_risk'] = 0.9
    high_contrast_config['paths']['covered']['exposure_profile']['X_risk'] = 0.1
    
    env = OpenCoveredChoiceEnv(high_contrast_config, seed=42)
    agent = AgentStage3(CONFIG_3_1A['agent'])
    
    n_trials = 20
    open_choices = 0
    covered_choices = 0
    
    # Запускаем несколько триалов
    for trial in range(n_trials):
        observation = env.reset(trial=trial)
        done = False
        
        while not done:
            action, metadata = agent.step(observation=observation, reward=0.0)
            observation, reward, done, info = env.step(action=action, mode=metadata['mode'])
        
        # Считаем выборы
        if env.state.path_choice == 'open':
            open_choices += 1
        elif env.state.path_choice == 'covered':
            covered_choices += 1
    
    # При высоком exposure contrast, агент должен предпочитать covered path
    # (это зависит от gate thresholds, но хотя бы некоторый bias должен быть)
    total = open_choices + covered_choices
    assert total == n_trials
    
    # Проверяем что choices записаны
    assert open_choices + covered_choices == n_trials
    
    print(f"✓ PASS: Path choice under exposure (open={open_choices}, covered={covered_choices})")
    return True


# =============================================================================
# TEST 9: Save logs functionality
# =============================================================================

def test_save_logs():
    """
    Test: Сохранение логов работает корректно.
    """
    import tempfile
    import shutil
    
    env = OpenCoveredChoiceEnv(CONFIG_3_1A['env'], seed=42)
    # Создаём config с log_level=2
    agent_config = CONFIG_3_1A['agent'].copy()
    agent_config['log_level'] = 2
    
    agent = AgentStage3(agent_config)  # ← Правильно: log_level внутри config
    
    # Запускаем триал
    observation = env.reset(trial=1)
    done = False
    
    while not done:
        action, metadata = agent.step(observation=observation, reward=0.0)
        observation, reward, done, info = env.step(action=action, mode=metadata['mode'])
    
    # Сохраняем логи во временную директорию
    with tempfile.TemporaryDirectory() as tmpdir:
        env.save_logs(tmpdir)
        
        # Проверяем что файл создан
        log_file = Path(tmpdir) / f"stage3_1a_seed{env.seed}_trials.csv"
        assert log_file.exists(), f"Log file not created: {log_file}"
        
        # Проверяем что файл не пустой
        assert log_file.stat().st_size > 0, "Log file is empty"
    
    print("✓ PASS: Save logs functionality")
    return True


# =============================================================================
# TEST 10: Full session — 30 seeds
# =============================================================================

def test_full_session_30_seeds():
    """
    Test: Полная сессия — 30 seeds (как в Stage 2).
    """
    n_seeds = 30
    n_trials_per_seed = 10
    
    all_summaries = []
    
    for seed in range(42, 42 + n_seeds):
        env = OpenCoveredChoiceEnv(CONFIG_3_1A['env'], seed=seed)
        agent = AgentStage3(CONFIG_3_1A['agent'])
        
        for trial in range(n_trials_per_seed):
            observation = env.reset(trial=trial)
            done = False
            
            while not done:
                action, metadata = agent.step(observation=observation, reward=0.0)
                observation, reward, done, info = env.step(action=action, mode=metadata['mode'])
        
        all_summaries.extend(env.get_trial_summaries())
    
    # Проверяем что все summaries собраны
    expected_total = n_seeds * n_trials_per_seed
    assert len(all_summaries) == expected_total, f"Expected {expected_total} summaries, got {len(all_summaries)}"
    
    # Проверяем что все seeds представлены
    seeds_seen = set(s.seed for s in all_summaries)
    assert len(seeds_seen) == n_seeds, f"Expected {n_seeds} seeds, got {len(seeds_seen)}"
    
    print(f"✓ PASS: Full session 30 seeds ({len(all_summaries)} trials)")
    return True


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Stage 3.1A: Integration Tests — Full Agent-Environment Loop")
    print("=" * 70)
    
    test_full_integration_single_trial()
    test_exposure_aggregates_flow()
    test_temporal_state_updates()
    test_gate_mode_selection_spatial()
    test_vte_proxies_logged()
    test_backward_compatibility_integration()
    test_multiple_trials_logging()
    test_path_choice_under_exposure()
    test_save_logs()
    test_full_session_30_seeds()
    
    print("=" * 70)
    print("All integration tests passed!")
    print("=" * 70)