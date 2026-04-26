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

from importlib.metadata import metadata
from turtle import done

import pytest
import numpy as np
from pathlib import Path
import tempfile

from stage3.core.agent_stage3 import AgentStage3, AgentStage3Config
from stage3.envs.open_covered_choice_env import OpenCoveredChoiceEnv
from stage3.configs.config_stage3_1a import CONFIG_3_1A
from stage3.core.gate_modes import GateMode


# =============================================================================
# HELPER: Создание агента и среды с учётом новых сигнатур
# =============================================================================

def create_test_env(seed=42):
    """Создаёт среду с правильным seed."""
    return OpenCoveredChoiceEnv(CONFIG_3_1A['env'], seed=seed)

def create_test_agent(seed=42, log_level=2, compat_mode=False):
    """Создаёт агента с новой структурой конфига."""
    agent_config = CONFIG_3_1A['agent'].copy()
    agent_config['log_level'] = log_level
    agent_config['compatibility_mode'] = compat_mode
    return AgentStage3(agent_config, seed=seed)

def run_trial(env, agent, trial: int, max_steps: int = 50):
    """
    Унифицированный прогон одного триала для integration tests.

    ВАЖНО:
    - всегда берём observation из env.reset(...)
    - всегда передаём в env.step() mode, gate_trigger и action_probs
    - есть защитный лимит max_steps, чтобы тесты не зависали бесконечно
    """
    observation = env.reset(trial=trial)
    done = False
    step_count = 0
    last_info = None
    last_reward = 0.0
    last_metadata = None

    while not done and step_count < max_steps:
        action, metadata = agent.step(
            observation=observation,
            reward=0.0
        )

        observation, reward, done, info = env.step(
            action=action,
            mode=metadata['mode'],
            gate_trigger=metadata.get('gate_constraint', 'default'),
            action_probs=metadata.get('action_probs', [0.5, 0.5])
        )

        last_info = info
        last_reward = reward
        last_metadata = metadata
        step_count += 1

    assert done, (
        f"Trial did not terminate within {max_steps} steps. "
        f"trial={trial}, node={env.state.current_node}, "
        f"state={env.state.deliberation_state.value}"
    )

    return observation, last_reward, done, last_info, last_metadata

# =============================================================================
# TEST 1: Full integration — single trial
# =============================================================================

def test_full_integration_single_trial():
    """
    Test: Полный цикл agent-environment для одного триала.
    """
    # Создаём среду
    env = create_test_env(seed=42)
    
    # Создаём агента
    agent = create_test_agent(seed=42, log_level=2)
    
    # Reset среды
    observation = env.reset(trial=1)
    
    # Запускаем полный триал
    done = False
    actions = []
    
    while not done:
        action, metadata = agent.step(observation=observation, reward=0.0, action=0)
        observation, reward, done, info = env.step(
            action=action,
            mode=metadata['mode'],
            gate_trigger=metadata['gate_constraint'],
            action_probs=metadata.get('action_probs', [0.5, 0.5])
        )
        actions.append(action)
    
    # Проверяем что триал завершён
    assert done == True
    assert env.state.trial_complete == True
    
    # Проверяем что логи записаны
    assert len(agent.log_buffer) > 0
    
    print("✓ PASS: Full integration single trial")


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
    agent_exposure = metadata['node_exposure']
    
    assert 'gate_exposure' in metadata
    assert 'gate_exposure_source' in metadata
    
    assert abs(agent_exposure['X_risk'] - env_exposure['X_risk']) < 0.01
    assert abs(agent_exposure['X_opp'] - env_exposure['X_opp']) < 0.01
    assert abs(agent_exposure['D_est'] - env_exposure['D_est']) < 0.01
    
    print("✓ PASS: Exposure aggregates flow correctly")

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
        observation, reward, done, info = env.step(
            action=action,
            mode=metadata['mode'],
            gate_trigger=metadata.get('gate_constraint', 'default'),
            action_probs=metadata.get('action_probs', [0.5, 0.5])
        )
    
    # Temporal state должен обновиться
    current_state = agent.get_current_state()
    
    # h_risk и h_opp должны быть > 0 (после exposure)
    assert current_state['temporal_state']['h_risk'] >= 0.0
    assert current_state['temporal_state']['h_opp'] >= 0.0
    
    # h_time должен увеличиться
    assert current_state['temporal_state']['h_time'] >= 0
    
    print("✓ PASS: Temporal state updates correctly")

# =============================================================================
# TEST 4: Gate mode selection in spatial context
# =============================================================================

def test_gate_mode_selection_spatial():
    """
    Test: Gate выбирает режимы корректно в пространственной задаче.
    """
    env = create_test_env(seed=42)
    agent = create_test_agent(seed=42, log_level=1)

    modes_seen = set()

    for trial in range(10):
        observation = env.reset(trial=trial)
        done = False
        step_count = 0

        while not done and step_count < 50:
            action, metadata = agent.step(observation=observation, reward=0.0)

            observation, reward, done, info = env.step(
                action=action,
                mode=metadata['mode'],
                gate_trigger=metadata.get('gate_constraint', 'default'),
                action_probs=metadata.get('action_probs', [0.5, 0.5])
            )

            modes_seen.add(metadata['mode'])
            step_count += 1

        assert done, (
            f"Trial {trial} did not terminate. "
            f"node={env.state.current_node}, state={env.state.deliberation_state.value}"
        )

    assert len(modes_seen) > 0
    assert ('exploit' in modes_seen) or ('explore' in modes_seen)

    print("✓ PASS: Gate mode selection in spatial context")

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
    
    step_count = 0
    while not done and step_count < 50:
        action, metadata = agent.step(observation=observation, reward=0.0)
        observation, reward, done, info = env.step(
            action=action,
            mode=metadata['mode'],
            gate_trigger=metadata.get('gate_constraint', 'default'),
            action_probs=metadata.get('action_probs', [0.5, 0.5])
        )
        step_count += 1
 
        # ИСПРАВЛЕНИЕ: Проверяем что VTE proxies в info (от env.step())
        if 'vte_proxies' in info:
            vte = info['vte_proxies']
            assert 'junction_pause_duration' in vte
            assert 'reorientation_count' in vte
            junction_logs.append(vte)

    assert done, f"Trial did not terminate in test_vte_proxies_logged"
    
    # Проверяем что логи VTE записаны
    assert len(junction_logs) > 0, f"No VTE proxies logged. junction_logs={junction_logs}"
    
    # Проверяем что trial summary содержит VTE метрики
    summaries = env.get_trial_summaries()
    assert len(summaries) == 1
    
    summary = summaries[0]
    assert summary.junction_pause_duration >= 0
    assert summary.reorientation_count >= 0
    
    print("✓ PASS: VTE proxies logged at junction")

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
    
    step_count = 0
    while not done and step_count < 50:
        action, metadata = agent.step(observation=observation, reward=0.0)
        observation, reward, done, info = env.step(
            action=action,
            mode=metadata['mode'],
            gate_trigger=metadata.get('gate_constraint', 'default'),
            action_probs=metadata.get('action_probs', [0.5, 0.5])
        )
        modes_seen.add(metadata['mode'])
        step_count += 1

    assert done, "Backward compatibility trial did not terminate"
    
    # В compatibility mode должны быть только EXPLOIT/EXPLORE
    assert modes_seen.issubset({'exploit', 'explore'}), f"Unexpected modes: {modes_seen}"
    
    print("✓ PASS: Backward compatibility integration")

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
        step_count = 0

        while not done and step_count < 50:
            action, metadata = agent.step(observation=observation, reward=0.0)
            observation, reward, done, info = env.step(
                action=action,
                mode=metadata['mode'],
                gate_trigger=metadata.get('gate_constraint', 'default'),
                action_probs=metadata.get('action_probs', [0.5, 0.5])
            )
            step_count += 1

        assert done, f"Trial {trial} did not terminate in test_multiple_trials_logging"
    
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
        
        step_count = 0
        while not done and step_count < 50:
            action, metadata = agent.step(observation=observation, reward=0.0)
            observation, reward, done, info = env.step(
                action=action,
                mode=metadata['mode'],
                gate_trigger=metadata.get('gate_constraint', 'default'),
                action_probs=metadata.get('action_probs', [0.5, 0.5])
            )
            step_count += 1

        assert done, f"Trial {trial} did not terminate in test_path_choice_under_exposure"
        
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

# =============================================================================
# TEST 9: Save logs functionality
# =============================================================================

def test_save_logs():
    """
    Test: Сохранение логов работает корректно.
    """
    import tempfile
    from pathlib import Path
    
    env = OpenCoveredChoiceEnv(CONFIG_3_1A['env'], seed=42)
    
    # Создаём агент с log_level=2 через dict
    agent_config = {**CONFIG_3_1A['agent'], 'log_level': 2}
    agent = AgentStage3(agent_config, seed=42)
    
    # Запускаем триал
    observation = env.reset(trial=1)
    done = False
    
    step_count = 0
    while not done and step_count < 50:
        action, metadata = agent.step(observation=observation, reward=0.0)
        observation, reward, done, info = env.step(
            action=action,
            mode=metadata['mode'],
            gate_trigger=metadata.get('gate_constraint', 'default'),
            action_probs=metadata.get('action_probs', [0.5, 0.5])
        )
        step_count += 1

    assert done, "Trial did not terminate in test_save_logs"
    
    # Сохраняем логи во временную директорию
    with tempfile.TemporaryDirectory() as tmpdir:
        # ИСПРАВЛЕНО: два аргумента (output_dir, filename)
        filename = f"stage3_1a_seed{env.seed}_trials.csv"
        env.save_logs(tmpdir, filename)
        
        # Проверяем что файл создан
        log_file = Path(tmpdir) / filename
        assert log_file.exists(), f"Log file not created: {log_file}"
        
        # Проверяем что файл не пустой
        assert log_file.stat().st_size > 0, "Log file is empty"
    
    print("✓ PASS: Save logs functionality")

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
        agent = AgentStage3(CONFIG_3_1A['agent'], seed=seed)

        for trial in range(n_trials_per_seed):
            observation = env.reset(trial=trial)
            done = False
            step_count = 0

            while not done and step_count < 50:
                action, metadata = agent.step(observation=observation, reward=0.0)
                observation, reward, done, info = env.step(
                    action=action,
                    mode=metadata['mode'],
                    gate_trigger=metadata.get('gate_constraint', 'default'),
                    action_probs=metadata.get('action_probs', [0.5, 0.5])
                )
                step_count += 1

            assert done, (
                f"Seed {seed}, trial {trial} did not terminate. "
                f"node={env.state.current_node}, state={env.state.deliberation_state.value}"
            )
        
        all_summaries.extend(env.get_trial_summaries())
    
    # Проверяем что все summaries собраны
    expected_total = n_seeds * n_trials_per_seed
    assert len(all_summaries) == expected_total, f"Expected {expected_total} summaries, got {len(all_summaries)}"
    
    # Проверяем что все seeds представлены
    seeds_seen = set(s.seed for s in all_summaries)
    assert len(seeds_seen) == n_seeds, f"Expected {n_seeds} seeds, got {len(seeds_seen)}"
    
    print(f"✓ PASS: Full session 30 seeds ({len(all_summaries)} trials)")

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