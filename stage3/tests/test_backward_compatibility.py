"""
Stage 3.0: Test Backward Compatibility with Stage 2.

Design Constraint:
При нулевых exposure/temporal trace, Stage 3.0 должен деградировать
в Stage 2 behavior с теми же конфигами.

Acceptance Criterion 8.1:
With zero exposure and zero temporal traces, Stage 3.0 reproduces Stage 2 behavior.

Author: Alex Snow (Aleksey L. Snigirov)
License: MIT
"""

import pytest
from stage3.core.agent_stage3 import AgentStage3, AgentStage3Config
from stage3.core.compatibility import Stage2CompatShim, Stage2CompatConfig, verify_backward_compatibility
from stage3.core.gate_modes import GateMode
from stage3.core.gate_inputs import GateInput, InstantDiagnostics, ExposureAggregates, TemporalState


# =============================================================================
# TEST 1: Zero exposure input creates Stage 2-compatible GateInput
# =============================================================================

def test_zero_exposure_input():
    """
    Test: Compatibility shim создаёт нулевые exposure/temporal.
    """
    shim = Stage2CompatShim(Stage2CompatConfig(enabled=True))
    
    instant = InstantDiagnostics(u_delta=0.5, u_entropy=0.3, u_volatility=0.2)
    gate_input = shim.create_compat_input(instant)
    
    # Проверяем что exposure нулевой
    assert gate_input.exposure.X_risk == 0.0
    assert gate_input.exposure.X_opp == 0.0
    assert gate_input.exposure.D_est == 0.0
    
    # Проверяем что temporal нулевой
    assert gate_input.temporal.h_risk == 0.0
    assert gate_input.temporal.h_opp == 0.0
    assert gate_input.temporal.h_time == 0
    
    print("✓ PASS: Zero exposure input")


# =============================================================================
# TEST 2: Compatibility mode restricts modes to {EXPLOIT, EXPLORE}
# =============================================================================

def test_compat_mode_restricts_modes():
    """
    Test: В compatibility mode доступны только EXPLOIT/EXPLORE.
    """
    shim_enabled = Stage2CompatShim(Stage2CompatConfig(enabled=True))
    shim_disabled = Stage2CompatShim(Stage2CompatConfig(enabled=False))
    
    # В compat mode — только 2 режима
    allowed_modes_enabled = shim_enabled.get_allowed_modes()
    assert len(allowed_modes_enabled) == 2
    assert GateMode.EXPLOIT in allowed_modes_enabled
    assert GateMode.EXPLORE in allowed_modes_enabled
    assert GateMode.EXPLOIT_SAFE not in allowed_modes_enabled
    assert GateMode.ABSENCE_CHECK not in allowed_modes_enabled
    
    # В normal mode — все 4 режима
    allowed_modes_disabled = shim_disabled.get_allowed_modes()
    assert len(allowed_modes_disabled) == 4
    
    print("✓ PASS: Compatibility mode restricts modes")


# =============================================================================
# TEST 3: Agent in compatibility mode produces Stage 2-like behavior
# =============================================================================

def test_agent_compat_mode_behavior():
    """
    Test: Агент в compatibility mode ведёт себя как Stage 2.
    """
    agent = AgentStage3(AgentStage3Config(compatibility_mode=True, log_level=0))
    
    observation = {
        'prediction_error': 0.5,
        'policy_entropy': 0.3,
        'q_values': [0.6, 0.4],
        'expected_reward': 0.5
    }
    
    # Несколько шагов
    modes_seen = set()
    for _ in range(10):
        action, metadata = agent.step(observation, reward=0.8)
        modes_seen.add(metadata['mode'])
    
    # В compat mode должны быть только EXPLOIT/EXPLORE
    assert modes_seen.issubset({'exploit', 'explore'}), f"Unexpected modes: {modes_seen}"
    
    print("✓ PASS: Agent compatibility mode behavior")


# =============================================================================
# TEST 4: Backward compatibility verification function
# =============================================================================

def test_backward_compatibility_verification():
    """
    Test: verify_backward_compatibility работает корректно.
    """
    # Passing case (within 5% tolerance)
    stage3_results = {
        'mb_interaction_coef': 0.312,
        'switching_latency_median': 35.0,
        'perseveration_median': 8.0
    }
    stage2_results = {
        'mb_interaction_coef': 0.310,
        'switching_latency_median': 35.5,
        'perseveration_median': 8.2
    }
    
    passed, message = verify_backward_compatibility(
        stage3_results,
        stage2_results,
        tolerance=0.05
    )
    
    assert passed, f"Should pass: {message}"
    
    # Failing case (> 5% deviation)
    stage3_results_bad = {
        'mb_interaction_coef': 0.500,  # Big deviation
        'switching_latency_median': 35.0,
        'perseveration_median': 8.0
    }
    
    passed, message = verify_backward_compatibility(
        stage3_results_bad,
        stage2_results,
        tolerance=0.05
    )
    
    assert not passed, f"Should fail: {message}"
    
    print("✓ PASS: Backward compatibility verification")


# =============================================================================
# TEST 5: Temporal state remains zero with zero exposure
# =============================================================================

def test_temporal_state_zero_with_zero_exposure():
    """
    Test: При нулевых exposure, temporal trace остаются ~0.
    """
    from stage3.core.temporal_state import TemporalStateUpdater, TemporalStateConfig
    
    config = TemporalStateConfig()
    updater = TemporalStateUpdater(config)
    
    state = TemporalState(h_risk=0.0, h_opp=0.0, h_time=0)
    
    # 10 обновлений с нулевыми exposure
    for _ in range(10):
        state = updater.update(
            state=state,
            X_risk=0.0,
            X_opp=0.0,
            salience=0.0,
            stakes=0.0
        )
    
    # Trace должны оставаться близки к нулю
    assert state.h_risk < 0.01, f"h_risk should be ~0, got {state.h_risk}"
    assert state.h_opp < 0.01, f"h_opp should be ~0, got {state.h_opp}"
    
    print("✓ PASS: Temporal state zero with zero exposure")


# =============================================================================
# TEST 6: Agent reset clears all state
# =============================================================================

def test_agent_reset_clears_state():
    """
    Test: agent.reset() очищает всё состояние.
    """
    agent = AgentStage3(AgentStage3Config(log_level=2))
    
    observation = {
        'prediction_error': 0.5,
        'policy_entropy': 0.3,
        'q_values': [0.6, 0.4],
        'expected_reward': 0.5
    }
    
    # Несколько шагов
    for _ in range(5):
        agent.step(observation, reward=0.8)
    
    # Проверяем что state не нулевой
    state = agent.get_current_state()
    assert state['trial'] == 5, f"Trial should be 5, got {state['trial']}"
    assert len(agent.get_logs()) == 5, "Should have 5 logs"
    
    # Reset
    agent.reset()
    
    # Проверяем что state нулевой
    state = agent.get_current_state()
    assert state['trial'] == 0, f"Trial should be 0 after reset, got {state['trial']}"
    assert len(agent.get_logs()) == 0, "Logs should be cleared after reset"
    
    print("✓ PASS: Agent reset clears state")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Stage 3.0: Test Backward Compatibility with Stage 2")
    print("=" * 70)
    
    test_zero_exposure_input()
    test_compat_mode_restricts_modes()
    test_agent_compat_mode_behavior()
    test_backward_compatibility_verification()
    test_temporal_state_zero_with_zero_exposure()
    test_agent_reset_clears_state()
    
    print("=" * 70)
    print("All tests passed!")
    print("=" * 70)