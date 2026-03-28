"""
Stage 3.0: Test Temporal State Update Logic.

Design Principle:
One-shot learning is an amplitude-dependent update regime of the same
temporal state, not a separate memory module.

Acceptance Criteria:
8.1 Backward compatibility (zero exposure → zero traces)
8.3 One-shot as update regime (high-amplitude event shifts traces)

Author: Alex Snow (Aleksey L. Snigirov)
License: MIT
"""

import pytest
import numpy as np
from stage3.core.temporal_state import TemporalStateUpdater, TemporalStateConfig, TemporalState
from stage3.core.gate_inputs import TemporalState as TemporalStateInput


# =============================================================================
# TEST 1: h_risk update with exponential smoothing
# =============================================================================

def test_h_risk_exponential_smoothing():
    """
    Test: h_risk обновляется с экспоненциальным сглаживанием.
    
    Formula:
    h_risk(t+1) = λ_r * h_risk(t) + (1 - λ_r) * X_risk(t)
    """
    config = TemporalStateConfig(lambda_risk=0.9)
    updater = TemporalStateUpdater(config)
    
    state = TemporalStateInput.zeros()
    
    # Первое обновление с X_risk = 0.5
    state = updater.update(
        state=state,
        X_risk=0.5,
        X_opp=0.0,
        salience=0.1,
        stakes=1.0
    )
    
    # Ожидаемое значение: 0.9 * 0.0 + 0.1 * 0.5 = 0.05
    expected = 0.05
    assert abs(state.h_risk - expected) < 0.01, f"h_risk should be {expected}, got {state.h_risk}"
    
    # Второе обновление с X_risk = 0.5
    state = updater.update(
        state=state,
        X_risk=0.5,
        X_opp=0.0,
        salience=0.1,
        stakes=1.0
    )
    
    # Ожидаемое значение: 0.9 * 0.05 + 0.1 * 0.5 = 0.095
    expected = 0.095
    assert abs(state.h_risk - expected) < 0.01, f"h_risk should be {expected}, got {state.h_risk}"
    
    print("✓ PASS: h_risk exponential smoothing")


# =============================================================================
# TEST 2: h_opp update with exponential smoothing
# =============================================================================

def test_h_opp_exponential_smoothing():
    """
    Test: h_opp обновляется с экспоненциальным сглаживанием.
    
    Formula:
    h_opp(t+1) = λ_o * h_opp(t) + (1 - λ_o) * X_opp(t)
    """
    config = TemporalStateConfig(lambda_opp=0.9)
    updater = TemporalStateUpdater(config)
    
    state = TemporalStateInput.zeros()
    
    # Обновление с X_opp = 0.3
    state = updater.update(
        state=state,
        X_risk=0.0,
        X_opp=0.3,
        salience=0.1,
        stakes=1.0
    )
    
    # Ожидаемое значение: 0.9 * 0.0 + 0.1 * 0.3 = 0.03
    expected = 0.03
    assert abs(state.h_opp - expected) < 0.01, f"h_opp should be {expected}, got {state.h_opp}"
    
    print("✓ PASS: h_opp exponential smoothing")


# =============================================================================
# TEST 3: h_time reset on salient events
# =============================================================================

def test_h_time_reset_on_salient_event():
    """
    Test: h_time сбрасывается при salient event.
    
    Formula:
    h_time(t+1) = 0, если salience > τ_sal
    h_time(t+1) = h_time(t) + 1, иначе
    """
    config = TemporalStateConfig(salience_threshold=0.5)
    updater = TemporalStateUpdater(config)
    
    state = TemporalStateInput(h_risk=0.0, h_opp=0.0, h_time=10)
    
    # Low salience → h_time increment
    state = updater.update(
        state=state,
        X_risk=0.1,
        X_opp=0.1,
        salience=0.3,  # < 0.5
        stakes=1.0
    )
    assert state.h_time == 11, f"h_time should increment to 11, got {state.h_time}"
    
    # High salience → h_time reset
    state = updater.update(
        state=state,
        X_risk=0.1,
        X_opp=0.1,
        salience=0.8,  # > 0.5
        stakes=1.0
    )
    assert state.h_time == 0, f"h_time should reset to 0, got {state.h_time}"
    
    print("✓ PASS: h_time reset on salient event")


# =============================================================================
# TEST 4: One-shot as amplitude-dependent update regime
# =============================================================================

def test_one_shot_amplitude_dependent():
    """
    Test: One-shot — это amplitude-dependent update regime, не отдельный модуль.
    
    Rationale:
    One-shot learning is not a separate subsystem.
    It is an amplitude-dependent update regime of TemporalState.
    """
    config = TemporalStateConfig(
        one_shot_threshold=5.0,
        one_shot_boost=2.0
    )
    updater = TemporalStateUpdater(config)
    
    state = TemporalStateInput.zeros()
    
    # Low amplitude (many-shot)
    state_low = updater.update(
        state=state,
        X_risk=0.3,
        X_opp=0.1,
        salience=0.2,  # Low salience
        stakes=1.0     # Low stakes
        # surprise_amplitude = 0.2 < 5.0 → no one-shot
    )
    
    assert state_low.one_shot_pending == False, "Low amplitude should not trigger one-shot"
    
    # High amplitude (one-shot)
    state_high = updater.update(
        state=state,
        X_risk=0.3,
        X_opp=0.1,
        salience=0.9,  # High salience
        stakes=10.0    # High stakes
        # surprise_amplitude = 9.0 > 5.0 → one-shot!
    )
    
    assert state_high.one_shot_pending == True, "High amplitude should trigger one-shot"
    assert state_high.one_shot_amplitude > 5.0, "Amplitude should be recorded"
    assert state_high.h_risk > state_low.h_risk, "One-shot should boost h_risk"
    assert state_high.h_opp > state_low.h_opp, "One-shot should boost h_opp"
    
    print("✓ PASS: One-shot as amplitude-dependent update regime")


# =============================================================================
# TEST 5: Backward compatibility (zero exposure → zero traces)
# =============================================================================

def test_backward_compatibility_zero_exposure():
    """
    Test: При нулевых exposure trace должны оставаться близки к нулю.
    
    Acceptance Criterion 8.1:
    With zero exposure and zero temporal traces, Stage 3.0 reproduces Stage 2 behavior.
    """
    config = TemporalStateConfig()
    updater = TemporalStateUpdater(config)
    
    state = TemporalStateInput.zeros()
    
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
    
    print("✓ PASS: Backward compatibility (zero exposure)")


# =============================================================================
# TEST 6: Trace dynamics simulation
# =============================================================================

def test_trace_dynamics_simulation():
    """
    Test: Проигрывание последовательности обновлений.
    """
    config = TemporalStateConfig()
    updater = TemporalStateUpdater(config)
    
    initial_state = TemporalStateInput.zeros()
    
    # Последовательность событий
    X_risk_seq = [0.1, 0.3, 0.5, 0.2, 0.1]
    X_opp_seq = [0.1, 0.1, 0.2, 0.1, 0.1]
    salience_seq = [0.2, 0.3, 0.8, 0.2, 0.2]
    
    states = updater.get_trace_dynamics(
        initial_state=initial_state,
        X_risk_sequence=X_risk_seq,
        X_opp_sequence=X_opp_seq,
        salience_sequence=salience_seq
    )
    
    # Должно быть 6 states (initial + 5 updates)
    assert len(states) == 6, f"Should have 6 states, got {len(states)}"
    
    # h_time должен сброситься на шаге 2 (salience=0.8)
    assert states[3].h_time == 0, f"h_time should reset at step 3, got {states[3].h_time}"
    
    print("✓ PASS: Trace dynamics simulation")


# =============================================================================
# TEST 7: Config validation
# =============================================================================

def test_config_validation():
    """
    Test: Валидация конфигурации TemporalStateConfig.
    """
    # Valid config
    config = TemporalStateConfig(
        lambda_risk=0.9,
        lambda_opp=0.9,
        salience_threshold=0.5,
        one_shot_threshold=5.0,
        one_shot_boost=2.0
    )
    assert config.lambda_risk == 0.9
    assert config.lambda_opp == 0.9
    
    # Invalid lambda_risk (> 1.0)
    with pytest.raises(ValueError, match="lambda_risk must be in"):
        TemporalStateConfig(lambda_risk=1.5)
    
    # Invalid lambda_opp (< 0.0)
    with pytest.raises(ValueError, match="lambda_opp must be in"):
        TemporalStateConfig(lambda_opp=-0.1)
    
    # Invalid salience_threshold (< 0)
    with pytest.raises(ValueError, match="salience_threshold must be"):
        TemporalStateConfig(salience_threshold=-0.5)
    
    print("✓ PASS: Config validation")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Stage 3.0: Test Temporal State Update Logic")
    print("=" * 70)
    
    test_h_risk_exponential_smoothing()
    test_h_opp_exponential_smoothing()
    test_h_time_reset_on_salient_event()
    test_one_shot_amplitude_dependent()
    test_backward_compatibility_zero_exposure()
    test_trace_dynamics_simulation()
    test_config_validation()
    
    print("=" * 70)
    print("All tests passed!")
    print("=" * 70)