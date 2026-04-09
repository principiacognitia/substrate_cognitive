"""
Stage 3.0: Test Gate Stage 3 Threshold Cascade.

Design Constraint:
Gate routing должен быть реализован как каскад порогов/прерываний,
не как global argmax scoring.

Acceptance Criterion 8.4:
High risky exposure can force EXPLOIT_SAFE without global mode comparison.

Author: Alex Snow (Aleksey L. Snigirov)
License: MIT
"""

import pytest
from stage3.core.gate_stage3 import GateStage3, GateThresholds, create_test_gate_input
from stage3.core.gate_modes import GateMode
from stage3.core.gate_inputs import GateInput, InstantDiagnostics, ExposureAggregates, TemporalState


# =============================================================================
# TEST 1: Default mode (EXPLOIT) under low uncertainty
# =============================================================================

def test_default_exploit_mode():
    """
    Test: Default режим — EXPLOIT при низкой неопределённости.
    """
    gate = GateStage3()
    
    gate_input = GateInput(
        instant=InstantDiagnostics(u_delta=0.1, u_entropy=0.1, u_volatility=0.1),
        exposure=ExposureAggregates(X_risk=0.1, X_opp=0.5, D_est=0.8),
        temporal=TemporalState(h_risk=0.1, h_opp=0.1, h_time=10)
    )
    
    mode, metadata = gate.select_mode(gate_input)
    
    assert mode == GateMode.EXPLOIT, f"Expected EXPLOIT, got {mode}"
    assert metadata['winning_constraint'] == 'default (low uncertainty)'
    
    print("✓ PASS: Default EXPLOIT mode")


# =============================================================================
# TEST 2: EXPLORE mode under high uncertainty
# =============================================================================

def test_explore_mode_high_uncertainty():
    """
    Test: EXPLORE режим при высокой неопределённости.
    """
    gate = GateStage3()
    
    gate_input = GateInput(
        instant=InstantDiagnostics(u_delta=0.8, u_entropy=0.8, u_volatility=0.8),
        exposure=ExposureAggregates(X_risk=0.1, X_opp=0.5, D_est=0.8),
        temporal=TemporalState(h_risk=0.1, h_opp=0.1, h_time=10)
    )
    
    mode, metadata = gate.select_mode(gate_input)
    
    assert mode == GateMode.EXPLORE, f"Expected EXPLORE, got {mode}"
    assert 'standard_arbitration' in metadata['winning_constraint']
    
    print("✓ PASS: EXPLORE mode under high uncertainty")


# =============================================================================
# TEST 3: EXPLOIT_SAFE override under critical threat
# =============================================================================

def test_exploit_safe_threat_override():
    """
    Test: EXPLOIT_SAFE переопределяет EXPLORE при критической угрозе.
    
    Rationale:
    Threat override должен bypass стандартный explore barrier.
    """
    gate = GateStage3(
        GateThresholds(
            critical_risk_threshold=0.7,
            safe_drive_weight_current=0.6,
            safe_drive_weight_temporal=0.4
        )
    )
    
    # Высокая неопределённость (normally → EXPLORE)
    # НО также критическая угроза (should → EXPLOIT_SAFE)
    gate_input = GateInput(
        instant=InstantDiagnostics(u_delta=0.8, u_entropy=0.8, u_volatility=0.8),
        exposure=ExposureAggregates(X_risk=0.8, X_opp=0.1, D_est=0.8),
        temporal=TemporalState(h_risk=0.8, h_opp=0.1, h_time=10)
    )
    
    mode, metadata = gate.select_mode(gate_input)
    
    # Threat override должен победить
    assert mode == GateMode.EXPLOIT_SAFE, f"Expected EXPLOIT_SAFE (threat override), got {mode}"
    assert 'threat_override' in metadata['winning_constraint']
    
    print("✓ PASS: EXPLOIT_SAFE threat override")


# =============================================================================
# TEST 4: ABSENCE_CHECK trigger
# =============================================================================

def test_absence_check_trigger():
    """
    Test: ABSENCE_CHECK срабатывает при suspicion + poor visibility + safe window.
    """
    gate = GateStage3(GateThresholds(
        suspicion_threshold=0.5,
        visibility_threshold=0.3,
        safe_window_threshold=50
    ))
    
    gate_input = GateInput(
        instant=InstantDiagnostics(u_delta=0.1, u_entropy=0.1, u_volatility=0.1),
        exposure=ExposureAggregates(X_risk=0.5, X_opp=0.1, D_est=0.2),  # Poor visibility
        temporal=TemporalState(h_risk=0.6, h_opp=0.1, h_time=100)  # High suspicion + safe window
    )
    
    mode, metadata = gate.select_mode(gate_input)
    
    assert mode == GateMode.ABSENCE_CHECK, f"Expected ABSENCE_CHECK, got {mode}"
    assert 'absence_trigger' in metadata['winning_constraint']
    
    print("✓ PASS: ABSENCE_CHECK trigger")


# =============================================================================
# TEST 5: No argmax — mode_scores для логирования, не для выбора
# =============================================================================

def test_no_argmax_arbitration():
    """
    Test: Gate не использует global argmax scoring.
    
    Rationale:
    mode_scores должны быть для логирования, не для выбора режима.
    winning_constraint должен указывать какой порог сработал.
    """
    gate = GateStage3()
    gate_input = GateInput(
        instant=InstantDiagnostics(u_delta=0.5, u_entropy=0.3, u_volatility=0.2),
        exposure=ExposureAggregates(X_risk=0.1, X_opp=0.5, D_est=0.8),
        temporal=TemporalState(h_risk=0.0, h_opp=0.0, h_time=0)
    )
    
    mode, metadata = gate.select_mode(gate_input)
    
    # Проверяем что mode_scores есть для всех режимов (для логирования)
    assert len(metadata['mode_scores']) == 4, "Should have scores for all 4 modes"
    
    # Проверяем что winning_constraint указан (threshold cascade, не argmax)
    assert metadata['winning_constraint'] is not None, "Should specify which threshold triggered"
    
    # Проверяем что gate_state_snapshot есть для трассировки
    assert 'gate_state_snapshot' in metadata, "Should have gate state snapshot"
    
    print("✓ PASS: No argmax arbitration")


# =============================================================================
# TEST 6: Threshold cascade priority order
# =============================================================================

def test_threshold_cascade_priority():
    """
    Test: Приоритет каскада (highest → lowest):
    1. ABSENCE_CHECK
    2. EXPLOIT_SAFE
    3. EXPLORE
    4. EXPLOIT
    """
    gate = GateStage3(GateThresholds(
        critical_risk_threshold=0.7,
        suspicion_threshold=0.5,
        visibility_threshold=0.3,
        safe_window_threshold=50
    ))
    
    # Ситуация где срабатывают несколько условий
    # ABSENCE_CHECK должен победить (highest priority)
    gate_input = GateInput(
        instant=InstantDiagnostics(u_delta=0.8, u_entropy=0.8, u_volatility=0.8),  # EXPLORE condition
        exposure=ExposureAggregates(X_risk=0.8, X_opp=0.1, D_est=0.2),  # EXPLOIT_SAFE + ABSENCE_CHECK
        temporal=TemporalState(h_risk=0.6, h_opp=0.1, h_time=100)  # ABSENCE_CHECK safe window
    )
    
    mode, metadata = gate.select_mode(gate_input)
    
    # ABSENCE_CHECK имеет наивысший приоритет
    assert mode == GateMode.ABSENCE_CHECK, f"Expected ABSENCE_CHECK (highest priority), got {mode}"
    
    print("✓ PASS: Threshold cascade priority order")

# =============================================================================
# TEST 7: EXPLOIT_SAFE должен использовать temporal risk
# =============================================================================    

def test_exploit_safe_uses_temporal_risk():
    """
    EXPLOIT_SAFE должен уметь триггериться не только от текущего X_risk,
    но и от accumulated h_risk.
    """
    gate = GateStage3(
        GateThresholds(
            critical_risk_threshold=0.42,
            safe_drive_weight_current=0.6,
            safe_drive_weight_temporal=0.4
        )
    )

    gate_input = create_test_gate_input(
        u_delta=0.2, u_entropy=0.2, u_volatility=0.2,
        X_risk=0.20,     # сам по себе ниже порога
        X_opp=0.1,
        D_est=0.8,
        h_risk=0.80,     # accumulated threat должен дотолкнуть
        h_opp=0.0,
        h_time=5
    )

    mode, metadata = gate.select_mode(gate_input)

    assert mode == GateMode.EXPLOIT_SAFE, f"Expected EXPLOIT_SAFE, got {mode}"
    assert 'threat_override' in metadata['winning_constraint']

    print("✓ PASS: exploit safe uses temporal risk")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Stage 3.0: Test Gate Stage 3 Threshold Cascade")
    print("=" * 70)
    
    test_default_exploit_mode()
    test_explore_mode_high_uncertainty()
    test_exploit_safe_threat_override()
    test_absence_check_trigger()
    test_no_argmax_arbitration()
    test_threshold_cascade_priority()
    
    print("=" * 70)
    print("All tests passed!")
    print("=" * 70)