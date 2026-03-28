"""
Stage 3.0: Test Action-Basin Equivalence.

Design Principle:
Два перцептивно разных стимула с близким exposure profile должны
давать схожий mode bias.

Acceptance Criterion 8.5:
Two perceptually different stimuli that induce similar exposure profiles
and require the same response regime must lead to similar Gate routing.

Author: Alex Snow (Aleksey L. Snigirov)
License: MIT
"""

import pytest
import numpy as np
from stage3.core.exposure_field import ExposureField
from stage3.core.gate_stage3 import GateStage3, GateThresholds
from stage3.core.gate_inputs import GateInput, InstantDiagnostics, ExposureAggregates, TemporalState
from stage3.core.gate_modes import GateMode


# =============================================================================
# TEST 1: Perceptually different stimuli with similar X-profile
# =============================================================================

def test_perceptually_different_similar_exposure():
    """
    Test: Два перцептивно разных стимула с близким X-profile.
    """
    field = ExposureField()
    
    # Стимул 1: numeric features
    obs1 = {
        'prediction_error': 0.8,
        'policy_entropy': 0.2,
        'q_values': [0.8, 0.2]
    }
    
    # Стимул 2: другие numeric features, но схожий exposure profile
    obs2 = {
        'prediction_error': 0.75,
        'policy_entropy': 0.25,
        'q_values': [0.75, 0.25]
    }
    
    # Вычисляем aggregates
    agg1 = field.compute_exposure(obs1, reward=0.8, trial=1)
    agg2 = field.compute_exposure(obs2, reward=0.75, trial=2)
    
    # Проверяем что aggregates близки (action basin equivalence)
    risk_diff = abs(agg1.X_risk - agg2.X_risk)
    opp_diff = abs(agg1.X_opp - agg2.X_opp)
    det_diff = abs(agg1.D_est - agg2.D_est)
    
    # Допустимое отклонение
    tolerance = 0.2
    
    assert risk_diff < tolerance, f"X_risk difference {risk_diff} > {tolerance}"
    assert opp_diff < tolerance, f"X_opp difference {opp_diff} > {tolerance}"
    assert det_diff < tolerance, f"D_est difference {det_diff} > {tolerance}"
    
    print("✓ PASS: Perceptually different stimuli with similar X-profile")


# =============================================================================
# TEST 2: Similar exposure → similar Gate mode selection
# =============================================================================

def test_similar_exposure_similar_mode():
    """
    Test: Схожий exposure → схожий выбор режима Gate.
    """
    gate = GateStage3()
    
    # Input 1: moderate risk
    input1 = GateInput(
        instant=InstantDiagnostics(u_delta=0.5, u_entropy=0.3, u_volatility=0.2),
        exposure=ExposureAggregates(X_risk=0.3, X_opp=0.5, D_est=0.6),
        temporal=TemporalState(h_risk=0.2, h_opp=0.3, h_time=10)
    )
    
    # Input 2: similar risk (slight variation)
    input2 = GateInput(
        instant=InstantDiagnostics(u_delta=0.48, u_entropy=0.32, u_volatility=0.22),
        exposure=ExposureAggregates(X_risk=0.32, X_opp=0.48, D_est=0.58),
        temporal=TemporalState(h_risk=0.22, h_opp=0.28, h_time=12)
    )
    
    mode1, metadata1 = gate.select_mode(input1)
    mode2, metadata2 = gate.select_mode(input2)
    
    # Режимы должны совпадать (или быть очень близкими)
    # Note: В threshold cascade режимы должны совпадать при схожих inputs
    assert mode1 == mode2, f"Modes should match: {mode1} vs {mode2}"
    
    print("✓ PASS: Similar exposure → similar Gate mode")


# =============================================================================
# TEST 3: Different exposure → different Gate mode
# =============================================================================

def test_different_exposure_different_mode():
    """
    Test: Разный exposure → разный выбор режима Gate.
    """
    gate = GateStage3(GateThresholds(critical_risk_threshold=0.7))
    
    # Input 1: low risk → EXPLOIT
    input1 = GateInput(
        instant=InstantDiagnostics(u_delta=0.1, u_entropy=0.1, u_volatility=0.1),
        exposure=ExposureAggregates(X_risk=0.1, X_opp=0.5, D_est=0.8),
        temporal=TemporalState(h_risk=0.1, h_opp=0.1, h_time=10)
    )
    
    # Input 2: critical risk → EXPLOIT_SAFE
    input2 = GateInput(
        instant=InstantDiagnostics(u_delta=0.1, u_entropy=0.1, u_volatility=0.1),
        exposure=ExposureAggregates(X_risk=0.8, X_opp=0.1, D_est=0.8),  # Critical
        temporal=TemporalState(h_risk=0.5, h_opp=0.1, h_time=10)
    )
    
    mode1, _ = gate.select_mode(input1)
    mode2, _ = gate.select_mode(input2)
    
    # Режимы должны различаться
    assert mode1 != mode2, f"Modes should differ: {mode1} vs {mode2}"
    assert mode1 == GateMode.EXPLOIT
    assert mode2 == GateMode.EXPLOIT_SAFE
    
    print("✓ PASS: Different exposure → different Gate mode")


# =============================================================================
# TEST 4: Action basin equivalence through exposure aggregation
# =============================================================================

def test_action_basin_through_aggregation():
    """
    Test: Action basin equivalence через exposure aggregation.
    
    Rationale:
    Seven — не метка объекта, а метка режима реагирования.
    """
    field = ExposureField()
    
    # Стимул 1: "змея" (high threat features)
    obs_snake = {
        'prediction_error': 0.9,
        'policy_entropy': 0.1,
        'q_values': [0.1, 0.9],
        'threat_indicator': 0.9
    }
    
    # Стимул 2: "палка" (low threat features)
    obs_stick = {
        'prediction_error': 0.2,
        'policy_entropy': 0.3,
        'q_values': [0.6, 0.4],
        'threat_indicator': 0.1
    }
    
    # Вычисляем aggregates
    agg_snake = field.compute_exposure(obs_snake, reward=-1.0, trial=1)
    agg_stick = field.compute_exposure(obs_stick, reward=0.5, trial=2)
    
    # Aggregates должны различаться (разные action basins)
    assert agg_snake.X_risk > agg_stick.X_risk, "Snake should have higher risk"
    assert agg_snake.X_opp < agg_stick.X_opp, "Snake should have lower opportunity"
    
    # Gate должен выбрать разные режимы
    gate = GateStage3(GateThresholds(critical_risk_threshold=0.5))
    
    input_snake = GateInput(
        instant=InstantDiagnostics(u_delta=0.9, u_entropy=0.1, u_volatility=0.5),
        exposure=agg_snake,
        temporal=TemporalState(h_risk=0.6, h_opp=0.1, h_time=10)
    )
    
    input_stick = GateInput(
        instant=InstantDiagnostics(u_delta=0.2, u_entropy=0.3, u_volatility=0.2),
        exposure=agg_stick,
        temporal=TemporalState(h_risk=0.1, h_opp=0.5, h_time=10)
    )
    
    mode_snake, _ = gate.select_mode(input_snake)
    mode_stick, _ = gate.select_mode(input_stick)
    
    # Разные stimulus → разные aggregates → разные режимы
    # (это и есть action basin equivalence)
    assert mode_snake != mode_stick, f"Modes should differ: {mode_snake} vs {mode_stick}"
    
    print("✓ PASS: Action basin through exposure aggregation")


# =============================================================================
# TEST 5: No semantic labels in action basin equivalence
# =============================================================================

def test_no_semantic_labels_in_basin():
    """
    Test: Action basin equivalence не использует семантические метки.
    
    Rationale:
    Design Constraint: No Ready Semions at Port.
    """
    field = ExposureField()
    
    # Observation без строковых меток (только numeric)
    obs_numeric = {
        'prediction_error': 0.5,
        'policy_entropy': 0.3,
        'q_values': [0.6, 0.4],
        'threat_level': 0.7  # Numeric, не строка!
    }
    
    # Это должно работать
    aggregates = field.compute_exposure(obs_numeric, reward=0.5, trial=1)
    
    assert isinstance(aggregates.X_risk, (int, float))
    assert isinstance(aggregates.X_opp, (int, float))
    assert isinstance(aggregates.D_est, (int, float))
    
    # Observation со строковой меткой должно выбросить ошибку
    obs_string = {
        'object_label': 'snake',  # Строка!
        'threat_level': 'high'
    }
    
    with pytest.raises(ValueError, match="No Ready Semions"):
        field.compute_exposure(obs_string, reward=0.5, trial=1)
    
    print("✓ PASS: No semantic labels in action basin")


# =============================================================================
# TEST 6: Exposure field tolerance for action basin equivalence
# =============================================================================

def test_exposure_field_tolerance():
    """
    Test: Exposure field имеет разумную толерантность для action basin.
    """
    field = ExposureField()
    
    # Серия схожих стимулов
    exposures = []
    for i in range(5):
        obs = {
            'prediction_error': 0.5 + i * 0.02,
            'policy_entropy': 0.3 - i * 0.02,
            'q_values': [0.6, 0.4]
        }
        agg = field.compute_exposure(obs, reward=0.5, trial=i)
        exposures.append(agg)
    
    # Проверяем что все exposures близки (в пределах action basin)
    for i in range(1, len(exposures)):
        risk_diff = abs(exposures[0].X_risk - exposures[i].X_risk)
        opp_diff = abs(exposures[0].X_opp - exposures[i].X_opp)
        
        assert risk_diff < 0.3, f"Risk drift too large: {risk_diff}"
        assert opp_diff < 0.3, f"Opportunity drift too large: {opp_diff}"
    
    print("✓ PASS: Exposure field tolerance")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Stage 3.0: Test Action-Basin Equivalence")
    print("=" * 70)
    
    test_perceptually_different_similar_exposure()
    test_similar_exposure_similar_mode()
    test_different_exposure_different_mode()
    test_action_basin_through_aggregation()
    test_no_semantic_labels_in_basin()
    test_exposure_field_tolerance()
    
    print("=" * 70)
    print("All tests passed!")
    print("=" * 70)