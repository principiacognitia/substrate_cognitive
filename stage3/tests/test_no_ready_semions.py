"""
Stage 3.0: Test No Ready Semions at Port.

Design Constraint:
Сенсорный вход не содержит предварительно классифицированных объектных меток
("snake", "stick", "food", "predator").

Вместо этого — валентно/экспозиционно-взвешенные эмбеддинги и агрегированные метрики.

Acceptance Criterion 8.2:
The sensory interface does not expose categorical object labels to the Gate.

Author: Alex Snow (Aleksey L. Snigirov)
License: MIT
"""

import pytest
import numpy as np
from stage3.core.gate_inputs import InstantDiagnostics, ExposureAggregates, TemporalState, GateInput
from stage3.core.exposure_field import ExposureField


# =============================================================================
# TEST 1: InstantDiagnostics rejects string values
# =============================================================================

def test_instant_diagnostics_rejects_strings():
    """
    Test: InstantDiagnostics не принимает строковые значения.
    
    Rationale:
    u_t должен содержать только numeric агрегаты (PE, entropy, volatility),
    не семантические метки.
    """
    # Попытка создать InstantDiagnostics со строкой должна выбросить ошибку
    with pytest.raises(ValueError, match="cannot be string"):
        InstantDiagnostics(
            u_delta="high_prediction_error",  # type: ignore
            u_entropy=0.3,
            u_volatility=0.2
        )
    
    # Попытка создать со строкой в u_entropy
    with pytest.raises(ValueError, match="cannot be string"):
        InstantDiagnostics(
            u_delta=0.5,
            u_entropy="high_entropy",  # type: ignore
            u_volatility=0.2
        )
    
    print("✓ PASS: InstantDiagnostics rejects string values")


def test_instant_diagnostics_accepts_numeric():
    """
    Test: InstantDiagnostics принимает numeric значения.
    """
    # Это должно работать без ошибок
    instant = InstantDiagnostics(
        u_delta=0.5,
        u_entropy=0.3,
        u_volatility=0.2,
        trial=1
    )
    
    assert isinstance(instant.u_delta, float)
    assert isinstance(instant.u_entropy, float)
    assert isinstance(instant.u_volatility, float)
    assert isinstance(instant.trial, int)
    
    # validate_no_semions должен вернуть True
    assert instant.validate_no_semions() == True
    
    print("✓ PASS: InstantDiagnostics accepts numeric values")


# =============================================================================
# TEST 2: ExposureAggregates rejects string values
# =============================================================================

def test_exposure_aggregates_rejects_strings():
    """
    Test: ExposureAggregates не принимает строковые значения в aggregates.
    
    Rationale:
    X_risk, X_opp, D_est должны быть numeric агрегатами,
    не категориальными метками.
    """
    # Попытка создать ExposureAggregates со строкой в X_risk
    with pytest.raises(ValueError, match="must be numeric"):
        ExposureAggregates(
            X_risk="high_risk",  # type: ignore
            X_opp=0.1,
            D_est=0.5
        )
    
    print("✓ PASS: ExposureAggregates rejects string values")


def test_exposure_aggregates_accepts_numeric():
    """
    Test: ExposureAggregates принимает numeric значения.
    """
    # Это должно работать без ошибок
    exposure = ExposureAggregates(
        X_risk=0.3,
        X_opp=0.1,
        D_est=0.5,
        region_id="test_region",  # region_id может быть str (для логирования)
        trial=1
    )
    
    assert isinstance(exposure.X_risk, float)
    assert isinstance(exposure.X_opp, float)
    assert isinstance(exposure.D_est, float)
    
    # validate_no_semions должен вернуть True (region_id — исключение)
    assert exposure.validate_no_semions() == True
    
    print("✓ PASS: ExposureAggregates accepts numeric values")


# =============================================================================
# TEST 3: ExposureField rejects observations with string labels
# =============================================================================

def test_exposure_field_rejects_string_labels():
    """
    Test: ExposureField отвергает observation со строковыми метками.
    
    Rationale:
    Design Constraint: No Ready Semions at Port.
    Сенсорный вход не должен содержать предварительно классифицированных
    объектных меток ("snake", "stick", etc.).
    """
    field = ExposureField()
    
    # Observation со строковой меткой должно выбросить ошибку
    bad_observation = {
        'object_label': 'snake',  # Строковая метка!
        'threat_level': 'high'
    }
    
    with pytest.raises(ValueError, match="No Ready Semions constraint violated"):
        field.compute_exposure(bad_observation)
    
    print("✓ PASS: ExposureField rejects string labels")


def test_exposure_field_accepts_numeric_observation():
    """
    Test: ExposureField принимает observation с numeric features.
    """
    field = ExposureField()
    
    # Observation с numeric features должно работать
    good_observation = {
        'prediction_error': 0.5,
        'policy_entropy': 0.3,
        'q_values': [0.6, 0.4],
        'expected_reward': 0.5
    }
    
    # Это должно работать без ошибок
    aggregates = field.compute_exposure(
        observation=good_observation,
        reward=0.8,
        trial=1
    )
    
    assert isinstance(aggregates.X_risk, float)
    assert isinstance(aggregates.X_opp, float)
    assert isinstance(aggregates.D_est, float)
    assert aggregates.validate_no_semions() == True
    
    print("✓ PASS: ExposureField accepts numeric observation")


# =============================================================================
# TEST 4: GateInput validates all layers
# =============================================================================

def test_gate_input_validates_all_layers():
    """
    Test: GateInput валидирует все три слоя на отсутствие семионов.
    """
    # Валидный GateInput
    valid_input = GateInput(
        instant=InstantDiagnostics(u_delta=0.5, u_entropy=0.3, u_volatility=0.2),
        exposure=ExposureAggregates(X_risk=0.1, X_opp=0.1, D_est=0.5),
        temporal=TemporalState(h_risk=0.0, h_opp=0.0, h_time=0)
    )
    
    is_valid, errors = valid_input.validate_all_layers()
    assert is_valid == True
    assert len(errors) == 0
    
    print("✓ PASS: GateInput validates all layers")


def test_gate_input_catches_semions():
    """
    Test: GateInput обнаруживает семионы в слоях.
    """
    # Невалидный InstantDiagnostics (со строкой)
    try:
        invalid_input = GateInput(
            instant=InstantDiagnostics(
                u_delta="high_error",  # type: ignore
                u_entropy=0.3,
                u_volatility=0.2
            ),
            exposure=ExposureAggregates.zeros(),
            temporal=TemporalState.zeros()
        )
        # __post_init__ должен выбросить ошибку
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Ожидаемое поведение
    
    print("✓ PASS: GateInput catches semions")


# =============================================================================
# TEST 5: Action-basin equivalence (semantic-free routing)
# =============================================================================

def test_action_basin_equivalence_semantic_free():
    """
    Test: Action-basin equivalence без семантических меток.
    
    Rationale:
    Два перцептивно разных стимула с близким X-profile должны
    давать схожий mode bias, несмотря на отсутствие объектных меток.
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
    
    print("✓ PASS: Action-basin equivalence (semantic-free)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Stage 3.0: Test No Ready Semions at Port")
    print("=" * 70)
    
    test_instant_diagnostics_rejects_strings()
    test_instant_diagnostics_accepts_numeric()
    test_exposure_aggregates_rejects_strings()
    test_exposure_aggregates_accepts_numeric()
    test_exposure_field_rejects_string_labels()
    test_exposure_field_accepts_numeric_observation()
    test_gate_input_validates_all_layers()
    test_gate_input_catches_semions()
    test_action_basin_equivalence_semantic_free()
    
    print("=" * 70)
    print("All tests passed!")
    print("=" * 70)