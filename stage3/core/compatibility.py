"""
Stage 3.0: Stage 2 Compatibility Shims.

Обеспечивает backward compatibility между Stage 2 и Stage 3.

Design Constraints:
- Backward Compatibility: при нулевых exposure/temporal trace, Stage 3.0 
  должен воспроизводить Stage 2 behavior с теми же конфигами
- No New Logic: этот модуль только предоставляет shims, не содержит новой логики
- Degradation Mode: явный режим деградации для тестирования

Acceptance Criterion:
С нулевыми X_risk, X_opp, D_est, h_risk, h_opp, h_time и ограниченным набором 
режимов {EXPLOIT, EXPLORE}, Stage 3.0 должен воспроизводить Stage 2A/2B 
behavior с точностью до численной погрешности (< 5% отклонение метрик).

Author: Alex Snow (Aleksey L. Snigirov)
License: MIT
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from stage3.core.gate_modes import GateMode
from stage3.core.gate_inputs import (
    GateInput,
    InstantDiagnostics,
    ExposureAggregates,
    TemporalState
)


@dataclass
class Stage2CompatConfig:
    """
    Конфигурация для Stage 2 compatibility mode.
    
    Attributes:
        enabled: Включить ли compatibility mode
        zero_exposure: Обнулить ли ExposureAggregates
        zero_temporal: Обнулить ли TemporalState
        restrict_modes: Ограничить ли режимы до {EXPLOIT, EXPLORE}
        tolerance: Допустимое отклонение от Stage 2 метрик (default: 5%)
    """
    enabled: bool = False
    zero_exposure: bool = True
    zero_temporal: bool = True
    restrict_modes: bool = True
    tolerance: float = 0.05
    
    def __post_init__(self):
        """Валидация конфигурации."""
        if not 0.0 <= self.tolerance <= 1.0:
            raise ValueError(f"tolerance must be in [0, 1]: {self.tolerance}")


class Stage2CompatShim:
    """
    Shim для обеспечения Stage 2 backward compatibility.
    
    Usage:
        shim = Stage2CompatShim()
        gate_input = shim.create_compat_input(instant_diagnostics)
        # gate_input будет иметь нулевые exposure и temporal
    """
    
    def __init__(self, config: Optional[Stage2CompatConfig] = None):
        """
        Инициализирует shim.
        
        Args:
            config: Конфигурация compatibility mode
        """
        self.config = config or Stage2CompatConfig()
    
    def create_compat_input(self, instant: InstantDiagnostics) -> GateInput:
        """
        Создаёт GateInput совместимый со Stage 2.
        
        Args:
            instant: InstantDiagnostics из Stage 2
        
        Returns:
            GateInput с нулевыми exposure и temporal (если enabled)
        """
        if not self.config.enabled:
            # Return as-is if compatibility mode is disabled
            return GateInput(
                instant=instant,
                exposure=ExposureAggregates(),
                temporal=TemporalState()
            )
        
        # Zero exposure if configured
        if self.config.zero_exposure:
            exposure = ExposureAggregates.zeros()
        else:
            exposure = ExposureAggregates()
        
        # Zero temporal if configured
        if self.config.zero_temporal:
            temporal = TemporalState.zeros()
        else:
            temporal = TemporalState()
        
        return GateInput(
            instant=instant,
            exposure=exposure,
            temporal=temporal
        )
    
    def get_allowed_modes(self) -> list:
        """
        Возвращает список разрешённых режимов для compatibility mode.
        
        Returns:
            List[GateMode]: Разрешённые режимы
        """
        if self.config.enabled and self.config.restrict_modes:
            # Stage 2 had only EXPLOIT and EXPLORE
            return [GateMode.EXPLOIT, GateMode.EXPLORE]
        else:
            # All Stage 3 modes allowed
            return [GateMode.EXPLOIT, GateMode.EXPLORE, 
                    GateMode.EXPLOIT_SAFE, GateMode.ABSENCE_CHECK]
    
    def is_compat_mode(self) -> bool:
        """
        Проверяет включён ли compatibility mode.
        
        Returns:
            True если compatibility mode активен
        """
        return self.config.enabled
    
    def validate_stage2_equivalence(
        self,
        stage3_metrics: Dict[str, float],
        stage2_metrics: Dict[str, float]
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Проверяет эквивалентность Stage 3 и Stage 2 метрик.
        
        Args:
            stage3_metrics: Метрики из Stage 3.0 запуска
            stage2_metrics: Метрики из Stage 2 запуска (baseline)
        
        Returns:
            (is_equivalent, deviations_dict)
            is_equivalent: True если отклонения в пределах tolerance
            deviations_dict: Отклонения по каждой метрике
        """
        deviations = {}
        is_equivalent = True
        
        for key, stage2_value in stage2_metrics.items():
            if key not in stage3_metrics:
                deviations[key] = float('inf')
                is_equivalent = False
                continue
            
            stage3_value = stage3_metrics[key]
            
            # Вычисляем относительное отклонение
            if stage2_value == 0:
                if stage3_value == 0:
                    deviation = 0.0
                else:
                    deviation = float('inf')
            else:
                deviation = abs(stage3_value - stage2_value) / abs(stage2_value)
            
            deviations[key] = deviation
            
            if deviation > self.config.tolerance:
                is_equivalent = False
        
        return is_equivalent, deviations


def create_stage2_compat_input(
    u_delta: float = 0.0,
    u_entropy: float = 0.0,
    u_volatility: float = 0.0
) -> GateInput:
    """
    Convenience function для создания Stage 2-compatible GateInput.
    
    Args:
        u_delta: Unsigned prediction error
        u_entropy: Policy entropy
        u_volatility: Volatility estimate
    
    Returns:
        GateInput с нулевыми Stage 3 полями
    """
    return GateInput(
        instant=InstantDiagnostics(
            u_delta=u_delta,
            u_entropy=u_entropy,
            u_volatility=u_volatility
        ),
        exposure=ExposureAggregates.zeros(),
        temporal=TemporalState.zeros()
    )


def verify_backward_compatibility(
    stage3_results: Dict,
    stage2_results: Dict,
    tolerance: float = 0.05
) -> Tuple[bool, str]:
    """
    Verifies that Stage 3.0 results match Stage 2 within tolerance.
    
    Args:
        stage3_results: Dict с метриками из Stage 3.0
        stage2_results: Dict с метриками из Stage 2
        tolerance: Допустимое отклонение (default: 5%)
    
    Returns:
        (passed, message)
        passed: True если тест пройден
        message: Сообщение с деталями
    """
    key_metrics = [
        'mb_interaction_coef',
        'mf_interaction_p',
        'switching_latency_median',
        'perseveration_median'
    ]
    
    deviations = {}
    max_deviation = 0.0
    
    for key in key_metrics:
        if key not in stage3_results or key not in stage2_results:
            continue
        
        s3_val = stage3_results[key]
        s2_val = stage2_results[key]
        
        if s2_val == 0:
            deviation = 0.0 if s3_val == 0 else float('inf')
        else:
            deviation = abs(s3_val - s2_val) / abs(s2_val)
        
        deviations[key] = deviation
        max_deviation = max(max_deviation, deviation)
    
    passed = max_deviation <= tolerance
    
    if passed:
        message = f"✓ Backward compatibility verified (max deviation: {max_deviation:.2%})"
    else:
        message = f"✗ Backward compatibility failed (max deviation: {max_deviation:.2%} > {tolerance:.2%})"
        message += f"\nDeviations: {deviations}"
    
    return passed, message


# =============================================================================
# TESTS (для быстрой проверки)
# =============================================================================

def test_compat_input_creation():
    """
    Test: Compatibility Input Creation.
    
    Проверяет что create_stage2_compat_input создаёт правильные нулевые поля.
    """
    gate_input = create_stage2_compat_input(
        u_delta=0.5,
        u_entropy=0.3,
        u_volatility=0.2
    )
    
    # Check instant diagnostics
    assert gate_input.instant.u_delta == 0.5
    assert gate_input.instant.u_entropy == 0.3
    assert gate_input.instant.u_volatility == 0.2
    
    # Check exposure is zeroed
    assert gate_input.exposure.X_risk == 0.0
    assert gate_input.exposure.X_opp == 0.0
    assert gate_input.exposure.D_est == 0.0
    
    # Check temporal is zeroed
    assert gate_input.temporal.h_risk == 0.0
    assert gate_input.temporal.h_opp == 0.0
    assert gate_input.temporal.h_time == 0
    
    print("✓ PASS: Compatibility Input Creation")
    return True


def test_compat_shim():
    """
    Test: Compatibility Shim.
    
    Проверяет что Stage2CompatShim работает корректно.
    """
    # Test with compatibility enabled
    shim_enabled = Stage2CompatShim(Stage2CompatConfig(enabled=True))
    instant = InstantDiagnostics(u_delta=0.5, u_entropy=0.3, u_volatility=0.2)
    gate_input = shim_enabled.create_compat_input(instant)
    
    assert gate_input.exposure.X_risk == 0.0
    assert gate_input.temporal.h_risk == 0.0
    
    # Test allowed modes
    allowed_modes = shim_enabled.get_allowed_modes()
    assert GateMode.EXPLOIT in allowed_modes
    assert GateMode.EXPLORE in allowed_modes
    assert GateMode.EXPLOIT_SAFE not in allowed_modes  # Restricted in compat mode
    assert GateMode.ABSENCE_CHECK not in allowed_modes  # Restricted in compat mode
    
    # Test with compatibility disabled
    shim_disabled = Stage2CompatShim(Stage2CompatConfig(enabled=False))
    allowed_modes = shim_disabled.get_allowed_modes()
    assert len(allowed_modes) == 4  # All modes allowed
    
    print("✓ PASS: Compatibility Shim")
    return True


def test_backward_compatibility_verification():
    """
    Test: Backward Compatibility Verification.
    
    Проверяет что verify_backward_compatibility работает корректно.
    """
    # Test passing case
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
    
    # Test failing case
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
    
    print("✓ PASS: Backward Compatibility Verification")
    return True


def test_validate_stage2_equivalence():
    """
    Test: Validate Stage 2 Equivalence.
    
    Проверяет что Stage2CompatShim.validate_stage2_equivalence работает.
    """
    shim = Stage2CompatShim(Stage2CompatConfig(enabled=True, tolerance=0.1))
    
    stage3_metrics = {
        'latency': 36.0,
        'perseveration': 8.5
    }
    stage2_metrics = {
        'latency': 35.0,
        'perseveration': 8.0
    }
    
    is_equivalent, deviations = shim.validate_stage2_equivalence(
        stage3_metrics,
        stage2_metrics
    )
    
    # Deviations should be within 10%
    assert is_equivalent, f"Should be equivalent: {deviations}"
    assert deviations['latency'] < 0.1
    assert deviations['perseveration'] < 0.1
    
    print("✓ PASS: Validate Stage 2 Equivalence")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("Stage 3.0: Compatibility Shims — Unit Tests")
    print("=" * 70)
    
    test_compat_input_creation()
    test_compat_shim()
    test_backward_compatibility_verification()
    test_validate_stage2_equivalence()
    
    print("=" * 70)
    print("All tests completed!")
    print("=" * 70)