"""
Stage 3.0: Gate Stage 3 Routing Logic.

Реализует выбор режима Gate через threshold cascade (не argmax!).

Design Constraints:
- One Gate only: все режимы обрабатываются единым Gate
- No argmax arbitration: режимы выбираются через каскад порогов/прерываний
- No ready semions: Gate получает агрегаты (X_risk, X_opp, D_est), не метки
- Ontology ≠ Engineering: ν, O, X — аналитическое разложение для трассировки

Threshold Cascade Priority (highest → lowest):
1. ABSENCE_CHECK — high stakes + poor visibility + sufficient safe window
2. EXPLOIT_SAFE — critical threat exposure (bypasses standard explore barrier)
3. EXPLORE — high uncertainty + low threat (Stage 2 logic)
4. EXPLOIT — default (low uncertainty)

Author: Alex Snow (Aleksey L. Snigirov)
License: MIT
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from stage3.core.gate_modes import GateMode, MODE_PRIORITY
from stage3.core.gate_inputs import GateInput, InstantDiagnostics, ExposureAggregates, TemporalState


@dataclass
class GateThresholds:
    """
    Конфигурация порогов для threshold cascade.
    
    Attributes:
        critical_risk_threshold: Порог для EXPLOIT_SAFE override
        suspicion_threshold: Порог для ABSENCE_CHECK (h_risk)
        visibility_threshold: Максимальная D_est для ABSENCE_CHECK
        safe_window_threshold: Минимальный h_time для ABSENCE_CHECK
        theta_mb: Mode switch threshold (Stage 2 наследие)
        theta_u: Uncertainty baseline (Stage 2 наследие)
    """
    critical_risk_threshold: float = 0.7
    suspicion_threshold: float = 0.5
    visibility_threshold: float = 0.3
    safe_window_threshold: int = 50
    theta_mb: float = 0.30
    theta_u: float = 1.5
    
    def __post_init__(self):
        """Валидация конфигурации."""
        if not 0.0 <= self.critical_risk_threshold <= 1.0:
            raise ValueError(f"critical_risk_threshold must be in [0, 1]: {self.critical_risk_threshold}")
        if not 0.0 <= self.suspicion_threshold <= 1.0:
            raise ValueError(f"suspicion_threshold must be in [0, 1]: {self.suspicion_threshold}")
        if not 0.0 <= self.visibility_threshold <= 1.0:
            raise ValueError(f"visibility_threshold must be in [0, 1]: {self.visibility_threshold}")
        if self.safe_window_threshold < 0:
            raise ValueError(f"safe_window_threshold must be >= 0: {self.safe_window_threshold}")
        if not 0.0 <= self.theta_mb <= 1.0:
            raise ValueError(f"theta_mb must be in [0, 1]: {self.theta_mb}")
        if self.theta_u < 0:
            raise ValueError(f"theta_u must be >= 0: {self.theta_u}")


class GateStage3:
    """
    Gate Stage 3 с threshold cascade routing.
    
    Design Principle: No argmax arbitration.
    
    Mode selection is implemented as a cascade of thresholds/interrupts,
    where stronger signals override weaker ones. No global scoring over
    all modes is performed.
    
    Usage:
        gate = GateStage3()
        mode = gate.select_mode(gate_input)
    """
    
    def __init__(self, thresholds: Optional[GateThresholds] = None):
        """
        Инициализирует Gate.
        
        Args:
            thresholds: Конфигурация порогов (default: GateThresholds())
        """
        self.thresholds = thresholds or GateThresholds()
    
    def select_mode(self, gate_input: GateInput) -> Tuple[GateMode, Dict]:
        """
        Выбирает режим через threshold cascade.
        
        Args:
            gate_input: Комбинированный вход (3 слоя)
        
        Returns:
            (selected_mode, metadata_dict)
            metadata_dict содержит:
                - mode_scores: Dict[GateMode, float] — scores для логирования
                - winning_constraint: str — какой порог сработал
                - gate_state_snapshot: Dict — snapshot состояния для трассировки
        """
        # Извлекаем три слоя
        instant = gate_input.instant
        exposure = gate_input.exposure
        temporal = gate_input.temporal
        
        # Инициализируем metadata для трассировки
        metadata = {
            'mode_scores': {},
            'winning_constraint': None,
            'gate_state_snapshot': {
                'u_delta': instant.u_delta,
                'u_entropy': instant.u_entropy,
                'u_volatility': instant.u_volatility,
                'X_risk': exposure.X_risk,
                'X_opp': exposure.X_opp,
                'D_est': exposure.D_est,
                'h_risk': temporal.h_risk,
                'h_opp': temporal.h_opp,
                'h_time': temporal.h_time
            }
        }
        
        # =====================================================================
        # THRESHOLD CASCADE (priority: highest → lowest)
        # =====================================================================
        
        # --- Priority 1: ABSENCE_CHECK ---------------------------------------
        # High stakes + poor visibility + sufficient safe window
        absence_check_score = self._compute_absence_check_score(exposure, temporal)
        metadata['mode_scores'][GateMode.ABSENCE_CHECK] = absence_check_score
        
        if self._should_trigger_absence_check(exposure, temporal):
            metadata['winning_constraint'] = 'absence_trigger (high stakes + poor visibility + safe window)'
            return GateMode.ABSENCE_CHECK, metadata
        
        # --- Priority 2: EXPLOIT_SAFE ----------------------------------------
        # Critical threat exposure (bypasses standard explore barrier)
        exploit_safe_score = self._compute_exploit_safe_score(exposure, temporal)
        metadata['mode_scores'][GateMode.EXPLOIT_SAFE] = exploit_safe_score
        
        if self._should_trigger_exploit_safe(exposure):
            metadata['winning_constraint'] = 'threat_override (critical risk exposure)'
            return GateMode.EXPLOIT_SAFE, metadata
        
        # --- Priority 3: EXPLORE ---------------------------------------------
        # High uncertainty + low threat (Stage 2 logic preserved)
        explore_score = self._compute_explore_score(instant, exposure, temporal)
        metadata['mode_scores'][GateMode.EXPLORE] = explore_score
        
        if self._should_trigger_explore(instant, exposure, temporal):
            metadata['winning_constraint'] = 'standard_arbitration (high uncertainty)'
            return GateMode.EXPLORE, metadata
        
        # --- Priority 4: EXPLOIT (default) -----------------------------------
        # Low uncertainty (Stage 2 default)
        exploit_score = self._compute_exploit_score(instant, exposure, temporal)
        metadata['mode_scores'][GateMode.EXPLOIT] = exploit_score
        
        metadata['winning_constraint'] = 'default (low uncertainty)'
        return GateMode.EXPLOIT, metadata
    
    def _compute_absence_check_score(self, exposure: ExposureAggregates, temporal: TemporalState) -> float:
        """
        Вычисляет score для ABSENCE_CHECK.
        
        Score высокий когда:
        - h_risk > suspicion_threshold (подозрение)
        - D_est < visibility_threshold (плохая видимость)
        - h_time > safe_window_threshold (можно позволить проверку)
        """
        # Нормализуем h_time в [0, 1] для сравнения
        h_time_normalized = min(1.0, temporal.h_time / 100.0)
        
        # Score = комбинация трёх условий
        suspicion_score = max(0.0, temporal.h_risk - self.thresholds.suspicion_threshold)
        visibility_score = max(0.0, self.thresholds.visibility_threshold - exposure.D_est)
        safe_window_score = h_time_normalized if temporal.h_time > self.thresholds.safe_window_threshold else 0.0
        
        # Агрегируем (multiplicative — все условия должны выполняться)
        score = suspicion_score * visibility_score * safe_window_score
        
        return float(score)
    
    def _compute_exploit_safe_score(self, exposure: ExposureAggregates, temporal: TemporalState) -> float:
        """
        Вычисляет score для EXPLOIT_SAFE.
        
        Score высокий когда:
        - X_risk высокий (текущая угроза)
        - h_risk высокий (накопленная угроза)
        """
        # Комбинация текущей и накопленной угрозы
        current_threat = exposure.X_risk
        accumulated_threat = temporal.h_risk
        
        # Score = weighted combination
        score = 0.6 * current_threat + 0.4 * accumulated_threat
        
        return float(score)
    
    def _compute_explore_score(self, instant: InstantDiagnostics, 
                                exposure: ExposureAggregates, 
                                temporal: TemporalState) -> float:
        """
        Вычисляет score для EXPLORE (Stage 2 logic + exposure modulation).
        
        Score высокий когда:
        - u_volatility высокий (неопределённость)
        - u_entropy высокий (policy uncertainty)
        - X_risk низкий (угроза не блокирует)
        """
        # Stage 2 uncertainty pressure
        uncertainty_pressure = instant.u_volatility * instant.u_entropy
        
        # Exposure modulation (threat reduces exploration)
        threat_modulation = 1.0 - exposure.X_risk
        
        # Score = uncertainty × (1 - threat)
        score = uncertainty_pressure * threat_modulation
        
        return float(score)
    
    def _compute_exploit_score(self, instant: InstantDiagnostics,
                                exposure: ExposureAggregates,
                                temporal: TemporalState) -> float:
        """
        Вычисляет score для EXPLOIT (default).
        
        Score высокий когда:
        - u_volatility низкий (уверенность)
        - u_entropy низкий (policy certainty)
        - X_opp высокий (возможность для эксплуатации)
        """
        # Stage 2 certainty
        certainty = (1.0 - instant.u_volatility) * (1.0 - instant.u_entropy)
        
        # Opportunity modulation
        opportunity_modulation = exposure.X_opp
        
        # Score = certainty × opportunity
        score = certainty * (0.5 + 0.5 * opportunity_modulation)
        
        return float(score)
    
    def _should_trigger_absence_check(self, exposure: ExposureAggregates, 
                                       temporal: TemporalState) -> bool:
        """
        Проверяет условие для ABSENCE_CHECK.
        
        Condition:
            h_risk > suspicion_threshold AND
            D_est < visibility_threshold AND
            h_time > safe_window_threshold
        """
        suspicion = temporal.h_risk > self.thresholds.suspicion_threshold
        poor_visibility = exposure.D_est < self.thresholds.visibility_threshold
        safe_window = temporal.h_time > self.thresholds.safe_window_threshold
        
        return suspicion and poor_visibility and safe_window
    
    def _should_trigger_exploit_safe(self, exposure: ExposureAggregates) -> bool:
        """
        Проверяет условие для EXPLOIT_SAFE (threat override).
        
        Condition:
            X_risk > critical_risk_threshold
        """
        return exposure.X_risk > self.thresholds.critical_risk_threshold
    
# ИСПРАВЛЕНО:
    def _should_trigger_explore(self, instant: InstantDiagnostics,
                                exposure: ExposureAggregates,
                                temporal: TemporalState) -> bool:  # ← Добавлен параметр
        """
        Проверяет условие для EXPLORE (Stage 2 logic preserved).
        
        Condition:
            sigma(w^T u_t - theta_U) × (1 - V_G) > theta_MB
            AND X_risk не блокирует
        
        For Stage 3.0, V_G is approximated from temporal state:
            V_G ≈ h_risk (accumulated threat reduces exploration)
        """
        # Compute uncertainty pressure (Stage 2 logic)
        uncertainty_pressure = instant.u_volatility * instant.u_entropy
        
        # Approximate V_G from temporal state (accumulated threat)
        v_g_approx = temporal.h_risk  # ← ТЕПЕРЬ РАБОТАЕТ
        
        # Stage 2 gate equation
        gate_output = uncertainty_pressure * (1.0 - v_g_approx)
        
        # Check threshold AND threat not blocking
        explore_triggered = gate_output > self.thresholds.theta_mb
        threat_not_blocking = exposure.X_risk < self.thresholds.critical_risk_threshold
        
        return explore_triggered and threat_not_blocking
    
    def reset(self):
        """
        Сбрасывает внутреннее состояние Gate (для новых эпизодов).
        """
        pass  # GateStage3 stateless; temporal state managed separately


# =============================================================================
# CONVENIENCE FUNCTIONS (для тестирования)
# =============================================================================

def create_test_gate_input(
    u_delta: float = 0.5,
    u_entropy: float = 0.3,
    u_volatility: float = 0.2,
    X_risk: float = 0.1,
    X_opp: float = 0.1,
    D_est: float = 0.5,
    h_risk: float = 0.0,
    h_opp: float = 0.0,
    h_time: int = 0
) -> GateInput:
    """
    Создает тестовый GateInput.
    
    Только для тестов!
    """
    from stage3.core.gate_inputs import InstantDiagnostics, ExposureAggregates, TemporalState
    
    return GateInput(
        instant=InstantDiagnostics(
            u_delta=u_delta,
            u_entropy=u_entropy,
            u_volatility=u_volatility
        ),
        exposure=ExposureAggregates(
            X_risk=X_risk,
            X_opp=X_opp,
            D_est=D_est
        ),
        temporal=TemporalState(
            h_risk=h_risk,
            h_opp=h_opp,
            h_time=h_time
        )
    )


# =============================================================================
# TESTS (для быстрой проверки)
# =============================================================================

def test_threshold_cascade():
    """
    Test: Threshold Cascade Routing.
    
    Проверяет что режимы выбираются через каскад порогов.
    """
    gate = GateStage3()
    
    # Test 1: Default (EXPLOIT)
    gate_input = create_test_gate_input(
        u_delta=0.1, u_entropy=0.1, u_volatility=0.1,
        X_risk=0.1, X_opp=0.5, D_est=0.8,
        h_risk=0.1, h_opp=0.1, h_time=10
    )
    mode, metadata = gate.select_mode(gate_input)
    assert mode == GateMode.EXPLOIT, f"Expected EXPLOIT, got {mode}"
    assert metadata['winning_constraint'] == 'default (low uncertainty)'
    
    # Test 2: EXPLORE (high uncertainty)
    gate_input = create_test_gate_input(
        u_delta=0.8, u_entropy=0.8, u_volatility=0.8,
        X_risk=0.1, X_opp=0.5, D_est=0.8,
        h_risk=0.1, h_opp=0.1, h_time=10
    )
    mode, metadata = gate.select_mode(gate_input)
    assert mode == GateMode.EXPLORE, f"Expected EXPLORE, got {mode}"
    assert metadata['winning_constraint'] == 'standard_arbitration (high uncertainty)'
    
    # Test 3: EXPLOIT_SAFE (critical threat)
    gate_input = create_test_gate_input(
        u_delta=0.1, u_entropy=0.1, u_volatility=0.1,
        X_risk=0.8, X_opp=0.1, D_est=0.8,  # High X_risk
        h_risk=0.5, h_opp=0.1, h_time=10
    )
    mode, metadata = gate.select_mode(gate_input)
    assert mode == GateMode.EXPLOIT_SAFE, f"Expected EXPLOIT_SAFE, got {mode}"
    assert metadata['winning_constraint'] == 'threat_override (critical risk exposure)'
    
    # Test 4: ABSENCE_CHECK (suspicion + poor visibility + safe window)
    gate_input = create_test_gate_input(
        u_delta=0.1, u_entropy=0.1, u_volatility=0.1,
        X_risk=0.5, X_opp=0.1, D_est=0.2,  # Poor visibility
        h_risk=0.6, h_opp=0.1, h_time=100  # High suspicion + safe window
    )
    mode, metadata = gate.select_mode(gate_input)
    assert mode == GateMode.ABSENCE_CHECK, f"Expected ABSENCE_CHECK, got {mode}"
    assert metadata['winning_constraint'] == 'absence_trigger (high stakes + poor visibility + safe window)'
    
    print("✓ PASS: Threshold Cascade Routing")
    return True


def test_no_argmax():
    """
    Test: No Argmax Arbitration.
    
    Проверяет что Gate не использует global scoring/argmax.
    """
    gate = GateStage3()
    gate_input = create_test_gate_input()
    
    mode, metadata = gate.select_mode(gate_input)
    
    # Проверяем что mode_scores есть для всех режимов (для логирования)
    assert len(metadata['mode_scores']) == 4, "Should have scores for all 4 modes"
    
    # Проверяем что winning_constraint указан (threshold cascade, не argmax)
    assert metadata['winning_constraint'] is not None, "Should specify which threshold triggered"
    
    # Проверяем что gate_state_snapshot есть для трассировки
    assert 'gate_state_snapshot' in metadata, "Should have gate state snapshot"
    
    print("✓ PASS: No Argmax Arbitration")
    return True


def test_backward_compatibility():
    """
    Test: Backward Compatibility (Stage 2 emulation).
    
    Проверяет что при нулевых exposure/temporal, Gate деградирует в Stage 2.
    """
    gate = GateStage3()
    
    # Нулевые exposure и temporal
    gate_input = create_test_gate_input(
        u_delta=0.5, u_entropy=0.3, u_volatility=0.2,
        X_risk=0.0, X_opp=0.0, D_est=0.0,
        h_risk=0.0, h_opp=0.0, h_time=0
    )
    
    mode, metadata = gate.select_mode(gate_input)
    
    # При нулевых exposure/temporal, должны работать Stage 2 правила
    # (EXPLOIT или EXPLORE в зависимости от uncertainty)
    assert mode in [GateMode.EXPLOIT, GateMode.EXPLORE], f"Unexpected mode: {mode}"
    
    print("✓ PASS: Backward Compatibility (Stage 2 emulation)")
    return True


def test_threat_override():
    """
    Test: Threat Override (EXPLOIT_SAFE bypasses EXPLORE).
    
    Проверяет что critical threat переопределяет standard exploration.
    """
    gate = GateStage3()
    
    # High uncertainty (normally would trigger EXPLORE)
    # BUT also critical threat (should override to EXPLOIT_SAFE)
    gate_input = create_test_gate_input(
        u_delta=0.8, u_entropy=0.8, u_volatility=0.8,  # High uncertainty
        X_risk=0.8, X_opp=0.1, D_est=0.8,  # Critical threat
        h_risk=0.5, h_opp=0.1, h_time=10
    )
    
    mode, metadata = gate.select_mode(gate_input)
    
    # Threat override should win over exploration
    assert mode == GateMode.EXPLOIT_SAFE, f"Expected EXPLOIT_SAFE (threat override), got {mode}"
    assert 'threat_override' in metadata['winning_constraint']
    
    print("✓ PASS: Threat Override (EXPLOIT_SAFE bypasses EXPLORE)")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("Stage 3.0: Gate Stage 3 — Unit Tests")
    print("=" * 70)
    
    test_threshold_cascade()
    test_no_argmax()
    test_backward_compatibility()
    test_threat_override()
    
    print("=" * 70)
    print("All tests completed!")
    print("=" * 70)