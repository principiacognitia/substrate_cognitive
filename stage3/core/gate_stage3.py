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
        theta_mb: Mode switch threshold
        theta_u: Uncertainty baseline
        safe_drive_weight_current: Вес текущей угрозы в safe override
        safe_drive_weight_temporal: Вес накопленной угрозы в safe override
        w_volatility: Вес u_volatility в uncertainty
        w_entropy: Вес u_entropy в uncertainty
        v_g_weight_hrisk: Вес h_risk в аппроксимации gate viscosity
        v_g_weight_xrisk: Вес X_risk в аппроксимации gate viscosity
    """
    critical_risk_threshold: float = 0.7
    suspicion_threshold: float = 0.5
    visibility_threshold: float = 0.3
    safe_window_threshold: int = 50

    theta_mb: float = 0.30
    theta_u: float = 1.5

    safe_drive_weight_current: float = 0.6
    safe_drive_weight_temporal: float = 0.4

    w_volatility: float = 1.0
    w_entropy: float = 1.0
    v_g_weight_hrisk: float = 0.7
    v_g_weight_xrisk: float = 0.3

    def __post_init__(self):
        """Валидация конфигурации."""
        for name, value in [
            ("critical_risk_threshold", self.critical_risk_threshold),
            ("suspicion_threshold", self.suspicion_threshold),
            ("visibility_threshold", self.visibility_threshold),
            ("theta_mb", self.theta_mb),
            ("safe_drive_weight_current", self.safe_drive_weight_current),
            ("safe_drive_weight_temporal", self.safe_drive_weight_temporal),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]: {value}")

        if self.safe_window_threshold < 0:
            raise ValueError(f"safe_window_threshold must be >= 0: {self.safe_window_threshold}")
        if self.theta_u < 0:
            raise ValueError(f"theta_u must be >= 0: {self.theta_u}")

        if (self.safe_drive_weight_current + self.safe_drive_weight_temporal) <= 0:
            raise ValueError("safe drive weights must sum to > 0")
        
        # 3.2B patch С validation for uncertainty signal weights and gate viscosity weights
        if self.safe_drive_weight_current < 0:
            raise ValueError(f"safe_drive_weight_current must be >= 0: {self.safe_drive_weight_current}")
        if self.safe_drive_weight_temporal < 0:
            raise ValueError(f"safe_drive_weight_temporal must be >= 0: {self.safe_drive_weight_temporal}")

        if self.w_volatility < 0:
            raise ValueError(f"w_volatility must be >= 0: {self.w_volatility}")
        if self.w_entropy < 0:
            raise ValueError(f"w_entropy must be >= 0: {self.w_entropy}")
        if self.v_g_weight_hrisk < 0:
            raise ValueError(f"v_g_weight_hrisk must be >= 0: {self.v_g_weight_hrisk}")
        if self.v_g_weight_xrisk < 0:
            raise ValueError(f"v_g_weight_xrisk must be >= 0: {self.v_g_weight_xrisk}")


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

    def _compute_uncertainty_signal(self, instant: InstantDiagnostics) -> float:
        """
        Вычисляет uncertainty через sigmoid(w_volatility * u_volatility +
        w_entropy * u_entropy - theta_u).
        """
        raw = (
            self.thresholds.w_volatility * instant.u_volatility +
            self.thresholds.w_entropy * instant.u_entropy -
            self.thresholds.theta_u
        )
        return float(1.0 / (1.0 + np.exp(-raw)))

    def _compute_vg_approx(self, exposure: ExposureAggregates, temporal: TemporalState) -> float:
        """
        Аппроксимация gate viscosity из accumulated risk и current risk.
        """
        vg = (
            self.thresholds.v_g_weight_hrisk * temporal.h_risk +
            self.thresholds.v_g_weight_xrisk * exposure.X_risk
        )
        return float(np.clip(vg, 0.0, 1.0))
    
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
                'h_time': temporal.h_time,
                'theta_u': self.thresholds.theta_u,
                'theta_mb': self.thresholds.theta_mb
            }
        }

        metadata['gate_state_snapshot']['uncertainty_signal'] = self._compute_uncertainty_signal(instant)
        metadata['gate_state_snapshot']['v_g_approx'] = self._compute_vg_approx(exposure, temporal)

        # Debug/Trace: Вычисляем все scores для логирования (хотя выбор будет через каскад)
        uncertainty_signal = self._compute_uncertainty_signal(instant)
        v_g_approx = self._compute_vg_approx(exposure, temporal)
        safe_drive = self._compute_exploit_safe_score(exposure, temporal)
        explore_gate_output = uncertainty_signal * (1.0 - v_g_approx)
        absence_triggered = self._should_trigger_absence_check(exposure, temporal)
        exploit_safe_triggered = safe_drive > self.thresholds.critical_risk_threshold
        explore_triggered = explore_gate_output > self.thresholds.theta_mb

        metadata['gate_state_snapshot'].update({
            'uncertainty_signal': uncertainty_signal,
            'v_g_approx': v_g_approx,
            'safe_drive': safe_drive,
            'explore_gate_output': explore_gate_output,

            'critical_risk_threshold': self.thresholds.critical_risk_threshold,
            'safe_drive_weight_current': self.thresholds.safe_drive_weight_current,
            'safe_drive_weight_temporal': self.thresholds.safe_drive_weight_temporal,
            'w_volatility': self.thresholds.w_volatility,
            'w_entropy': self.thresholds.w_entropy,
            'v_g_weight_hrisk': self.thresholds.v_g_weight_hrisk,
            'v_g_weight_xrisk': self.thresholds.v_g_weight_xrisk,

            'absence_triggered': absence_triggered,
            'exploit_safe_triggered': exploit_safe_triggered,
            'explore_triggered': explore_triggered,
        })
        
        # =====================================================================
        # THRESHOLD CASCADE (priority: highest → lowest)
        # =====================================================================
        
        # --- Priority 1: ABSENCE_CHECK ---------------------------------------
        # High stakes + poor visibility + sufficient safe window
        absence_check_score = self._compute_absence_check_score(exposure, temporal)
        metadata['mode_scores'][GateMode.ABSENCE_CHECK] = absence_check_score
        
#       if self._should_trigger_absence_check(exposure, temporal):
        if absence_triggered:
            metadata['winning_constraint'] = 'absence_trigger (high stakes + poor visibility + safe window)'
            return GateMode.ABSENCE_CHECK, metadata
        
        # --- Priority 2: EXPLOIT_SAFE ----------------------------------------
        # Critical threat exposure (bypasses standard explore barrier)
        exploit_safe_score = self._compute_exploit_safe_score(exposure, temporal)
        metadata['mode_scores'][GateMode.EXPLOIT_SAFE] = exploit_safe_score
        
#       if self._should_trigger_exploit_safe(exposure, temporal):
        if exploit_safe_triggered:
            metadata['winning_constraint'] = 'threat_override (safe_drive from current + temporal risk)'
            return GateMode.EXPLOIT_SAFE, metadata
        
        # --- Priority 3: EXPLORE ---------------------------------------------
        # High uncertainty + low threat (Stage 2 logic preserved)
        explore_score = self._compute_explore_score(instant, exposure, temporal)
        metadata['mode_scores'][GateMode.EXPLORE] = explore_score
        
#       if self._should_trigger_explore(instant, exposure, temporal):
        if explore_triggered:
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
        Patch B: Вычисляет score для EXPLOIT_SAFE.
        safe_drive = w_x * X_risk + w_h * h_risk
        
        Score высокий когда:
        - X_risk высокий (текущая угроза)
        - h_risk высокий (накопленная угроза)
        """
        current_threat = exposure.X_risk
        accumulated_threat = temporal.h_risk

        score = (
            self.thresholds.safe_drive_weight_current * current_threat +
            self.thresholds.safe_drive_weight_temporal * accumulated_threat
        )

        return float(score)
    
    def _compute_explore_score(self, instant: InstantDiagnostics,
                                exposure: ExposureAggregates,
                                temporal: TemporalState) -> float:
        """
        Вычисляет score для EXPLORE через uncertainty-threshold stage.

        Score высокий когда:
        - uncertainty signal высокий
        - effective gate viscosity низкая
        """
        uncertainty_signal = self._compute_uncertainty_signal(instant)
        v_g_approx = self._compute_vg_approx(exposure, temporal)

        score = uncertainty_signal * (1.0 - v_g_approx)

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
    
    def _should_trigger_exploit_safe(
        self,
        exposure: ExposureAggregates,
        temporal: TemporalState
    ) -> bool:
        """
        Patch B:
        EXPLOIT_SAFE depends on both current and accumulated threat.

        safe_drive = w_x * X_risk + w_h * h_risk
        trigger if safe_drive > critical_risk_threshold
        """
        safe_drive = self._compute_exploit_safe_score(exposure, temporal)
        return safe_drive > self.thresholds.critical_risk_threshold
    
    # Stage 3.2B patch C: EXPLORE depends on uncertainty signal and gate viscosity approximation
    def _should_trigger_explore(self, instant: InstantDiagnostics,
                                exposure: ExposureAggregates,
                                temporal: TemporalState) -> bool:
        """
        Проверяет условие для EXPLORE.

        Condition:
            sigmoid(w_volatility * u_volatility + w_entropy * u_entropy - theta_u)
            * (1 - v_g_approx) > theta_mb

        where:
            v_g_approx = clip(v_g_weight_hrisk * h_risk + v_g_weight_xrisk * X_risk, 0, 1)

        EXPLOIT_SAFE остаётся выше по каскаду и должен отрабатывать раньше.
        """
        uncertainty_signal = self._compute_uncertainty_signal(instant)
        v_g_approx = self._compute_vg_approx(exposure, temporal)

        gate_output = uncertainty_signal * (1.0 - v_g_approx)

        return gate_output > self.thresholds.theta_mb
    
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
    assert metadata['winning_constraint'] == 'threat_override (safe_drive from current + temporal risk)'
    
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