"""
Stage 3.1B: Temporal State Update Logic.

Реализует обновление сжатой временной истории (h_t):
- h_risk: exponentially smoothed short-lived risky exposure trace
- h_opp: exponentially smoothed short-lived opportunity trace
- h_time: time trace since last high-salience event

Importance traces:
- q_neg: accumulated negative-event importance
- q_pos: accumulated positive-event importance

Design Constraints:
- One-Shot как режим обновления (amplitude-dependent), не отдельный модуль
- Минимальный набор trace (3 core только)
- Никакой логики в gate_inputs.py — всё обновление здесь

Формулы обновления:
    h_risk(t+1) = λ_r * h_risk(t) + (1 - λ_r) * X_risk(t)
    h_opp(t+1)  = λ_o * h_opp(t) + (1 - λ_o) * X_opp(t)
    h_time(t+1) = 0, если salience > τ_sal
                = h_time(t) + 1, иначе

One-Shot Regime:
    Если surprise_amplitude > one_shot_threshold:
        h_risk ← h_risk + one_shot_boost * X_risk
        h_opp  ← h_opp  + one_shot_boost * X_opp

    Design Principle:
    Importance is not a symbolic tag and not a scheduler flag.
    It is a continuous state variable that modulates the time constants
    of already existing trace dynamics.
        
Author: Alex Snow (Aleksey L. Snigirov)
License: MIT
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np

from stage3.core.gate_inputs import TemporalState


@dataclass
class TemporalStateConfig:
    """
    Конфигурация непрерывной temporal dynamics для Stage 3.1B rebuild.

    ВАЖНО:
    - никаких persistence windows
    - никаких hidden schedulers
    - one-shot effect возникает через importance traces q_neg / q_pos
    - q traces модулируют time constants already-existing temporal traces
    """

    # Base update rates for h traces
    # Интерпретация: чем меньше lambda, тем медленнее текущий вход переписывает trace
    lambda_risk: float = 0.10
    lambda_opp: float = 0.10

    # Importance trace decay
    rho_neg: float = 0.98
    rho_pos: float = 0.95

    # Importance gain
    k_neg: float = 1.0
    k_pos: float = 0.7

    # Baseline field threshold below which input is treated as ordinary background
    theta_baseline: float = 0.25

    # Debug/diagnostic threshold for classifying event as one-shot
    theta_shot: float = 5.0

    # Coupling from importance traces to effective update rates
    w_neg_to_risk: float = 2.0
    w_pos_to_opp: float = 1.0

    # Existing
    salience_threshold: float = 0.5

    # Safety clip
    q_clip: float = 10.0

    def __post_init__(self):
        for name, value in [
            ("lambda_risk", self.lambda_risk),
            ("lambda_opp", self.lambda_opp),
            ("rho_neg", self.rho_neg),
            ("rho_pos", self.rho_pos),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]: {value}")

        for name, value in [
            ("k_neg", self.k_neg),
            ("k_pos", self.k_pos),
            ("theta_baseline", self.theta_baseline),
            ("theta_shot", self.theta_shot),
            ("w_neg_to_risk", self.w_neg_to_risk),
            ("w_pos_to_opp", self.w_pos_to_opp),
            ("salience_threshold", self.salience_threshold),
            ("q_clip", self.q_clip),
        ]:
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0: {value}")


class TemporalStateUpdater:
    """
    Обновляет TemporalState на каждом шаге.
    
    Design Principle: One-shot is an amplitude-dependent update regime,
    not a separate memory module.
    
    Usage:
        updater = TemporalStateUpdater()
        state = TemporalState.zeros()
        state = updater.update(state, X_risk=0.3, X_opp=0.1, salience=0.8, stakes=1.0)
    """
    
    def __init__(self, config: Optional[TemporalStateConfig] = None):
        """
        Инициализирует updater.
        
        Args:
            config: Конфигурация (default: TemporalStateConfig())
        """
        self.config = config or TemporalStateConfig()
        
    def update(
        self,
        state: TemporalState,
        X_risk: float,
        X_opp: float,
        salience: float,
        stakes: float = 1.0
    ) -> TemporalState:
        """
        Обновляет temporal state через непрерывные importance traces.

        Principle:
        - q_neg / q_pos не являются ярлыками событий
        - они являются long-lived continuous state variables
        - их роль — модулировать effective update rates already-existing traces
        """

        X_risk = float(X_risk)
        X_opp = float(X_opp)
        salience = float(salience)
        stakes = float(stakes)

        surprise_amplitude = max(0.0, salience) * max(0.0, stakes)
        is_one_shot = surprise_amplitude > self.config.theta_shot

        # ------------------------------------------------------------------
        # A. Importance drive from current field
        # ------------------------------------------------------------------
        risk_excess = max(0.0, X_risk - self.config.theta_baseline)
        opp_excess = max(0.0, X_opp - self.config.theta_baseline)

        shot_neg = risk_excess * surprise_amplitude
        shot_pos = opp_excess * surprise_amplitude

        # ------------------------------------------------------------------
        # B. Continuous importance traces
        # ------------------------------------------------------------------
        q_neg_new = self.config.rho_neg * state.q_neg + self.config.k_neg * shot_neg
        q_pos_new = self.config.rho_pos * state.q_pos + self.config.k_pos * shot_pos

        q_neg_new = float(np.clip(q_neg_new, 0.0, self.config.q_clip))
        q_pos_new = float(np.clip(q_pos_new, 0.0, self.config.q_clip))

        # ------------------------------------------------------------------
        # C. Effective update rates
        # High q -> slower relaxation -> smaller lambda_eff
        # ------------------------------------------------------------------
        lambda_risk_eff = self.config.lambda_risk / (1.0 + self.config.w_neg_to_risk * q_neg_new)
        lambda_opp_eff = self.config.lambda_opp / (1.0 + self.config.w_pos_to_opp * q_pos_new)

        lambda_risk_eff = float(np.clip(lambda_risk_eff, 1e-6, 1.0))
        lambda_opp_eff = float(np.clip(lambda_opp_eff, 1e-6, 1.0))

        # ------------------------------------------------------------------
        # D. Trace update
        # ------------------------------------------------------------------
        h_risk_new = (1.0 - lambda_risk_eff) * state.h_risk + lambda_risk_eff * X_risk
        h_opp_new = (1.0 - lambda_opp_eff) * state.h_opp + lambda_opp_eff * X_opp

        h_risk_new = float(np.clip(h_risk_new, 0.0, 1.0))
        h_opp_new = float(np.clip(h_opp_new, 0.0, 1.0))

        # ------------------------------------------------------------------
        # E. h_time
        # ------------------------------------------------------------------
        if salience > self.config.salience_threshold:
            h_time_new = 0
        else:
            h_time_new = state.h_time + 1

        # ------------------------------------------------------------------
        # F. Debug classification only
        # ------------------------------------------------------------------
        if is_one_shot and shot_neg > shot_pos and shot_neg > 0.0:
            one_shot_type = "negative"
        elif is_one_shot and shot_pos > shot_neg and shot_pos > 0.0:
            one_shot_type = "positive"
        else:
            one_shot_type = "none"

        return TemporalState(
            h_risk=h_risk_new,
            h_opp=h_opp_new,
            h_time=int(h_time_new),
            q_neg=q_neg_new,
            q_pos=q_pos_new,
            one_shot_pending=bool(is_one_shot),
            one_shot_amplitude=float(surprise_amplitude),
            one_shot_type=one_shot_type
        )

    def reset(self) -> TemporalState:
        """
        Сбрасывает temporal state к нулю.
        """
        return TemporalState.zeros()
    
    def get_trace_dynamics(
        self,
        initial_state: TemporalState,
        X_risk_sequence: list,
        X_opp_sequence: list,
        salience_sequence: list,
        stakes_sequence: Optional[list] = None
    ) -> list:
        """
        Проигрывает последовательность обновлений для анализа динамики.
        
        Args:
            initial_state: Начальный state
            X_risk_sequence: Последовательность X_risk values
            X_opp_sequence: Последовательность X_opp values
            salience_sequence: Последовательность salience values
            stakes_sequence: Последовательность stakes values (optional)
        
        Returns:
            List of TemporalState после каждого шага
        """
        if stakes_sequence is None:
            stakes_sequence = [1.0] * len(X_risk_sequence)
        
        states = [initial_state]
        current_state = initial_state
        
        for i in range(len(X_risk_sequence)):
            current_state = self.update(
                state=current_state,
                X_risk=X_risk_sequence[i],
                X_opp=X_opp_sequence[i],
                salience=salience_sequence[i],
                stakes=stakes_sequence[i]
            )
            states.append(current_state)
        
        return states


# =============================================================================
# CONVENIENCE FUNCTIONS (для тестирования)
# =============================================================================

def create_test_state(
    h_risk: float = 0.0,
    h_opp: float = 0.0,
    h_time: int = 0
) -> TemporalState:
    """
    Создает тестовый TemporalState.
    
    Только для тестов!
    """
    return TemporalState(
        h_risk=h_risk,
        h_opp=h_opp,
        h_time=h_time
    )


# =============================================================================
# TESTS (для быстрой проверки)
# =============================================================================

def test_temporal_state_update():
    updater = TemporalStateUpdater()
    state = TemporalState.zeros()

    state = updater.update(
        state=state,
        X_risk=0.3,
        X_opp=0.1,
        salience=0.2,
        stakes=1.0
    )

    assert state.h_time == 1, f"h_time should be 1, got {state.h_time}"
    assert state.h_risk > 0, "h_risk should increase"
    assert state.h_opp > 0, "h_opp should increase"
    assert state.q_neg == 0.0, f"q_neg should remain 0 for non-extreme event, got {state.q_neg}"
    assert state.q_pos == 0.0, f"q_pos should remain 0 for non-extreme event, got {state.q_pos}"
    print("✓ PASS: Temporal State Update")
    return True


def test_q_neg_rises_on_high_negative_event():
    updater = TemporalStateUpdater()
    state = TemporalState.zeros()

    state = updater.update(
        state=state,
        X_risk=0.6,
        X_opp=0.0,
        salience=0.9,
        stakes=10.0
    )

    assert state.q_neg > 0.0, f"q_neg should rise, got {state.q_neg}"
    assert state.q_pos == 0.0, f"q_pos should stay 0, got {state.q_pos}"
    assert state.one_shot_type == "negative", f"expected negative shot, got {state.one_shot_type}"
    print("✓ PASS: q_neg rises on high negative event")
    return True


def test_q_neg_slows_risk_relaxation():
    updater = TemporalStateUpdater()
    state = TemporalState.zeros()

    state = updater.update(
        state=state,
        X_risk=0.6,
        X_opp=0.0,
        salience=0.9,
        stakes=10.0
    )
    shocked_h = state.h_risk

    for _ in range(10):
        state = updater.update(
            state=state,
            X_risk=0.1,
            X_opp=0.1,
            salience=0.1,
            stakes=1.0
        )

    assert state.q_neg > 0.0, f"q_neg should still be > 0, got {state.q_neg}"
    assert 0.20 < state.h_risk < shocked_h, f"h_risk should decay slowly, got {state.h_risk}"
    print("✓ PASS: q_neg slows risk relaxation")
    return True


def test_q_traces_stay_quiet_without_extreme_events():
    updater = TemporalStateUpdater()
    state = TemporalState.zeros()

    for _ in range(20):
        state = updater.update(
            state=state,
            X_risk=0.1,
            X_opp=0.1,
            salience=0.1,
            stakes=1.0
        )

    assert state.q_neg == 0.0, f"q_neg should stay 0, got {state.q_neg}"
    assert state.q_pos == 0.0, f"q_pos should stay 0, got {state.q_pos}"
    print("✓ PASS: q traces stay quiet without extreme events")
    return True


def test_backward_compatibility():
    updater = TemporalStateUpdater()
    state = TemporalState.zeros()

    for _ in range(10):
        state = updater.update(
            state=state,
            X_risk=0.0,
            X_opp=0.0,
            salience=0.0,
            stakes=0.0
        )

    assert state.h_risk < 0.01, f"h_risk should be ~0, got {state.h_risk}"
    assert state.h_opp < 0.01, f"h_opp should be ~0, got {state.h_opp}"
    assert state.q_neg == 0.0, f"q_neg should be ~0, got {state.q_neg}"
    assert state.q_pos == 0.0, f"q_pos should be ~0, got {state.q_pos}"
    print("✓ PASS: Backward Compatibility")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("Stage 3.1B Rebuild: Temporal State — Smoke Tests")
    print("=" * 70)

    test_temporal_state_update()
    test_q_neg_rises_on_high_negative_event()
    test_q_neg_slows_risk_relaxation()
    test_q_traces_stay_quiet_without_extreme_events()
    test_backward_compatibility()

    print("=" * 70)
    print("All tests completed!")
    print("=" * 70)