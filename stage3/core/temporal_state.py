"""
Stage 3.0: Temporal State Update Logic.

Реализует обновление сжатой временной истории (h_t):
- h_risk: exponentially smoothed risky exposure trace
- h_opp: exponentially smoothed opportunity trace
- h_time: time trace since last high-salience event

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
    Конфигурация для temporal state update.
    
    Attributes:
        lambda_risk: Decay rate для h_risk (default: 0.9)
        lambda_opp: Decay rate для h_opp (default: 0.9)
        salience_threshold: Порог для сброса h_time (default: 0.5)
        one_shot_threshold: Порог амплитуды для one-shot update (default: 5.0)
        one_shot_boost: Множитель для one-shot update (default: 2.0)
    """
    lambda_risk: float = 0.9
    lambda_opp: float = 0.9
    salience_threshold: float = 0.5
    one_shot_threshold: float = 5.0
    one_shot_boost: float = 2.0
    
    def __post_init__(self):
        """Валидация конфигурации."""
        if not 0.0 <= self.lambda_risk <= 1.0:
            raise ValueError(f"lambda_risk must be in [0, 1]: {self.lambda_risk}")
        if not 0.0 <= self.lambda_opp <= 1.0:
            raise ValueError(f"lambda_opp must be in [0, 1]: {self.lambda_opp}")
        if self.salience_threshold < 0:
            raise ValueError(f"salience_threshold must be >= 0: {self.salience_threshold}")
        if self.one_shot_threshold < 0:
            raise ValueError(f"one_shot_threshold must be >= 0: {self.one_shot_threshold}")
        if self.one_shot_boost < 0:
            raise ValueError(f"one_shot_boost must be >= 0: {self.one_shot_boost}")


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
        Обновляет temporal state.
        
        Args:
            state: Текущий TemporalState
            X_risk: Current risky exposure (из exposure_field.py)
            X_opp: Current opportunity exposure (из exposure_field.py)
            salience: Salience of current event (unsigned PE или novelty)
            stakes: Stakes/modulator для one-shot detection
        
        Returns:
            Обновлённый TemporalState
        """
        # Вычисляем surprise amplitude для one-shot detection
        surprise_amplitude = salience * stakes
        
        # One-shot regime detection
        is_one_shot = surprise_amplitude > self.config.one_shot_threshold
        
        # === Обновление h_risk ===
        # Базовое обновление (exponential smoothing)
        h_risk_new = (
            self.config.lambda_risk * state.h_risk +
            (1 - self.config.lambda_risk) * X_risk
        )
        
        # One-shot boost (если high-amplitude event)
        if is_one_shot:
            h_risk_new += self.config.one_shot_boost * X_risk
            h_risk_new = np.clip(h_risk_new, 0.0, 1.0)  # Clip to valid range
        
        # === Обновление h_opp ===
        # Базовое обновление (exponential smoothing)
        h_opp_new = (
            self.config.lambda_opp * state.h_opp +
            (1 - self.config.lambda_opp) * X_opp
        )
        
        # One-shot boost (если high-amplitude event)
        if is_one_shot:
            h_opp_new += self.config.one_shot_boost * X_opp
            h_opp_new = np.clip(h_opp_new, 0.0, 1.0)  # Clip to valid range
        
        # === Обновление h_time ===
        # Сброс если salient event, иначе инкремент
        if salience > self.config.salience_threshold:
            h_time_new = 0
        else:
            h_time_new = state.h_time + 1
        
        # === Создаём новый state ===
        new_state = TemporalState(
            h_risk=float(h_risk_new),
            h_opp=float(h_opp_new),
            h_time=int(h_time_new),
            # Debug metadata
            one_shot_pending=is_one_shot,
            one_shot_amplitude=float(surprise_amplitude)
        )
        
        return new_state
    
    def reset(self) -> TemporalState:
        """
        Сбрасывает temporal state к нулю.
        
        Returns:
            TemporalState.zeros()
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
    """
    Test: Temporal State Update.
    
    Проверяет что h_risk, h_opp, h_time обновляются корректно.
    """
    updater = TemporalStateUpdater()
    state = TemporalState.zeros()
    
    # Обновление с низким salience
    state = updater.update(
        state=state,
        X_risk=0.3,
        X_opp=0.1,
        salience=0.2,  # Низкий salience
        stakes=1.0
    )
    
    assert state.h_time == 1, f"h_time should be 1, got {state.h_time}"
    assert state.h_risk > 0, "h_risk should increase"
    assert state.h_opp > 0, "h_opp should increase"
    assert not state.one_shot_pending, "Should not be one-shot"
    
    print("✓ PASS: Temporal State Update")
    return True


def test_one_shot_as_update_regime():
    """
    Test: One-Shot as Amplitude-Dependent Update Regime.
    
    Проверяет что one-shot — это не отдельный модуль, а режим обновления.
    """
    updater = TemporalStateUpdater(
        TemporalStateConfig(one_shot_threshold=5.0, one_shot_boost=2.0)
    )
    state = TemporalState.zeros()
    
    # Low amplitude (many-shot)
    state_low = updater.update(
        state=state,
        X_risk=0.3,
        X_opp=0.1,
        salience=0.2,  # Low salience
        stakes=1.0     # Low stakes
        # surprise_amplitude = 0.2 < 5.0 → no one-shot
    )
    
    assert not state_low.one_shot_pending, "Low amplitude should not trigger one-shot"
    
    # High amplitude (one-shot)
    state_high = updater.update(
        state=state,
        X_risk=0.3,
        X_opp=0.1,
        salience=0.9,  # High salience
        stakes=10.0    # High stakes
        # surprise_amplitude = 9.0 > 5.0 → one-shot!
    )
    
    assert state_high.one_shot_pending, "High amplitude should trigger one-shot"
    assert state_high.one_shot_amplitude > 5.0, "Amplitude should be recorded"
    assert state_high.h_risk > state_low.h_risk, "One-shot should boost h_risk"
    assert state_high.h_opp > state_low.h_opp, "One-shot should boost h_opp"
    
    print("✓ PASS: One-Shot as Update Regime")
    return True


def test_h_time_reset():
    """
    Test: h_time Reset on Salient Event.
    
    Проверяет что h_time сбрасывается при salient event.
    """
    updater = TemporalStateUpdater(
        TemporalStateConfig(salience_threshold=0.5)
    )
    state = TemporalState(h_risk=0.0, h_opp=0.0, h_time=10)
    
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
    
    print("✓ PASS: h_time Reset on Salient Event")
    return True


def test_trace_dynamics():
    """
    Test: Trace Dynamics Simulation.
    
    Проверяет что get_trace_dynamics работает корректно.
    """
    updater = TemporalStateUpdater()
    initial_state = TemporalState.zeros()
    
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
    
    print("✓ PASS: Trace Dynamics Simulation")
    return True


def test_backward_compatibility():
    """
    Test: Backward Compatibility (Stage 2 emulation).
    
    Проверяет что при нулевых exposure trace, behavior деградирует в Stage 2.
    """
    updater = TemporalStateUpdater()
    state = TemporalState.zeros()
    
    # Нулевые exposure → trace должны оставаться близки к нулю
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
    
    print("✓ PASS: Backward Compatibility (Stage 2 emulation)")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("Stage 3.0: Temporal State — Unit Tests")
    print("=" * 70)
    
    test_temporal_state_update()
    test_one_shot_as_update_regime()
    test_h_time_reset()
    test_trace_dynamics()
    test_backward_compatibility()
    
    print("=" * 70)
    print("All tests completed!")
    print("=" * 70)