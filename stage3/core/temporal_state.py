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
    # При высоком q_neg, lambda_risk_eff будет меньше, что замедляет обновление 
    # h_risk и позволяет ему сохраняться дольше.
    lambda_risk: float = 0.10
    lambda_opp: float = 0.10
    lambda_input_risk: float = 0.10
    lambda_input_opp: float = 0.10

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
    w_qneg_input: float = 1.0
    w_qpos_input: float = 0.0


    # Existing
    salience_threshold: float = 0.5

    # Safety clip
    q_clip: float = 10.0

    def __post_init__(self):
        for name, value in [
            ("lambda_risk", self.lambda_risk),
            ("lambda_opp", self.lambda_opp),
            ("lambda_input_risk", self.lambda_input_risk),
            ("lambda_input_opp", self.lambda_input_opp),
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
            ("w_qneg_input", self.w_qneg_input),
            ("w_qpos_input", self.w_qpos_input),
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
        stakes: float = 1.0,
        event_X_risk: Optional[float] = None,
        event_X_opp: Optional[float] = None,
        event_source_id: Optional[str] = None,
        source_input_map: Optional[Dict[str, float]] = None,
    ) -> TemporalState:
        """
        Обновляет temporal state через непрерывные importance traces.

        Principle:
        - q_neg / q_pos не являются ярлыками событий
        - они являются continuous state variables
        - global q_pos/h_opp работают как прежде
        - positive source-local carryover хранится отдельно по source_id
        """

        X_risk = float(X_risk)
        X_opp = float(X_opp)
        salience = float(salience)
        stakes = float(stakes)

        surprise_amplitude = max(0.0, salience) * max(0.0, stakes)
        surprise_excess = max(0.0, surprise_amplitude - self.config.theta_shot)
        is_one_shot = surprise_excess > 0.0

        # ------------------------------------------------------------------
        # A. Importance drive from event source field
        # ------------------------------------------------------------------
        has_event_override = (event_X_risk is not None) or (event_X_opp is not None)

        if has_event_override:
            source_X_risk = float(0.0 if event_X_risk is None else event_X_risk)
            source_X_opp = float(0.0 if event_X_opp is None else event_X_opp)
        else:
            source_X_risk = float(X_risk)
            source_X_opp = float(X_opp)

        risk_excess = max(0.0, source_X_risk - self.config.theta_baseline)
        opp_excess = max(0.0, source_X_opp - self.config.theta_baseline)

        shot_neg = risk_excess * surprise_excess
        shot_pos = opp_excess * surprise_excess

        # ------------------------------------------------------------------
        # B. Global continuous importance traces
        # ------------------------------------------------------------------
        q_neg_new = self.config.rho_neg * state.q_neg + self.config.k_neg * shot_neg
        q_pos_new = self.config.rho_pos * state.q_pos + self.config.k_pos * shot_pos

        q_neg_new = float(np.clip(q_neg_new, 0.0, self.config.q_clip))
        q_pos_new = float(np.clip(q_pos_new, 0.0, self.config.q_clip))

        # ------------------------------------------------------------------
        # C. Source-local positive importance traces
        # ------------------------------------------------------------------
        event_source_id_str = ""
        if event_source_id is not None and str(event_source_id).strip():
            event_source_id_str = str(event_source_id).strip()

        q_pos_local_new: Dict[str, float] = {}
        local_q_keys = set(state.q_pos_local.keys())
        if event_source_id_str:
            local_q_keys.add(event_source_id_str)

        for sid in local_q_keys:
            prev_local_q = float(state.q_pos_local.get(sid, 0.0))
            local_shot_pos = shot_pos if (sid == event_source_id_str) else 0.0
            new_local_q = self.config.rho_pos * prev_local_q + self.config.k_pos * local_shot_pos
            new_local_q = float(np.clip(new_local_q, 0.0, self.config.q_clip))
            if new_local_q > 1e-9:
                q_pos_local_new[sid] = new_local_q

        # ------------------------------------------------------------------
        # D. Effective global update rates
        # ------------------------------------------------------------------
        lambda_risk_eff = self.config.lambda_risk / (1.0 + self.config.w_neg_to_risk * state.q_neg)
        lambda_opp_eff = self.config.lambda_opp / (1.0 + self.config.w_pos_to_opp * state.q_pos)

        lambda_risk_eff = float(np.clip(lambda_risk_eff, 1e-6, 1.0))
        lambda_opp_eff = float(np.clip(lambda_opp_eff, 1e-6, 1.0))

        # ------------------------------------------------------------------
        # E. Global trace update
        # ------------------------------------------------------------------
        risk_gain = 1.0 + self.config.w_qneg_input * (state.q_neg / (1.0 + state.q_neg))
        X_risk_eff = float(np.clip(X_risk * risk_gain, 0.0, 1.0))

        opp_gain = 1.0 + self.config.w_qpos_input * (state.q_pos / (1.0 + state.q_pos))
        X_opp_eff = float(np.clip(X_opp * opp_gain, 0.0, 1.0))

        h_risk_new = (
            (1.0 - lambda_risk_eff) * state.h_risk +
            lambda_risk_eff * X_risk_eff
        )

        h_opp_new = (
            (1.0 - lambda_opp_eff) * state.h_opp +
            lambda_opp_eff * X_opp_eff
        )

        h_risk_new = float(np.clip(h_risk_new, 0.0, 1.0))
        h_opp_new = float(np.clip(h_opp_new, 0.0, 1.0))

        # ------------------------------------------------------------------
        # F. Source-local positive trace update
        # source_input_map is policy-side numeric affordance input, not Gate routing.
        # ------------------------------------------------------------------
        source_input_map_clean: Dict[str, float] = {}
        if isinstance(source_input_map, dict):
            for raw_sid, raw_val in source_input_map.items():
                sid = str(raw_sid).strip() if raw_sid is not None else ""
                if not sid:
                    continue
                try:
                    source_input_map_clean[sid] = float(np.clip(float(raw_val), 0.0, 1.0))
                except (TypeError, ValueError):
                    continue

        # --------------------------------------------------------------
        # STRICT source-local rule:
        # only sources with an actual positive event trace may receive
        # ongoing local appetitive input.
        #
        # This prevents ordinary option_reward_values at junction from
        # seeding h_opp_local on every visible option.
        # --------------------------------------------------------------
        active_local_keys = set(q_pos_local_new.keys())
        if event_source_id_str:
            active_local_keys.add(event_source_id_str)

        h_opp_local_new: Dict[str, float] = {}

        # Existing local traces may decay even if no longer active.
        local_h_keys = set(state.h_opp_local.keys()) | active_local_keys

        for sid in local_h_keys:
            prev_local_h = float(state.h_opp_local.get(sid, 0.0))
            local_q = float(q_pos_local_new.get(sid, state.q_pos_local.get(sid, 0.0)))

            lambda_opp_local_eff = self.config.lambda_opp / (1.0 + self.config.w_pos_to_opp * local_q)
            lambda_opp_local_eff = float(np.clip(lambda_opp_local_eff, 1e-6, 1.0))

            local_gain = 1.0 + self.config.w_qpos_input * (local_q / (1.0 + local_q))

            # Only active event-tagged sources receive ongoing input.
            # All non-active sources simply decay toward zero.
            if sid in active_local_keys:
                local_input = float(source_input_map_clean.get(sid, 0.0))
            else:
                local_input = 0.0

            local_input_eff = float(np.clip(local_input * local_gain, 0.0, 1.0))

            new_local_h = (
                (1.0 - lambda_opp_local_eff) * prev_local_h +
                lambda_opp_local_eff * local_input_eff
            )
            new_local_h = float(np.clip(new_local_h, 0.0, 1.0))

            if new_local_h > 1e-9 or local_q > 1e-9:
                h_opp_local_new[sid] = new_local_h

        # ------------------------------------------------------------------
        # G. h_time
        # ------------------------------------------------------------------
        if salience > self.config.salience_threshold:
            h_time_new = 0
        else:
            h_time_new = state.h_time + 1

        # ------------------------------------------------------------------
        # H. Debug classification only
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
            q_pos_local=q_pos_local_new,
            h_opp_local=h_opp_local_new,
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
    # For a low-salience, low-stakes event, q_neg and q_pos should remain  zero.
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

def test_q_pos_rises_on_high_positive_event():
    updater = TemporalStateUpdater()
    state = TemporalState.zeros()

    state = updater.update(
        state=state,
        X_risk=0.0,
        X_opp=1.0,
        salience=0.9,
        stakes=10.0
    )

    assert state.q_pos > 0.0, f"q_pos should rise, got {state.q_pos}"
    assert state.q_neg == 0.0, f"q_neg should stay 0, got {state.q_neg}"
    assert state.one_shot_type == "positive", f"expected positive shot, got {state.one_shot_type}"
    print("✓ PASS: q_pos rises on high positive event")
    return True


def test_q_neg_slows_risk_relaxation():
    """
    Pure relaxation test.

    Here q_neg should affect only relaxation speed, not effective input gain.
    Therefore w_qneg_input is forced to 0.0 in both branches.
    """
    cfg_modulated = TemporalStateConfig(
        lambda_risk=0.10,
        lambda_opp=0.10,
        lambda_input_risk=0.10,
        lambda_input_opp=0.10,
        rho_neg=0.98,
        rho_pos=0.95,
        k_neg=1.0,
        k_pos=0.7,
        theta_baseline=0.25,
        theta_shot=5.0,
        w_neg_to_risk=2.0,
        w_pos_to_opp=1.0,
        w_qneg_input=0.0,
    )

    cfg_control = TemporalStateConfig(
        lambda_risk=0.10,
        lambda_opp=0.10,
        lambda_input_risk=0.10,
        lambda_input_opp=0.10,
        rho_neg=0.98,
        rho_pos=0.95,
        k_neg=1.0,
        k_pos=0.7,
        theta_baseline=0.25,
        theta_shot=5.0,
        w_neg_to_risk=0.0,
        w_pos_to_opp=1.0,
        w_qneg_input=0.0,
    )

    upd_mod = TemporalStateUpdater(cfg_modulated)
    upd_ctl = TemporalStateUpdater(cfg_control)

    state_mod = TemporalState.zeros()
    state_ctl = TemporalState.zeros()

    # Shock step
    state_mod = upd_mod.update(
        state=state_mod,
        X_risk=0.6,
        X_opp=0.0,
        salience=0.9,
        stakes=10.0
    )
    state_ctl = upd_ctl.update(
        state=state_ctl,
        X_risk=0.6,
        X_opp=0.0,
        salience=0.9,
        stakes=10.0
    )

    shocked_h_mod = state_mod.h_risk
    shocked_h_ctl = state_ctl.h_risk

    # Post-shock pure relaxation regime:
    # the target field must be BELOW the shocked state, otherwise this is not a decay test.
    for _ in range(10):
        state_mod = upd_mod.update(
            state=state_mod,
            X_risk=0.0,
            X_opp=0.0,
            salience=0.1,
            stakes=1.0
        )
        state_ctl = upd_ctl.update(
            state=state_ctl,
            X_risk=0.0,
            X_opp=0.0,
            salience=0.1,
            stakes=1.0
        )

    assert state_mod.q_neg > 0.0, f"q_neg should still be > 0, got {state_mod.q_neg}"
    assert state_ctl.q_neg > 0.0, f"control q_neg should also be > 0, got {state_ctl.q_neg}"

    assert state_mod.h_risk > state_ctl.h_risk, (
        f"importance-modulated trace should decay more slowly toward zero-risk baseline: "
        f"mod={state_mod.h_risk}, ctl={state_ctl.h_risk}"
    )

    assert state_mod.h_risk < shocked_h_mod, (
        f"modulated trace should still relax somewhat after shock: "
        f"final={state_mod.h_risk}, shock={shocked_h_mod}"
    )
    assert state_ctl.h_risk < shocked_h_ctl, (
        f"control trace should also relax after shock: "
        f"final={state_ctl.h_risk}, shock={shocked_h_ctl}"
    )

    print("✓ PASS: q_neg slows risk relaxation")
    return True

def test_q_neg_input_gain_increases_postshock_carryover():
    """
    Input-gain test.

    Here both branches have the same relaxation coupling w_neg_to_risk,
    but only the modulated branch has w_qneg_input > 0.
    This isolates the effect of amplified effective post-shock risk input.
    """
    cfg_modulated = TemporalStateConfig(
        lambda_risk=0.10,
        lambda_opp=0.10,
        lambda_input_risk=0.10,
        lambda_input_opp=0.10,
        rho_neg=0.98,
        rho_pos=0.95,
        k_neg=1.0,
        k_pos=0.7,
        theta_baseline=0.25,
        theta_shot=5.0,
        w_neg_to_risk=2.0,
        w_pos_to_opp=1.0,
        w_qneg_input=1.0,
    )

    cfg_control = TemporalStateConfig(
        lambda_risk=0.10,
        lambda_opp=0.10,
        lambda_input_risk=0.10,
        lambda_input_opp=0.10,
        rho_neg=0.98,
        rho_pos=0.95,
        k_neg=1.0,
        k_pos=0.7,
        theta_baseline=0.25,
        theta_shot=5.0,
        w_neg_to_risk=2.0,
        w_pos_to_opp=1.0,
        w_qneg_input=0.0,
    )

    upd_mod = TemporalStateUpdater(cfg_modulated)
    upd_ctl = TemporalStateUpdater(cfg_control)

    state_mod = TemporalState.zeros()
    state_ctl = TemporalState.zeros()

    # Shock step
    state_mod = upd_mod.update(
        state=state_mod,
        X_risk=0.6,
        X_opp=0.0,
        salience=0.9,
        stakes=10.0
    )
    state_ctl = upd_ctl.update(
        state=state_ctl,
        X_risk=0.6,
        X_opp=0.0,
        salience=0.9,
        stakes=10.0
    )

    # Post-shock low-risk regime
    for _ in range(10):
        state_mod = upd_mod.update(
            state=state_mod,
            X_risk=0.1,
            X_opp=0.1,
            salience=0.1,
            stakes=1.0
        )
        state_ctl = upd_ctl.update(
            state=state_ctl,
            X_risk=0.1,
            X_opp=0.1,
            salience=0.1,
            stakes=1.0
        )

    assert state_mod.q_neg > 0.0, f"modulated q_neg should be > 0, got {state_mod.q_neg}"
    assert state_ctl.q_neg > 0.0, f"control q_neg should be > 0, got {state_ctl.q_neg}"

    assert np.isclose(state_mod.q_neg, state_ctl.q_neg, atol=1e-9), (
        f"q_neg dynamics should match when only w_qneg_input differs: "
        f"mod={state_mod.q_neg}, ctl={state_ctl.q_neg}"
    )

    assert state_mod.h_risk > state_ctl.h_risk, (
        f"input-gain modulation should increase post-shock carryover: "
        f"mod={state_mod.h_risk}, ctl={state_ctl.h_risk}"
    )

    print("✓ PASS: q_neg input gain increases post-shock carryover")
    return True

def test_q_pos_slows_opp_relaxation():
    """
    Pure relaxation test for positive importance.

    Here q_pos should affect only relaxation speed, not effective input gain.
    Therefore w_qpos_input is forced to 0.0 in both branches.
    """
    cfg_modulated = TemporalStateConfig(
        lambda_risk=0.10,
        lambda_opp=0.10,
        lambda_input_risk=0.10,
        lambda_input_opp=0.10,
        rho_neg=0.98,
        rho_pos=0.95,
        k_neg=1.0,
        k_pos=0.7,
        theta_baseline=0.25,
        theta_shot=5.0,
        w_neg_to_risk=2.0,
        w_pos_to_opp=1.0,
        w_qneg_input=0.0,
        w_qpos_input=0.0,
    )

    cfg_control = TemporalStateConfig(
        lambda_risk=0.10,
        lambda_opp=0.10,
        lambda_input_risk=0.10,
        lambda_input_opp=0.10,
        rho_neg=0.98,
        rho_pos=0.95,
        k_neg=1.0,
        k_pos=0.7,
        theta_baseline=0.25,
        theta_shot=5.0,
        w_neg_to_risk=2.0,
        w_pos_to_opp=0.0,
        w_qneg_input=0.0,
        w_qpos_input=0.0,
    )

    upd_mod = TemporalStateUpdater(cfg_modulated)
    upd_ctl = TemporalStateUpdater(cfg_control)

    state_mod = TemporalState.zeros()
    state_ctl = TemporalState.zeros()

    # Treat step
    state_mod = upd_mod.update(
        state=state_mod,
        X_risk=0.0,
        X_opp=1.0,
        salience=0.9,
        stakes=10.0
    )
    state_ctl = upd_ctl.update(
        state=state_ctl,
        X_risk=0.0,
        X_opp=1.0,
        salience=0.9,
        stakes=10.0
    )

    shocked_h_mod = state_mod.h_opp
    shocked_h_ctl = state_ctl.h_opp

    for _ in range(10):
        state_mod = upd_mod.update(
            state=state_mod,
            X_risk=0.0,
            X_opp=0.0,
            salience=0.1,
            stakes=1.0
        )
        state_ctl = upd_ctl.update(
            state=state_ctl,
            X_risk=0.0,
            X_opp=0.0,
            salience=0.1,
            stakes=1.0
        )

    assert state_mod.q_pos > 0.0, f"q_pos should still be > 0, got {state_mod.q_pos}"
    assert state_ctl.q_pos > 0.0, f"control q_pos should also be > 0, got {state_ctl.q_pos}"

    assert state_mod.h_opp > state_ctl.h_opp, (
        f"importance-modulated opportunity trace should decay more slowly: "
        f"mod={state_mod.h_opp}, ctl={state_ctl.h_opp}"
    )

    assert state_mod.h_opp < shocked_h_mod, (
        f"modulated opportunity trace should still relax somewhat after treat: "
        f"final={state_mod.h_opp}, treat={shocked_h_mod}"
    )
    assert state_ctl.h_opp < shocked_h_ctl, (
        f"control opportunity trace should also relax after treat: "
        f"final={state_ctl.h_opp}, treat={shocked_h_ctl}"
    )

    print("✓ PASS: q_pos slows opportunity relaxation")
    return True


def test_q_pos_input_gain_increases_posttreat_carryover():
    """
    Input-gain test for positive importance.

    Here both branches have the same relaxation coupling w_pos_to_opp,
    but only the modulated branch has w_qpos_input > 0.
    """
    cfg_modulated = TemporalStateConfig(
        lambda_risk=0.10,
        lambda_opp=0.10,
        lambda_input_risk=0.10,
        lambda_input_opp=0.10,
        rho_neg=0.98,
        rho_pos=0.95,
        k_neg=1.0,
        k_pos=0.7,
        theta_baseline=0.25,
        theta_shot=5.0,
        w_neg_to_risk=2.0,
        w_pos_to_opp=1.0,
        w_qneg_input=0.0,
        w_qpos_input=1.0,
    )

    cfg_control = TemporalStateConfig(
        lambda_risk=0.10,
        lambda_opp=0.10,
        lambda_input_risk=0.10,
        lambda_input_opp=0.10,
        rho_neg=0.98,
        rho_pos=0.95,
        k_neg=1.0,
        k_pos=0.7,
        theta_baseline=0.25,
        theta_shot=5.0,
        w_neg_to_risk=2.0,
        w_pos_to_opp=1.0,
        w_qneg_input=0.0,
        w_qpos_input=0.0,
    )

    upd_mod = TemporalStateUpdater(cfg_modulated)
    upd_ctl = TemporalStateUpdater(cfg_control)

    state_mod = TemporalState.zeros()
    state_ctl = TemporalState.zeros()

    # Treat step
    state_mod = upd_mod.update(
        state=state_mod,
        X_risk=0.0,
        X_opp=1.0,
        salience=0.9,
        stakes=10.0
    )
    state_ctl = upd_ctl.update(
        state=state_ctl,
        X_risk=0.0,
        X_opp=1.0,
        salience=0.9,
        stakes=10.0
    )

    # Post-treat low-opportunity regime
    for _ in range(10):
        state_mod = upd_mod.update(
            state=state_mod,
            X_risk=0.1,
            X_opp=0.1,
            salience=0.1,
            stakes=1.0
        )
        state_ctl = upd_ctl.update(
            state=state_ctl,
            X_risk=0.1,
            X_opp=0.1,
            salience=0.1,
            stakes=1.0
        )

    assert state_mod.q_pos > 0.0, f"modulated q_pos should be > 0, got {state_mod.q_pos}"
    assert state_ctl.q_pos > 0.0, f"control q_pos should be > 0, got {state_ctl.q_pos}"

    assert np.isclose(state_mod.q_pos, state_ctl.q_pos, atol=1e-9), (
        f"q_pos dynamics should match when only w_qpos_input differs: "
        f"mod={state_mod.q_pos}, ctl={state_ctl.q_pos}"
    )

    assert state_mod.h_opp > state_ctl.h_opp, (
        f"positive input-gain modulation should increase post-treat carryover: "
        f"mod={state_mod.h_opp}, ctl={state_ctl.h_opp}"
    )

    print("✓ PASS: q_pos input gain increases post-treat carryover")
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

def test_event_override_blocks_false_positive_q_pos():
    updater = TemporalStateUpdater()
    state = TemporalState.zeros()

    state = updater.update(
        state=state,
        X_risk=0.1,      # текущий фон
        X_opp=1.0,       # потенциально опасный ложный positive source
        salience=0.9,
        stakes=10.0,
        event_X_risk=0.6,
        event_X_opp=0.0
    )

    assert state.q_neg > 0.0, f"q_neg should rise, got {state.q_neg}"
    assert state.q_pos == 0.0, f"q_pos should stay 0, got {state.q_pos}"
    assert state.one_shot_type == "negative", f"expected negative shot, got {state.one_shot_type}"
    print("✓ PASS: event override blocks false positive q_pos")
    return True

def test_event_override_blocks_false_negative_q_neg():
    updater = TemporalStateUpdater()
    state = TemporalState.zeros()

    state = updater.update(
        state=state,
        X_risk=1.0,      # текущий фон, который не должен дать false q_neg
        X_opp=0.1,
        salience=0.9,
        stakes=10.0,
        event_X_risk=0.0,
        event_X_opp=1.0
    )

    assert state.q_pos > 0.0, f"q_pos should rise, got {state.q_pos}"
    assert state.q_neg == 0.0, f"q_neg should stay 0, got {state.q_neg}"
    assert state.one_shot_type == "positive", f"expected positive shot, got {state.one_shot_type}"
    print("✓ PASS: event override blocks false negative q_neg")
    return True

if __name__ == "__main__":
    print("=" * 70)
    print("Stage 3.1B Rebuild: Temporal State — Smoke Tests")
    print("=" * 70)

    test_temporal_state_update()
    test_q_neg_rises_on_high_negative_event()
    test_q_pos_rises_on_high_positive_event()
    test_q_neg_slows_risk_relaxation()
    test_q_neg_input_gain_increases_postshock_carryover()
    test_q_pos_slows_opp_relaxation()
    test_q_pos_input_gain_increases_posttreat_carryover()
    test_q_traces_stay_quiet_without_extreme_events()
    test_backward_compatibility()
    test_event_override_blocks_false_positive_q_pos()
    test_event_override_blocks_false_negative_q_neg()

    print("=" * 70)
    print("All tests completed!")
    print("=" * 70)