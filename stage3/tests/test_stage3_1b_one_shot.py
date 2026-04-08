"""
Stage 3.1B: One-Shot Integration Tests.

Проверяет:
1. что AgentStage3 реально использует one_shot_stakes / one_shot_salience
   из observation при обновлении TemporalState;
2. что OpenCoveredChoiceEnv применяет one_shot_reward только на target trial/path;
3. что env прокидывает one-shot tags обратно в observation после shock event.

Design Principle:
One-shot learning is not a separate memory module.
It is an amplitude-dependent update regime of the same TemporalState.

Author: Alex Snow (Aleksey L. Snigirov)
License: MIT
"""

import copy
import pytest

from stage3.core.agent_stage3 import AgentStage3
from stage3.envs.open_covered_choice_env import OpenCoveredChoiceEnv
from stage3.configs.config_stage3_1b import (
    AGENT_CONFIG_3_1B,
    build_env_config_for_condition,
    get_one_shot_protocol,
)


# =============================================================================
# HELPERS
# =============================================================================

def _make_agent_config_for_one_shot_test():
    """
    Создаёт конфиг агента, где one-shot гарантированно пересекает threshold.
    """
    config = copy.deepcopy(AGENT_CONFIG_3_1B)
    config["log_level"] = 0

    temporal_cfg = config.setdefault("temporal_state_config", {})
    temporal_cfg["one_shot_threshold"] = 5.0
    temporal_cfg["one_shot_boost"] = 2.0

    return config


def _make_one_shot_observation():
    """
    Минимально достаточное observation для AgentStage3.step().
    """
    return {
        # === Instant diagnostics ===
        "prediction_error": 0.1,
        "policy_entropy": 0.69,
        "volatility": 0.05,

        # === Exposure aggregates ===
        "X_risk": 0.6,
        "X_opp": 0.5,
        "D_est": 0.9,

        # === Spatial / deliberation state ===
        "node_id_encoded": 0.5,
        "distance_to_goal": 3.0,
        "at_junction": 1.0,
        "deliberation_state": "deliberating",
        "committed_path_encoded": 0.0,
        "tick": 1,
        "trial": 30,
        "heading_state": 0.0,
        "heading_change": 0.0,
        "state": "deliberating",

        # === Action selection ===
        "q_values": [0.5, 0.5],
        "risk_values": [0.6, 0.2],
        "expected_reward": 0.8,

        # === One-shot tags ===
        "one_shot_active": 1.0,
        "one_shot_trial": 30.0,
        "one_shot_fired": 1.0,
        "one_shot_salience": 0.9,
        "one_shot_stakes": 10.0,
        "one_shot_reward": -5.0,
    }


def _make_one_shot_env():
    """
    Создаёт env для deterministic one-shot tests.

    Важно:
    - reward_prob(open)=0
    - threat_penalty(open)=0
    Тогда на shock trial reward должен быть ровно one_shot_reward.
    """
    env_config = copy.deepcopy(build_env_config_for_condition("R1_T2"))

    protocol = get_one_shot_protocol()
    if "one_shot" in protocol:
        env_config["one_shot"] = copy.deepcopy(protocol["one_shot"])
    else:
        env_config["one_shot"] = copy.deepcopy(protocol)

    # Make open-path reward deterministic and isolate one-shot reward only
    env_config["paths"]["open"]["reward_prob"] = 0.0
    env_config["paths"]["open"]["base_reward"] = 1.0
    env_config["paths"]["open"]["reward_bonus"] = 0.0
    env_config["paths"]["open"]["threat_penalty"] = 0.0
    env_config["paths"]["open"]["threat_prob"] = 0.0

    # Covered path irrelevant for these tests, but keep deterministic too
    env_config["paths"]["covered"]["reward_prob"] = 0.0
    env_config["paths"]["covered"]["base_reward"] = 1.0
    env_config["paths"]["covered"]["reward_bonus"] = 0.0
    env_config["paths"]["covered"]["threat_penalty"] = 0.0
    env_config["paths"]["covered"]["threat_prob"] = 0.0

    return OpenCoveredChoiceEnv(env_config, seed=42)


# =============================================================================
# TEST 1: Agent uses one-shot stakes / salience from observation
# =============================================================================

def test_agent_uses_one_shot_stakes_from_observation():
    """
    Test: AgentStage3 должен использовать one_shot_stakes и one_shot_salience
    из observation при обновлении TemporalState.

    Expected:
    - stakes_used == 10.0
    - salience_used >= 0.9
    - one_shot_pending == True
    - one_shot_amplitude > threshold
    """
    agent = AgentStage3(_make_agent_config_for_one_shot_test(), seed=42)
    observation = _make_one_shot_observation()

    action, metadata = agent.step(
        observation=observation,
        reward=-5.0
    )

    assert metadata["stakes_used"] == 10.0, (
        f"Agent should use one_shot_stakes=10.0, got {metadata['stakes_used']}"
    )
    assert metadata["salience_used"] >= 0.9, (
        f"Agent should use salience >= 0.9, got {metadata['salience_used']}"
    )
    assert metadata["one_shot_pending"] is True, "One-shot should be detected in TemporalState"
    assert metadata["one_shot_amplitude"] > 5.0, (
        f"One-shot amplitude should exceed threshold, got {metadata['one_shot_amplitude']}"
    )

    # sanity: TemporalState must actually move
    ts = metadata["temporal_state"]
    assert ts["h_risk"] > 0.0, "h_risk should increase after one-shot"
    assert ts["h_opp"] > 0.0, "h_opp should increase after one-shot"

    print("✓ PASS: Agent uses one-shot stakes / salience from observation")


# =============================================================================
# TEST 2: Env applies one-shot reward only on target trial/path
# =============================================================================

def test_env_applies_one_shot_reward_only_on_target_trial_and_path():
    """
    Test: OpenCoveredChoiceEnv должен применять one_shot_reward
    только на target trial и только на configured path.

    Setup:
    - target path = open
    - target trial = one_shot_trial
    - base Bernoulli reward disabled for determinism

    Expected:
    - On shock trial/open path: reward == one_shot_reward, last_one_shot_fired=True
    - On next trial/open path: reward == 0.0, last_one_shot_fired=False
    """
    env = _make_one_shot_env()
    shock_trial = env.one_shot_trial

    # Shock trial on open path
    env.reset(trial=shock_trial)
    env.state.current_node = env.goal_node
    env.state.committed_path = "open"

    reward_shock = env._compute_bernoulli_reward()

    assert env.state.last_one_shot_fired is True, "One-shot should fire on target trial/path"
    assert reward_shock == pytest.approx(env.one_shot_reward), (
        f"Shock reward should equal configured one_shot_reward={env.one_shot_reward}, got {reward_shock}"
    )

    # Next trial on same path -> no one-shot
    env.reset(trial=shock_trial + 1)
    env.state.current_node = env.goal_node
    env.state.committed_path = "open"

    reward_nonshock = env._compute_bernoulli_reward()

    assert env.state.last_one_shot_fired is False, "One-shot must not fire outside target trial"
    assert reward_nonshock == pytest.approx(0.0), (
        f"Non-shock reward should be 0.0 with Bernoulli disabled, got {reward_nonshock}"
    )

    print("✓ PASS: Env applies one-shot reward only on target trial/path")


# =============================================================================
# TEST 3: Env exposes one-shot tags in observation after shock
# =============================================================================

def test_env_exposes_one_shot_tags_in_observation_after_shock():
    """
    Test: После shock event env должен вернуть one-shot tags в observation,
    чтобы agent мог использовать их на следующем step.

    Expected observation fields:
    - one_shot_fired == 1.0
    - one_shot_salience == configured salience
    - one_shot_stakes == configured stakes
    - one_shot_reward == configured reward
    """
    env = _make_one_shot_env()
    shock_trial = env.one_shot_trial

    env.reset(trial=shock_trial)
    env.state.current_node = env.goal_node
    env.state.committed_path = "open"

    _ = env._compute_bernoulli_reward()
    obs = env._get_observation()

    assert obs["one_shot_fired"] == 1.0, "Observation should expose one_shot_fired=1.0 after shock"
    assert obs["one_shot_salience"] == pytest.approx(env.one_shot_salience), (
        f"Observation salience should be {env.one_shot_salience}, got {obs['one_shot_salience']}"
    )
    assert obs["one_shot_stakes"] == pytest.approx(env.one_shot_stakes), (
        f"Observation stakes should be {env.one_shot_stakes}, got {obs['one_shot_stakes']}"
    )
    assert obs["one_shot_reward"] == pytest.approx(env.one_shot_reward), (
        f"Observation one_shot_reward should be {env.one_shot_reward}, got {obs['one_shot_reward']}"
    )

    print("✓ PASS: Env exposes one-shot tags in observation after shock")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Stage 3.1B: One-Shot Integration Tests")
    print("=" * 70)

    test_agent_uses_one_shot_stakes_from_observation()
    test_env_applies_one_shot_reward_only_on_target_trial_and_path()
    test_env_exposes_one_shot_tags_in_observation_after_shock()

    print("=" * 70)
    print("All tests passed!")
    print("=" * 70)