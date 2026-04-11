import json
import sys
import pandas as pd

path = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
trial_min = int(sys.argv[3]) if len(sys.argv) > 3 else 29
trial_max = int(sys.argv[4]) if len(sys.argv) > 4 else 32

rows = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

df = pd.DataFrame(rows)

def pick(d, key):
    return d.get(key) if isinstance(d, dict) else None

df["critical_risk_threshold"] = df["gate_state_snapshot"].apply(lambda d: pick(d, "critical_risk_threshold"))
df["exploit_safe_triggered"] = df["gate_state_snapshot"].apply(lambda d: pick(d, "exploit_safe_triggered"))
df["explore_triggered"] = df["gate_state_snapshot"].apply(lambda d: pick(d, "explore_triggered"))
df["risk_source"] = df["action_policy_debug"].apply(lambda d: pick(d, "risk_source"))
df["q_source"] = df["action_policy_debug"].apply(lambda d: pick(d, "q_source"))

mask = (
    (df["seed"] == seed) &
    (df["trial"].between(trial_min, trial_max)) &
    (df["real_junction_choice_row"] == True)
)

cols = [
    "seed", "trial", "tick",
    "pre_node_id", "post_node_id",
    "mode", "gate_trigger", "action",
    "pre_X_risk", "h_risk", "safe_drive",
    "critical_risk_threshold", "exploit_safe_triggered", "explore_triggered",
    "pre_q_values", "pre_risk_values",
    "pre_option_reward_values", "pre_option_risk_values",
    "policy_q_values", "policy_risk_values", "action_probs",
    "q_source", "risk_source",
    "candidate_path", "committed_path",
    "pre_one_shot_fired", "post_one_shot_fired",
    "step_one_shot_from_pending", "salience_used", "stakes_used"
]

print(df.loc[mask, cols].to_string(index=False))