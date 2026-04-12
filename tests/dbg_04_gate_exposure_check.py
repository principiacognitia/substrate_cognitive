import json
import sys
import pandas as pd

path = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
trial_min = int(sys.argv[3]) if len(sys.argv) > 3 else 30
trial_max = int(sys.argv[4]) if len(sys.argv) > 4 else 31

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

mask = (
    (df["seed"] == seed) &
    (df["trial"].between(trial_min, trial_max)) &
    (df["real_junction_choice_row"] == True)
)

cols = [
    "seed", "trial", "tick",
    "pre_node_id", "mode", "gate_trigger", "action",
    "node_X_risk", "gate_X_risk", "gate_exposure_source",
    "h_risk", "safe_drive", "critical_risk_threshold",
    "exploit_safe_triggered", "explore_triggered",
    "pre_option_risk_values", "policy_risk_values", "action_probs"
]

print(df.loc[mask, cols].to_string(index=False))