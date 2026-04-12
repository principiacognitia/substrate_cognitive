import json
import sys
import pandas as pd

path = sys.argv[1]
trial_min = int(sys.argv[2]) if len(sys.argv) > 2 else 30
trial_max = int(sys.argv[3]) if len(sys.argv) > 3 else 40

rows = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

df = pd.DataFrame(rows)

def pick(d, key):
    return d.get(key) if isinstance(d, dict) else None

df["exploit_safe_triggered"] = df["gate_state_snapshot"].apply(lambda d: pick(d, "exploit_safe_triggered"))
df["explore_triggered"] = df["gate_state_snapshot"].apply(lambda d: pick(d, "explore_triggered"))
df["critical_risk_threshold"] = df["gate_state_snapshot"].apply(lambda d: pick(d, "critical_risk_threshold"))

mask = (
    (df["real_junction_choice_row"] == True) &
    (df["trial"].between(trial_min, trial_max))
)

cols = [
    "seed", "trial", "tick", "mode", "gate_trigger",
    "gate_X_risk", "h_risk", "safe_drive", "critical_risk_threshold",
    "exploit_safe_triggered", "explore_triggered",
    "action", "action_probs", "candidate_path", "committed_path"
]

print(df.loc[mask, cols].to_string(index=False))