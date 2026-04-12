import json
import sys
import pandas as pd

path = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
trial_min = int(sys.argv[3]) if len(sys.argv) > 3 else 30
trial_max = int(sys.argv[4]) if len(sys.argv) > 4 else 33

rows = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

df = pd.DataFrame(rows)

def pick(d, key):
    return d.get(key) if isinstance(d, dict) else None

df["q_safe"] = df["action_policy_debug"].apply(lambda d: pick(d, "q_safe"))
df["beta_used"] = df["action_policy_debug"].apply(lambda d: pick(d, "beta_used"))

mask = (
    (df["seed"] == seed) &
    (df["trial"].between(trial_min, trial_max)) &
    (df["real_junction_choice_row"] == True) &
    (df["mode"] == "exploit_safe")
)

cols = [
    "seed", "trial", "tick", "mode", "action",
    "gate_X_risk", "h_risk", "safe_drive",
    "policy_q_values", "policy_risk_values", "q_safe",
    "beta_used", "action_probs"
]

print(df.loc[mask, cols].to_string(index=False))