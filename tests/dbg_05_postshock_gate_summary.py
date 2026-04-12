import json
import sys
import pandas as pd

path = sys.argv[1]
trial_min = int(sys.argv[2]) if len(sys.argv) > 2 else 31
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

mask = (
    (df["real_junction_choice_row"] == True) &
    (df["trial"].between(trial_min, trial_max))
)

summary = (
    df.loc[mask]
      .groupby(["seed", "trial"], as_index=False)
      .agg(
          gate_X_risk_max=("gate_X_risk", "max"),
          h_risk_max=("h_risk", "max"),
          safe_drive_max=("safe_drive", "max"),
          exploit_safe_any=("exploit_safe_triggered", "max"),
          explore_any=("explore_triggered", "max"),
      )
)

print(summary.to_string(index=False))