import json
import sys
import pandas as pd

path = sys.argv[1]
trial_min = int(sys.argv[2]) if len(sys.argv) > 2 else 28
trial_max = int(sys.argv[3]) if len(sys.argv) > 3 else 45

rows = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

df = pd.DataFrame(rows)

mask = (
    (df["trial"].between(trial_min, trial_max)) &
    (df["real_junction_choice_row"] == True)
)

cols = [
    "seed", "trial", "tick", "mode", "gate_trigger",
    "node_X_risk", "gate_X_risk",
    "h_risk", "h_opp",
    "q_neg", "q_pos",
    "one_shot_type", "one_shot_amplitude",
    "safe_drive", "action_probs",
    "candidate_path", "committed_path"
]

print(df.loc[mask, cols].to_string(index=False))