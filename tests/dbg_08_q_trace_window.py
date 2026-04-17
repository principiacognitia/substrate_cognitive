import json
import sys
import pandas as pd

path = sys.argv[1]
trial_min = int(sys.argv[2]) if len(sys.argv) > 2 else 28
trial_max = int(sys.argv[3]) if len(sys.argv) > 3 else 45

rows = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

df = pd.DataFrame(rows)

required = ["seed", "trial", "tick", "real_junction_choice_row"]
for col in required:
    if col not in df.columns:
        df[col] = pd.NA

df = df.dropna(subset=["seed", "trial", "tick"]).copy()

df["seed"] = df["seed"].astype(int)
df["trial"] = df["trial"].astype(int)
df["tick"] = df["tick"].astype(int)

mask = (
    df["trial"].between(trial_min, trial_max) &
    (df["real_junction_choice_row"] == True)
)

cols = [
    "seed", "trial", "tick", "mode", "gate_trigger",
    "node_X_risk", "gate_X_risk",
    "one_shot_source_X_risk", "one_shot_source_X_opp", "one_shot_source_reward",
    "h_risk", "h_opp",
    "q_neg", "q_pos",
    "one_shot_type", "one_shot_amplitude",
    "safe_drive", "action_probs",
    "candidate_path", "committed_path"
]

out = df.loc[mask, cols].sort_values(["seed", "trial", "tick"]).copy()

with pd.option_context(
    "display.max_columns", None,
    "display.width", 2000,
    "display.expand_frame_repr", False,
    "display.max_colwidth", 80,
):
    print(out.to_string(index=False))