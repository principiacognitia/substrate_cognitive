import json
import pandas as pd

path = r"E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1b\one_shot_full_forced_20260410_214759\one_shot_full_debug_trace.jsonl"

rows = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

df = pd.DataFrame(rows)

cols = [
    "seed", "trial", "tick", "node_id", "deliberation_state",
    "mode", "gate_trigger", "action", "reward",
    "X_risk", "h_risk",
    "safe_drive", "uncertainty_signal", "v_g_approx", "explore_gate_output",
    "one_shot_fired", "one_shot_pending", "forced_action_applied"
]

print(
    df.loc[
        (df["trial"].between(30, 31)) & (df["at_junction"] == True),
        cols
    ].to_string(index=False)
)

for _, row in df.loc[(df["trial"].between(29, 35)) & (df["at_junction"] == True)].iterrows():
    print("=" * 80)
    print(f"trial={row['trial']} tick={row['tick']} mode={row['mode']} action={row['action']}")
    print("gate_state_snapshot:", row["gate_state_snapshot"])
    print("action_policy_debug:", row["action_policy_debug"])