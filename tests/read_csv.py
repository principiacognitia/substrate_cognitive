import pandas as pd

path = r"E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1b\one_shot_full_forced_20260410_214759\one_shot_full_all_steps.csv"

df = pd.read_csv(path)

cols = [
    "seed", "trial", "tick", "node_id", "at_junction", "deliberation_state",
    "mode", "gate_trigger", "action", "reward",
    "u_delta", "u_entropy", "u_volatility",
    "X_risk", "h_risk", "h_time",
    "salience", "stakes",
    "one_shot_pending", "one_shot_active", "one_shot_trial", "forced_action_applied"
]

print(
    df.loc[
        (df["trial"].between(30, 31)) & (df["at_junction"] == True),
        cols
    ].to_string(index=False)
)