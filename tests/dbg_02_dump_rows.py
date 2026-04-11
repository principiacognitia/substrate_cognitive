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

mask = (
    (df["seed"] == seed) &
    (df["trial"].between(trial_min, trial_max))
)

for _, row in df.loc[mask].iterrows():
    print("=" * 100)
    print(
        f"seed={row['seed']} trial={row['trial']} tick={row['tick']} "
        f"pre={row.get('pre_node_id')} post={row.get('post_node_id')} "
        f"mode={row.get('mode')} action={row.get('action')} reward={row.get('reward')}"
    )
    print(
        f"pre_at_junction={row.get('pre_at_junction')} "
        f"post_at_junction={row.get('post_at_junction')} "
        f"real_junction_choice_row={row.get('real_junction_choice_row')}"
    )
    print("pre_q_values:", row.get("pre_q_values"))
    print("pre_risk_values:", row.get("pre_risk_values"))
    print("pre_option_reward_values:", row.get("pre_option_reward_values"))
    print("pre_option_risk_values:", row.get("pre_option_risk_values"))
    print("pre_option_visibility_values:", row.get("pre_option_visibility_values"))
    print("pre_option_expected_threat_values:", row.get("pre_option_expected_threat_values"))
    print("policy_q_values:", row.get("policy_q_values"))
    print("policy_risk_values:", row.get("policy_risk_values"))
    print("action_probs:", row.get("action_probs"))
    print("gate_state_snapshot:", row.get("gate_state_snapshot"))
    print("action_policy_debug:", row.get("action_policy_debug"))