"""
Level 2: Sanity Check — валидация MB/MF сигнатур.
ИСПРАВЛЕННАЯ ВЕРСИЯ: trans_factor = 1/-1 (не 1/0), N_TRIALS=2000.
"""

import sys
sys.path.insert(0, 'E:/CRS-1/substrate_cognitive')

import numpy as np
import pandas as pd
import statsmodels.api as sm
from stage2.twostep.env_twostep import TwoStepEnv
from stage2.core.baselines import MFAgent, MBAgent
from stage2.twostep.config_twostep import DEBUG_SANITY_SEED

N_TRIALS_TEST = 2000

def run_agent(env: TwoStepEnv, agent, n_trials: int = N_TRIALS_TEST) -> pd.DataFrame:
    data = []
    prev_a1 = None
    
    for trial in range(n_trials):
        s1 = env.reset()
        a1 = agent.select_action_stage1(s1)
        s2, trans_type = env.step_stage1(a1)
        a2 = agent.select_action_stage2(s2)
        reward, done, info = env.step_stage2(a2)
        agent.update(a1, a2, reward, s2, trans_type, s1)
        
        stay = 1 if (prev_a1 is not None and a1 == prev_a1) else 0
        # ИСПРАВЛЕНИЕ: 1 = common, -1 = rare (не 1/0!)
        trans_factor = 1 if trans_type == 'common' else -1
        
        data.append({
            'trial': trial,
            'reward': reward,
            'trans_factor': trans_factor,
            'stay': stay
        })
        
        prev_a1 = a1
    
    df = pd.DataFrame(data)
    df['interaction'] = df['reward'] * df['trans_factor']
    return df

def logistic_regression(df: pd.DataFrame):
    """
    Stay ~ reward + trans_factor + interaction
    """
    X = sm.add_constant(df[['reward', 'trans_factor', 'interaction']])
    model = sm.Logit(df['stay'], X)
    try:
        result = model.fit(disp=0, maxiter=100)
    except:
        return None
    return result

def main():
    print("=" * 70)
    print("Level 2: Sanity Check — MB/MF Signatures")
    print("=" * 70)
    print(f"Trials: {N_TRIALS_TEST}, Seed: {DEBUG_SANITY_SEED}")
    print(f"Task: 2-armed bandit at stage 2 (canonical Daw et al., 2011)")
    print(f"Regression: stay ~ reward + trans_factor(1/-1) + interaction")
    print()
    
    env = TwoStepEnv(seed=DEBUG_SANITY_SEED)
    
    # MF-only
    print("Запуск MF-only агента...")
    mf_agent = MFAgent(alpha=0.35, beta=4.0, seed=DEBUG_SANITY_SEED)
    mf_data = run_agent(env, mf_agent)
    mf_result = logistic_regression(mf_data)
    
    if mf_result is not None:
        mf_p_reward = mf_result.pvalues['reward']
        mf_p_interaction = mf_result.pvalues['interaction']
        mf_coef_reward = mf_result.params['reward']
        mf_coef_interaction = mf_result.params['interaction']
        
        print(f"\nMF Agent:")
        print(f"  reward coef     = {mf_coef_reward:.4f} (p = {mf_p_reward:.4f})")
        print(f"  interaction coef= {mf_coef_interaction:.4f} (p = {mf_p_interaction:.4f})")
    else:
        print("\nMF Agent: regression failed")
        mf_p_interaction = 1.0
        mf_coef_reward = 0.0
    
    # MB-only
    print("\nЗапуск MB-only агента...")
    env = TwoStepEnv(seed=DEBUG_SANITY_SEED)
    mb_agent = MBAgent(beta=4.0, seed=DEBUG_SANITY_SEED)
    mb_data = run_agent(env, mb_agent)
    mb_result = logistic_regression(mb_data)
    
    if mb_result is not None:
        mb_p_reward = mb_result.pvalues['reward']
        mb_p_interaction = mb_result.pvalues['interaction']
        mb_coef_reward = mb_result.params['reward']
        mb_coef_interaction = mb_result.params['interaction']
        
        print(f"\nMB Agent:")
        print(f"  reward coef     = {mb_coef_reward:.4f} (p = {mb_p_reward:.4f})")
        print(f"  interaction coef= {mb_coef_interaction:.4f} (p = {mb_p_interaction:.4f})")
    else:
        print("\nMB Agent: regression failed")
        mb_p_interaction = 1.0
        mb_coef_interaction = 0.0
    
    # Валидация
    print()
    print("=" * 70)
    print("Валидация критериев")
    print("=" * 70)
    
    passed = True
    
    if mf_coef_reward > 0.1:
        print(f"✓ MF reward coef > 0.1 ({mf_coef_reward:.4f})")
    else:
        print(f"✗ MF reward coef должен быть > 0.1 ({mf_coef_reward:.4f})")
        passed = False
    
    if mf_p_interaction > 0.10:
        print(f"✓ MF interaction p > 0.10 ({mf_p_interaction:.4f})")
    else:
        print(f"✗ MF interaction p должен быть > 0.10 ({mf_p_interaction:.4f})")
        passed = False
    
    if mb_p_interaction < 0.01:
        print(f"✓ MB interaction p < 0.01 ({mb_p_interaction:.4f})")
    else:
        print(f"✗ MB interaction p должен быть < 0.01 ({mb_p_interaction:.4f})")
        passed = False
    
    if mb_coef_interaction > 0.2:
        print(f"✓ MB interaction coef > 0.2 ({mb_coef_interaction:.4f})")
    else:
        print(f"✗ MB interaction coef должен быть > 0.2 ({mb_coef_interaction:.4f})")
        passed = False
    
    print()
    if passed:
        print("=" * 70)
        print("✓ Level 2: Sanity Check PASSED (MB/MF сигнатуры различимы)")
        print("=" * 70)
        return True
    else:
        print("=" * 70)
        print("✗ Level 2: Sanity Check FAILED")
        print("=" * 70)
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)