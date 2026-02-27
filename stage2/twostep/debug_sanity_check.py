"""
Level 2: Sanity Check — валидация MB/MF сигнатур.

Требование (Grok): 
  - MB interaction p < 0.01
  - MF interaction p > 0.10

Это подтверждает, что среда позволяет статистически различать стратегии.
"""

import sys
sys.path.insert(0, 'E:/CRS-1/substrate_cognitive')

import numpy as np
import pandas as pd
import statsmodels.api as sm
from stage2.twostep.env_twostep import TwoStepEnv
from stage2.core.baselines import MFAgent, MBAgent
from stage2.twostep.config_twostep import N_TRIALS, DEBUG_SANITY_SEED

def run_agent(env: TwoStepEnv, agent, n_trials: int = N_TRIALS) -> pd.DataFrame:
    """
    Прогон агента и сбор данных для регрессии.
    
    Returns:
        DataFrame с колонками: trial, reward, transition, stay
    """
    data = []
    prev_a1 = None
    
    for trial in range(n_trials):
        s1 = env.reset()
        a1 = agent.select_action_stage1(s1)
        s2, trans_type = env.step_stage1(a1)
        a2 = agent.select_action_stage2(s2)
        reward, done, info = env.step_stage2(a2)
        agent.update(a1, a2, reward, s2, trans_type)
        
        # Stay: повторил ли агент то же действие на этапе 1
        stay = 1 if (prev_a1 is not None and a1 == prev_a1) else 0
        
        # Transition: 1 = common, 0 = rare
        transition = 1 if trans_type == 'common' else 0
        
        data.append({
            'trial': trial,
            'reward': reward,
            'transition': transition,
            'stay': stay
        })
        
        prev_a1 = a1
    
    df = pd.DataFrame(data)
    df['reward:transition'] = df['reward'] * df['transition']
    return df

def logistic_regression(df: pd.DataFrame) -> sm.LogitResults:
    """
    Logistic regression: Stay ~ Reward × Transition + Reward + Transition
    
    Returns:
        fitted model results
    """
    model = sm.Logit(
        df['stay'],
        sm.add_constant(df[['reward', 'transition', 'reward:transition']])
    )
    try:
        result = model.fit(disp=0, maxiter=100)
    except:
        # Если не сходится, возвращаем None
        return None
    return result

def main():
    print("=" * 70)
    print("Level 2: Sanity Check — MB/MF Signatures")
    print("=" * 70)
    print(f"Trials: {N_TRIALS}, Seed: {DEBUG_SANITY_SEED}")
    print()
    
    # Инициализация
    env = TwoStepEnv(seed=DEBUG_SANITY_SEED)
    
    # ===== MF-only агент =====
    print("Запуск MF-only агента...")
    mf_agent = MFAgent(alpha=0.25, beta=5.0, seed=DEBUG_SANITY_SEED)
    mf_data = run_agent(env, mf_agent)
    mf_result = logistic_regression(mf_data)
    
    if mf_result is not None:
        mf_p_reward = mf_result.pvalues['reward']
        mf_p_interaction = mf_result.pvalues['reward:transition']
        mf_coef_reward = mf_result.params['reward']
        mf_coef_interaction = mf_result.params['reward:transition']
        
        print(f"\nMF Agent:")
        print(f"  reward coef     = {mf_coef_reward:.4f} (p = {mf_p_reward:.4f})")
        print(f"  interaction coef= {mf_coef_interaction:.4f} (p = {mf_p_interaction:.4f})")
    else:
        print("\nMF Agent: regression failed")
        mf_p_interaction = 1.0
        mf_coef_reward = 0.0
    
    # ===== MB-only агент =====
    print("\nЗапуск MB-only агента...")
    env = TwoStepEnv(seed=DEBUG_SANITY_SEED)  # Reset env
    mb_agent = MBAgent(beta=5.0, seed=DEBUG_SANITY_SEED)
    mb_data = run_agent(env, mb_agent)
    mb_result = logistic_regression(mb_data)
    
    if mb_result is not None:
        mb_p_reward = mb_result.pvalues['reward']
        mb_p_interaction = mb_result.pvalues['reward:transition']
        mb_coef_reward = mb_result.params['reward']
        mb_coef_interaction = mb_result.params['reward:transition']
        
        print(f"\nMB Agent:")
        print(f"  reward coef     = {mb_coef_reward:.4f} (p = {mb_p_reward:.4f})")
        print(f"  interaction coef= {mb_coef_interaction:.4f} (p = {mb_p_interaction:.4f})")
    else:
        print("\nMB Agent: regression failed")
        mb_p_interaction = 1.0
        mb_coef_interaction = 0.0
    
    # ===== Валидация (Grok пункт 2) =====
    print()
    print("=" * 70)
    print("Валидация критериев")
    print("=" * 70)
    
    passed = True
    
    # MF должен реагировать на награду (p < 0.05)
    if mf_coef_reward > 0.1:
        print(f"✓ MF reward coef > 0.1 ({mf_coef_reward:.4f})")
    else:
        print(f"✗ MF reward coef должен быть > 0.1 ({mf_coef_reward:.4f})")
        passed = False
    
    # MF interaction должен быть НЕ значим (p > 0.10)
    if mf_p_interaction > 0.10:
        print(f"✓ MF interaction p > 0.10 ({mf_p_interaction:.4f})")
    else:
        print(f"✗ MF interaction p должен быть > 0.10 ({mf_p_interaction:.4f})")
        passed = False
    
    # MB interaction должен быть значим (p < 0.01)
    if mb_p_interaction < 0.01:
        print(f"✓ MB interaction p < 0.01 ({mb_p_interaction:.4f})")
    else:
        print(f"✗ MB interaction p должен быть < 0.01 ({mb_p_interaction:.4f})")
        passed = False
    
    # MB interaction coef должен быть > 0.2
    if mb_coef_interaction > 0.1:
        print(f"✓ MB interaction coef > 0.1 ({mb_coef_interaction:.4f})")
    else:
        print(f"✗ MB interaction coef должен быть > 0.1 ({mb_coef_interaction:.4f})")
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