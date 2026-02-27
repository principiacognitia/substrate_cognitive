"""
Level 1: Unit Test для среды Two-Step.
Проверяет статистику переходов и наград.
"""

import sys
sys.path.insert(0, 'E:/CRS-1/substrate_cognitive')

import numpy as np
import csv
from stage2.twostep.env_twostep import TwoStepEnv
from stage2.twostep.config_twostep import N_TRIALS, COMMON_TRANS_PROB, DEBUG_ENV_SEED

def test_environment():
    """Тестирует статистику среды."""
    env = TwoStepEnv(seed=DEBUG_ENV_SEED)
    
    print("=" * 60)
    print("Level 1: Unit Test (Environment)")
    print("=" * 60)
    
    transitions = {'common': 0, 'rare': 0}
    rewards = []
    
    # CSV logger
    with open('logs/twostep/debug_env_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['trial', 's1', 'a1', 's2', 'trans_type', 'a2', 'reward'])
        
        for trial in range(N_TRIALS):
            s1 = env.reset()
            a1 = np.random.choice([0, 1])
            s2, trans_type = env.step_stage1(a1)
            transitions[trans_type] += 1
            
            a2 = np.random.choice([0, 1])
            reward, done, _ = env.step_stage2(a2)
            rewards.append(reward)
            
            env.log_trial(reward)
            writer.writerow([trial, s1, a1, s2, trans_type, a2, reward])
    
    # Валидация
    common_ratio = transitions['common'] / sum(transitions.values())
    mean_reward = np.mean(rewards)
    
    print(f"\nСтатистика ({N_TRIALS} триалов):")
    print(f"  Common transitions: {transitions['common']} ({common_ratio:.3f})")
    print(f"  Rare transitions: {transitions['rare']} ({1-common_ratio:.3f})")
    print(f"  Ожидалось common: ~{COMMON_TRANS_PROB}")
    print(f"  Mean reward: {mean_reward:.3f}")
    
    # Проверка
    assert 0.65 < common_ratio < 0.75, f"Ошибка в матрице переходов! common_ratio={common_ratio:.3f}"
    assert mean_reward > 0.0, f"Средняя награда должна быть > 0, получил {mean_reward:.3f}"
    
    print()
    print("=" * 60)
    print("✓ Level 1: Unit Test PASSED")
    print("=" * 60)
    print(f"\nЛог сохранён: logs/twostep/debug_env_log.csv")
    return True

if __name__ == "__main__":
    try:
        test_environment()
    except AssertionError as e:
        print(f"\n✗ ASSERTION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        sys.exit(1)