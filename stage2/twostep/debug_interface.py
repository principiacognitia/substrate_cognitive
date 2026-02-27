"""
Level 0: Interface Contract Test.
Проверяет сигнатуры и диапазоны возвращаемых значений TwoStepEnv.
"""

import sys
sys.path.insert(0, 'E:/CRS-1/substrate_cognitive')

import numpy as np
from stage2.twostep.env_twostep import TwoStepEnv
from stage2.twostep.config_twostep import COMMON_TRANS_PROB

def test_interface():
    """Тестирует интерфейс среды."""
    env = TwoStepEnv(seed=42)
    
    print("=" * 60)
    print("Level 0: Interface Contract Test")
    print("=" * 60)
    
    # Test reset()
    s1 = env.reset()
    assert isinstance(s1, (int, np.integer)), f"s1 должен быть int, получил {type(s1)}"
    assert s1 in [0, 1], f"s1 должен быть в [0, 1], получил {s1}"
    print(f"✓ reset() → s1={s1}")
    
    # Test step_stage1()
    s2, trans_type = env.step_stage1(action=0)
    assert isinstance(s2, (int, np.integer)), f"s2 должен быть int, получил {type(s2)}"
    assert s2 in [0, 1, 2, 3], f"s2 должен быть в [0,1,2,3], получил {s2}"
    assert trans_type in ['common', 'rare'], f"trans_type должен быть 'common' или 'rare', получил {trans_type}"
    print(f"✓ step_stage1(0) → s2={s2}, trans_type={trans_type}")
    
    # Test step_stage2()
    reward, done, info = env.step_stage2(action=0)
    assert isinstance(reward, (int, float)), f"reward должен быть числом, получил {type(reward)}"
    assert 0.0 <= reward <= 1.0, f"reward должен быть в [0, 1], получил {reward}"
    assert isinstance(done, bool), f"done должен быть bool, получил {type(done)}"
    assert isinstance(info, dict), f"info должен быть dict, получил {type(info)}"
    print(f"✓ step_stage2(0) → reward={reward}, done={done}")
    
    # Test get_transition_type()
    env.reset()
    env.step_stage1(0)
    tt = env.get_transition_type()
    assert tt in ['common', 'rare'], f"get_transition_type() должен возвращать 'common' или 'rare', получил {tt}"
    print(f"✓ get_transition_type() → {tt}")
    
    # Test get_reward_probs()
    probs = env.get_reward_probs()
    assert isinstance(probs, np.ndarray), f"get_reward_probs() должен возвращать np.ndarray"
    assert len(probs) == 4, f"Должно быть 4 вероятности, получил {len(probs)}"
    assert all(0.0 <= p <= 1.0 for p in probs), f"Все вероятности должны быть в [0, 1]"
    print(f"✓ get_reward_probs() → {probs}")
    
    print()
    print("=" * 60)
    print("✓ Level 0: Interface Contract Test PASSED")
    print("=" * 60)
    return True

if __name__ == "__main__":
    try:
        test_interface()
    except AssertionError as e:
        print(f"\n✗ ASSERTION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        sys.exit(1)