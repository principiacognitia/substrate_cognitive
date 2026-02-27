"""
Визуализация результатов MVP.
"""

import matplotlib.pyplot as plt
import numpy as np
from config import CHANGEPOINT_TRIAL, ROLLING_WINDOW

def plot_results(metrics, results):
    """
    Создаёт 4-панельный дашборд результатов.
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('T-maze MVP: Rheological Cognitive Architecture', fontsize=14, fontweight='bold')
    
    trials = np.array(metrics.trials)
    
    # ===== Панель 1: Accuracy =====
    accuracy = metrics.compute_accuracy(window=ROLLING_WINDOW)
    axes[0].plot(trials, accuracy, 'b-', linewidth=2, label='Accuracy (rolling)')
    axes[0].axvline(x=CHANGEPOINT_TRIAL, color='r', linestyle='--', linewidth=2, label='Changepoint')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'[1] Learning Accuracy | Verdict: {results["learning"]["passed"]}')
    
    # ===== Панель 2: Gate Mode =====
    explore = np.array([1 if m == 'EXPLORE' else 0 for m in metrics.modes])
    explore_smooth = metrics.compute_explore_rate(window=5)
    axes[1].fill_between(trials, explore_smooth, alpha=0.5, color='orange', label='EXPLORE rate')
    axes[1].plot(trials, explore_smooth, 'orange', linewidth=2)
    axes[1].axvline(x=CHANGEPOINT_TRIAL, color='r', linestyle='--', linewidth=2)
    axes[1].set_ylabel('EXPLORE Rate')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title(f'[2] Gate Mode (EXPLOIT→EXPLORE) | Verdict: {results["gate_switch"]["passed"]}')
    
    # ===== Панель 3: V_G Trajectory =====
    V_G_arr = np.array(metrics.V_G_history)
    axes[2].plot(trials, V_G_arr, 'g-', linewidth=2, label='V_G (gate viscosity)')
    axes[2].axvline(x=CHANGEPOINT_TRIAL, color='r', linestyle='--', linewidth=2)
    axes[2].set_ylabel('V_G')
    axes[2].set_ylim(0, 1)
    axes[2].legend(loc='lower right')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title(f'[3] Gate Rheology | Verdict: {results["gate_learning"]["passed"]}')
    
    # ===== Панель 4: V_p Trajectory =====
    V_p_arr = np.array(metrics.V_p_history)
    axes[3].plot(trials, V_p_arr, 'purple', linewidth=2, label='V_p (pattern viscosity)')
    axes[3].axvline(x=CHANGEPOINT_TRIAL, color='r', linestyle='--', linewidth=2)
    axes[3].set_ylabel('V_p')
    axes[3].set_xlabel('Trial')
    axes[3].set_ylim(0, 1)
    axes[3].legend(loc='lower right')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_title(f'[4] Pattern Rheology | Verdict: {results["hysteresis"]["passed"]}')
    
    plt.tight_layout()
    plt.savefig('mvp_results.png', dpi=150, bbox_inches='tight')
    print("\n[INFO] Results saved to 'mvp_results.png'")
    plt.show()