"""
Stage 3.0: Agent Orchestration with Stochastic Policy.

Оркестрирует полный цикл агента с mode-specific softmax policies.

Design Principles:
- Policy stochasticity via softmax (не argmax)
- Mode-specific inverse temperatures (beta_exploit, beta_explore, beta_safe)
- Risk-penalized values for EXPLOIT_SAFE
- Full logging of action_probs, q_values, risk_values

Author: Alex Snow (Aleksey L. Snigirov)
License: MIT
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import numpy as np

from stage3.core.gate_modes import GateMode, ALL_MODES
from stage3.core.gate_inputs import (
    GateInput,
    InstantDiagnostics,
    ExposureAggregates,
    TemporalState
)
from stage3.core.exposure_field import ExposureField
from stage3.core.temporal_state import TemporalStateUpdater, TemporalStateConfig
from stage3.core.gate_stage3 import GateStage3, GateThresholds
from stage3.core.compatibility import Stage2CompatShim, Stage2CompatConfig


@dataclass
class AgentLog:
    """
    Лог одного шага агента (расширенный для Stage 3.1).
    
    Соответствует logging contract из ТЗ.
    """
    seed: int = 0
    trial: int = 0
    tick: int = 0
    node_id: str = ""
    at_junction: bool = False
    state_machine_state: str = ""
    mode_before: str = ""
    mode_after: str = ""
    gate_trigger: str = ""
    q_values: List[float] = field(default_factory=list)
    risk_values: List[float] = field(default_factory=list)
    action_probs: List[float] = field(default_factory=list)
    sampled_action: int = 0
    candidate_path: str = ""
    committed_path: str = ""
    reward: float = 0.0
    salience: float = 0.0
    u_delta: float = 0.0
    u_entropy: float = 0.0
    u_volatility: float = 0.0
    X_risk: float = 0.0
    X_opp: float = 0.0
    D_est: float = 0.0
    h_risk: float = 0.0
    h_opp: float = 0.0
    h_time: int = 0
    one_shot_fired: bool = False


@dataclass
class AgentStage3Config:
    """
    Конфигурация агента Stage 3.0.
    
    Разделение параметров по ролям:
    - stage2_legacy: Backward compatibility (обучение + viscosity)
    - action_policy: Stochastic action selection (mode-specific)
    - temporal_state_config: Trace dynamics
    - gate_thresholds: Threshold cascade
    - exposure_field_config: Exposure computation
    """
    compatibility_mode: bool = False
    log_level: int = 1
    
    # === Stage 2 Legacy (backward compatibility) ===
    stage2_legacy: Dict = field(default_factory=lambda: {
        'alpha': 0.35,
        'beta': 4.0,
        'k_use': 0.08,
        'k_melt': 0.20,
        'lambda_decay': 0.01,
        'tau_vol': 0.50,
    })
    
    # === Stage 3 Action Policy (mode-specific stochastic selection) ===
    action_policy: Dict = field(default_factory=lambda: {
        'beta_exploit': 4.0,
        'beta_explore': 1.0,
        'beta_safe': 5.0,
        'lambda_risk': 2.0,
        'epsilon_explore': 0.0,
        'commit_confidence': 0.7,  # Threshold for deliberation commit
        'max_deliberation_ticks': 10,  # Max ticks in DELIBERATING state
    })
    
    # === Stage 3 Core Configs ===
    exposure_field_config: Dict = field(default_factory=dict)
    temporal_state_config: Dict = field(default_factory=dict)
    gate_thresholds: Dict = field(default_factory=dict)
    
    # === Aliases для совместимости с config_stage3_1a.py ===
    exposure_field: Dict = field(default_factory=dict, repr=False, compare=False)
    temporal_state: Dict = field(default_factory=dict, repr=False, compare=False)
    
    def __post_init__(self):
        """
        Валидация и миграция параметров.
        """
        # Миграция: beta_exploit по умолчанию от legacy beta
        if 'beta_exploit' not in self.action_policy:
            self.action_policy['beta_exploit'] = self.stage2_legacy.get('beta', 4.0)
        
        # Валидация stage2_legacy
        required_legacy = ['alpha', 'beta', 'k_use', 'k_melt', 'lambda_decay', 'tau_vol']
        for key in required_legacy:
            if key not in self.stage2_legacy:
                raise ValueError(f"stage2_legacy must contain '{key}'")
        
        # Валидация action_policy
        required_policy = ['beta_exploit', 'beta_explore', 'beta_safe', 'lambda_risk']
        for key in required_policy:
            if key not in self.action_policy:
                raise ValueError(f"action_policy must contain '{key}'")


class AgentStage3:
    """
    Agent Stage 3 с mode-specific stochastic policies.
    
    Architecture:
        Environment → ExposureField → TemporalState → Gate → Stochastic Policy → Action
    
    Usage:
        config = AgentStage3Config()
        agent = AgentStage3(config, seed=42)
        
        for trial in range(n_trials):
            observation, reward, done = env.step(action)
            action, metadata = agent.step(observation, reward)
    """
    
    def __init__(
        self,
        config: Optional[Union[AgentStage3Config, Dict[str, Any]]] = None,
        seed: Optional[int] = None
    ):
        """
        Инициализирует агента Stage 3.0.
        
        Args:
            config: Конфигурация агента (dataclass или dict)
            seed: Random seed для воспроизводимости
        """
        # Конвертируем dict в dataclass если нужно
        if config is None:
            self.config = AgentStage3Config()
        elif isinstance(config, dict):
            config_copy = config.copy()
            
            # Map aliases
            if 'temporal_state' in config_copy and 'temporal_state_config' not in config_copy:
                config_copy['temporal_state_config'] = config_copy.pop('temporal_state')
            
            if 'exposure_field' in config_copy and 'exposure_field_config' not in config_copy:
                config_copy['exposure_field_config'] = config_copy.pop('exposure_field')
            
            # Remove aliases
            config_copy.pop('exposure_field', None)
            config_copy.pop('temporal_state', None)
            
            self.config = AgentStage3Config(**config_copy)
        else:
            self.config = config
        
        # === Инициализация RNG (Task 1) ===
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.seed = seed
        else:
            self.rng = np.random.default_rng()
            self.seed = 0
        
        # Инициализация компонентов
        self.exposure_field = ExposureField(**self.config.exposure_field_config)
        
        temporal_config = TemporalStateConfig(**self.config.temporal_state_config)
        self.temporal_updater = TemporalStateUpdater(temporal_config)
        
        gate_thresholds = GateThresholds(**self.config.gate_thresholds)
        self.gate = GateStage3(gate_thresholds)
        
        # Backward compatibility shim
        compat_config = Stage2CompatConfig(enabled=self.config.compatibility_mode)
        self.compat_shim = Stage2CompatShim(compat_config)
        
        # Внутреннее состояние
        self.current_temporal_state = TemporalState.zeros()
        self.trial_count = 0
        self.log_buffer: List[AgentLog] = []
        
        # Stage 2 совместимость
        self.u_delta_history: List[float] = []
        self.u_entropy_history: List[float] = []
        self.u_volatility_history: List[float] = []
    
    def step(
        self,
        observation: Dict,
        reward: Optional[float] = None,
        action: Optional[int] = None,
        salience: Optional[float] = None
    ) -> Tuple[int, Dict]:
        """
        Один шаг агента с mode-specific stochastic policy.
        
        Args:
            observation: Raw observation из среды
            reward: Reward signal
            action: Previous action
            salience: Salience estimate
        
        Returns:
            (action, metadata_dict)
            metadata_dict содержит:
                - mode: selected Gate mode
                - exposure: ExposureAggregates
                - temporal_state: TemporalState
                - action_probs: probabilities for all actions
                - q_values: Q-values for all actions
                - gate_constraint: which threshold triggered
        """
        # =====================================================================
        # 1. Вычисление Instant Diagnostics (u_t)
        # =====================================================================
        instant_diagnostics = self._compute_instant_diagnostics(
            observation, reward, action
        )
        
        # =====================================================================
        # 2. Вычисление Exposure Aggregates
        # =====================================================================
        # Если observation уже содержит exposure aggregates (из spatial env),
        # используем их напрямую
        if 'X_risk' in observation and 'X_opp' in observation and 'D_est' in observation:
            exposure_aggregates = ExposureAggregates(
                X_risk=observation['X_risk'],
                X_opp=observation['X_opp'],
                D_est=observation['D_est']
            )
        else:
            exposure_aggregates = self.exposure_field.compute_exposure(
                observation=observation,
                action=action,
                reward=reward,
                trial=self.trial_count
            )
        
        # =====================================================================
        # 3. Обновление Temporal State
        # =====================================================================
        if salience is None:
            salience = instant_diagnostics.u_delta
        
        self.current_temporal_state = self.temporal_updater.update(
            state=self.current_temporal_state,
            X_risk=exposure_aggregates.X_risk,
            X_opp=exposure_aggregates.X_opp,
            salience=salience,
            stakes=1.0
        )
        
        # =====================================================================
        # 4. Backward Compatibility
        # =====================================================================
        if self.config.compatibility_mode:
            gate_input = self.compat_shim.create_compat_input(instant_diagnostics)
        else:
            gate_input = GateInput(
                instant=instant_diagnostics,
                exposure=exposure_aggregates,
                temporal=self.current_temporal_state
            )
        
        # =====================================================================
        # 5. Gate Mode Selection (threshold cascade)
        # =====================================================================
        selected_mode, gate_metadata = self.gate.select_mode(gate_input)
        
        # =====================================================================
        # 6. Action Selection с mode-specific stochastic policy (Task 2)
        # =====================================================================
        action, action_metadata = self._select_action_for_mode_stochastic(
            mode=selected_mode,
            observation=observation,
            instant_diagnostics=instant_diagnostics,
            exposure=exposure_aggregates
        )
        
        # =====================================================================
        # 7. Логирование (расширенное для Stage 3.1)
        # =====================================================================
        if self.config.log_level > 0:
            log_entry = self._create_log_entry(
                observation=observation,
                exposure=exposure_aggregates,
                temporal_state=self.current_temporal_state,
                instant_diagnostics=instant_diagnostics,
                mode_scores=gate_metadata['mode_scores'],
                selected_mode=selected_mode,
                action=action,
                reward=reward or 0.0,
                gate_constraint=gate_metadata['gate_constraint'],
                action_metadata=action_metadata
            )
            self.log_buffer.append(log_entry)
        
        self.trial_count += 1
        
        # Metadata для внешнего использования
        metadata = {
            'trial': self.trial_count - 1,
            'mode': str(selected_mode),
            'action': action,
            'exposure': {
                'X_risk': exposure_aggregates.X_risk,
                'X_opp': exposure_aggregates.X_opp,
                'D_est': exposure_aggregates.D_est
            },
            'temporal_state': {
                'h_risk': self.current_temporal_state.h_risk,
                'h_opp': self.current_temporal_state.h_opp,
                'h_time': self.current_temporal_state.h_time
            },
            'instant_diagnostics': {
                'u_delta': instant_diagnostics.u_delta,
                'u_entropy': instant_diagnostics.u_entropy,
                'u_volatility': instant_diagnostics.u_volatility
            },
            'mode_scores': {str(k): v for k, v in gate_metadata['mode_scores'].items()},
            'gate_constraint': gate_metadata['winning_constraint'],
            'one_shot_pending': self.current_temporal_state.one_shot_pending,
            # === Новые поля для Stage 3.1 ===
            'action_probs': action_metadata.get('action_probs', []),
            'q_values': action_metadata.get('q_values', []),
            'risk_values': action_metadata.get('risk_values', []),
        }
        
        return action, metadata
    
    def _select_action_for_mode_stochastic(
        self,
        mode: GateMode,
        observation: Dict,
        instant_diagnostics: InstantDiagnostics,
        exposure: ExposureAggregates
    ) -> Tuple[int, Dict]:
        """
        Task 2: Mode-specific stochastic policy через softmax.
        
        Design Principle: Principled stochasticity, не hand-coded random.
        
        Returns:
            (action, metadata)
            metadata содержит action_probs, q_values, risk_values для логирования
        """
        q_values = observation.get('q_values', [0.5, 0.5])
        q_values = np.array(q_values, dtype=np.float64)
        n_actions = len(q_values)
        
        # === Получаем параметры из action_policy ===
        beta_exploit = self.config.action_policy.get('beta_exploit', 
                              self.config.stage2_legacy.get('beta', 4.0))
        beta_explore = self.config.action_policy.get('beta_explore', 1.0)
        beta_safe = self.config.action_policy.get('beta_safe', 5.0)
        lambda_risk = self.config.action_policy.get('lambda_risk', 2.0)
        epsilon_explore = self.config.action_policy.get('epsilon_explore', 0.0)
        
        # === Риск для каждого действия (из exposure) ===
        # Для простоты: open path (action=0) имеет риск X_risk, covered (action=1) имеет 0
        risk_values = np.array([exposure.X_risk, 0.0])[:n_actions]
        
        # === EXPLOIT: Softmax с высоким beta ===
        if mode == GateMode.EXPLOIT:
            logits = beta_exploit * q_values
            probs = self._softmax(logits)
            action = self.rng.choice(n_actions, p=probs)
        
        # === EXPLORE: Softmax с низким beta (более случайно) ===
        elif mode == GateMode.EXPLORE:
            if epsilon_explore > 0:
                # Epsilon-soft policy
                if self.rng.random() < epsilon_explore:
                    action = self.rng.randint(0, n_actions)
                    probs = np.ones(n_actions) / n_actions
                else:
                    logits = beta_explore * q_values
                    probs = self._softmax(logits)
                    action = self.rng.choice(n_actions, p=probs)
            else:
                # Pure softmax с низким beta
                logits = beta_explore * q_values
                probs = self._softmax(logits)
                action = self.rng.choice(n_actions, p=probs)
        
        # === EXPLOIT_SAFE: Softmax над risk-penalized values ===
        elif mode == GateMode.EXPLOIT_SAFE:
            # Penalized Q-values: Q_safe = Q - lambda_risk * risk
            q_safe = q_values - lambda_risk * risk_values
            
            logits = beta_safe * q_safe
            probs = self._softmax(logits)
            action = self.rng.choice(n_actions, p=probs)
        
        # === ABSENCE_CHECK: Как EXPLORE (пока нет full scan policy) ===
        elif mode == GateMode.ABSENCE_CHECK:
            logits = beta_explore * q_values
            probs = self._softmax(logits)
            action = self.rng.choice(n_actions, p=probs)
        
        else:
            # Fallback: равномерное распределение
            probs = np.ones(n_actions) / n_actions
            action = self.rng.choice(n_actions, p=probs)
        
        # Metadata для логирования
        metadata = {
            'action_probs': probs.tolist(),
            'q_values': q_values.tolist(),
            'risk_values': risk_values.tolist(),
            'sampled_action': int(action)
        }
        
        return int(action), metadata
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """
        Numerically stable softmax.
        
        Args:
            logits: Raw logits (beta * Q)
        
        Returns:
            Probability distribution
        """
        # Subtract max for numerical stability
        logits_shifted = logits - np.max(logits)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits)
        
        # Clip to avoid numerical issues
        probs = np.clip(probs, 1e-10, 1.0 - 1e-10)
        probs = probs / np.sum(probs)  # Renormalize
        
        return probs
    
    def _compute_instant_diagnostics(
        self,
        observation: Dict,
        reward: Optional[float],
        action: Optional[int]
    ) -> InstantDiagnostics:
        """Вычисляет InstantDiagnostics из observation/reward."""
        if reward is not None:
            expected_reward = observation.get('expected_reward', 0.0)
            u_delta = abs(reward - expected_reward)
        else:
            u_delta = observation.get('prediction_error', 0.0)
        
        u_entropy = observation.get('policy_entropy', 0.0)
        
        # u_volatility (EMA of u_delta)
        alpha = 0.3
        if len(self.u_delta_history) > 0:
            prev_volatility = self.u_volatility_history[-1] if self.u_volatility_history else 0.0
            u_volatility = alpha * u_delta + (1 - alpha) * prev_volatility
        else:
            u_volatility = u_delta
        
        self.u_delta_history.append(u_delta)
        self.u_entropy_history.append(u_entropy)
        self.u_volatility_history.append(u_volatility)
        
        return InstantDiagnostics(
            u_delta=float(u_delta),
            u_entropy=float(u_entropy),
            u_volatility=float(u_volatility),
            trial=self.trial_count
        )
    
    def _create_log_entry(
        self,
        observation: Dict,
        exposure: ExposureAggregates,
        temporal_state: TemporalState,
        instant_diagnostics: InstantDiagnostics,
        mode_scores: Dict,
        selected_mode: GateMode,
        action: int,
        reward: float,
        gate_constraint: str,
        action_metadata: Dict
    ) -> AgentLog:
        """Создаёт расширенный лог entry для Stage 3.1."""
        return AgentLog(
            seed=self.seed,
            trial=self.trial_count,
            tick=self.trial_count,
            node_id=observation.get('node_id', 'unknown'),
            at_junction=observation.get('at_junction', False),
            state_machine_state=observation.get('state', 'TRAVERSING_PATH'),
            mode_before='',
            mode_after=str(selected_mode),
            gate_trigger=gate_constraint,
            q_values=action_metadata.get('q_values', []),
            risk_values=action_metadata.get('risk_values', []),
            action_probs=action_metadata.get('action_probs', []),
            sampled_action=action,
            candidate_path=observation.get('candidate_path', ''),
            committed_path=observation.get('committed_path', ''),
            reward=reward,
            salience=instant_diagnostics.u_delta,
            u_delta=instant_diagnostics.u_delta,
            u_entropy=instant_diagnostics.u_entropy,
            u_volatility=instant_diagnostics.u_volatility,
            X_risk=exposure.X_risk,
            X_opp=exposure.X_opp,
            D_est=exposure.D_est,
            h_risk=temporal_state.h_risk,
            h_opp=temporal_state.h_opp,
            h_time=temporal_state.h_time,
            one_shot_fired=temporal_state.one_shot_pending
        )
    
    def reset(self):
        """Сбрасывает внутреннее состояние агента."""
        self.current_temporal_state = TemporalState.zeros()
        self.trial_count = 0
        self.log_buffer = []
        self.u_delta_history = []
        self.u_entropy_history = []
        self.u_volatility_history = []
    
    def get_current_state(self) -> Dict:
        """Возвращает текущее внутреннее состояние."""
        return {
            'trial': self.trial_count,
            'temporal_state': {
                'h_risk': self.current_temporal_state.h_risk,
                'h_opp': self.current_temporal_state.h_opp,
                'h_time': self.current_temporal_state.h_time
            },
            'compatibility_mode': self.config.compatibility_mode,
            'log_buffer_size': len(self.log_buffer)
        }
    
# =============================================================================
# CONVENIENCE FUNCTIONS (для тестирования)
# =============================================================================

def create_test_agent(
    compatibility_mode: bool = False,
    log_level: int = 1,
    seed: Optional[int] = None
) -> AgentStage3:
    """
    Создаёт тестового агента Stage 3.0.
    
    Используется в интеграционных тестах (stage3/tests/).
    
    Args:
        compatibility_mode: Включить ли Stage 2 emulation
        log_level: Уровень логирования (0=none, 1=summary, 2=full)
        seed: Random seed для воспроизводимости
    
    Returns:
        AgentStage3 конфигурированный для тестирования
    """
    config = AgentStage3Config(
        compatibility_mode=compatibility_mode,
        log_level=log_level
    )
    return AgentStage3(config, seed=seed)    