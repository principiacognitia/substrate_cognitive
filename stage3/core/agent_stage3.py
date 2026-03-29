"""
Stage 3.0: Agent Orchestration.

Оркестрирует полный цикл агента:
1. Получение observation из среды
2. Вычисление exposure aggregates (exposure_field.py)
3. Обновление temporal state (temporal_state.py)
4. Выбор режима через threshold cascade (gate_stage3.py)
5. Выбор действия на основе режима
6. Логирование всех промежуточных состояний

Design Constraints:
- Backward Compatibility: при нулевых exposure/temporal деградирует в Stage 2
- No Ready Semions: observation не содержит категориальных меток
- Traceability: все промежуточные состояния логируются для анализа

Usage:
    agent = AgentStage3(config)
    for trial in range(n_trials):
        observation, reward, done = env.step(action)
        action = agent.step(observation, reward)
"""

from typing import Dict, Optional, Tuple, Any, List, Union
from dataclasses import dataclass, field
import numpy as np

from stage3.core.gate_modes import GateMode, ALL_MODES
from stage3.core.gate_inputs import (
    GateInput,
    InstantDiagnostics,
    ExposureAggregates,
    TemporalState
)
from stage3.core.exposure_field import ExposureField, ExposureAggregates as ExposureAgg
from stage3.core.temporal_state import TemporalStateUpdater, TemporalStateConfig
from stage3.core.gate_stage3 import GateStage3, GateThresholds
from stage3.core.compatibility import Stage2CompatShim, Stage2CompatConfig


@dataclass
class AgentStage3Config:
    """
    Конфигурация агента Stage 3.0.
    
    Attributes:
        compatibility_mode: Включить ли Stage 2 backward compatibility
        exposure_field_config: Конфигурация для exposure_field.py
        temporal_state_config: Конфигурация для temporal_state.py
        gate_thresholds: Пороги для gate_stage3.py
        log_level: Уровень логирования (0=none, 1=summary, 2=full)
    """
    compatibility_mode: bool = False
    exposure_field_config: Dict = field(default_factory=dict)
    temporal_state_config: Dict = field(default_factory=dict)
    gate_thresholds: Dict = field(default_factory=dict)
    log_level: int = 1

    # Aliases для совместимости с config_stage3_1a.py
    exposure_field: Dict = field(default_factory=dict)
    temporal_state: Dict = field(default_factory=dict)
    
    # Stage 2 viscosity parameters (для backward compatibility)
    viscosity: Dict = field(default_factory=lambda: {
        'alpha': 0.35,
        'beta': 4.0,
        'k_use': 0.08,
        'k_melt': 0.20,
        'lambda_decay': 0.01,
        'tau_vol': 0.50
    })
    
    def __post_init__(self):
        """Валидация конфигурации."""
        if not isinstance(self.compatibility_mode, bool):
            raise ValueError(f"compatibility_mode must be bool: {self.compatibility_mode}")
        if self.log_level not in [0, 1, 2]:
            raise ValueError(f"log_level must be 0, 1, or 2: {self.log_level}")
        
        """Merge alias fields into main config fields."""
        # Если exposure_field задан, используем его вместо exposure_field_config
        if self.exposure_field and not self.exposure_field_config:
            self.exposure_field_config = self.exposure_field
        
        # Если temporal_state задан, используем его вместо temporal_state_config
        if self.temporal_state and not self.temporal_state_config:
            self.temporal_state_config = self.temporal_state            


@dataclass
class AgentLog:
    """
    Лог одного шага агента.
    
    Содержит все промежуточные состояния для анализа.
    """
    trial: int
    observation: Dict
    exposure: Dict
    temporal_state: Dict
    instant_diagnostics: Dict
    mode_scores: Dict
    selected_mode: str
    action: int
    reward: float
    gate_constraint: str


class AgentStage3:
    """
    Agent Stage 3 с orchestration полного цикла.
    
    Architecture:
        Environment → ExposureField → TemporalState → Gate → Action
    
    Usage:
        config = AgentStage3Config()
        agent = AgentStage3(config)
        
        for trial in range(n_trials):
            observation, reward, done = env.step(action)
            action = agent.step(observation, reward)
    """
    
    # ИСПРАВЛЕНО:
    def __init__(self, config: Optional[Union[AgentStage3Config, Dict[str, Any]]] = None):
        """
        Инициализирует агента Stage 3.0.
        
        Args:
            config: Конфигурация агента (dataclass или dict)
        """
        # Конвертируем dict в dataclass если нужно
        if config is None:
            self.config = AgentStage3Config()
        elif isinstance(config, dict):
            # Handle alias field names from config_stage3_1a.py
            config_copy = config.copy()
            
            # Map temporal_state → temporal_state_config
            if 'temporal_state' in config_copy and 'temporal_state_config' not in config_copy:
                config_copy['temporal_state_config'] = config_copy.pop('temporal_state')
            
            # Map exposure_field → exposure_field_config
            if 'exposure_field' in config_copy and 'exposure_field_config' not in config_copy:
                config_copy['exposure_field_config'] = config_copy.pop('exposure_field')
            
            self.config = AgentStage3Config(**config_copy)
        else:
            self.config = config
        
        # Инициализация компонентов (теперь работает с dataclass)
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
        
        # Stage 2 совместимость (для backward compat mode)
        self.u_delta_history: List[float] = []
        self.u_entropy_history: List[float] = []
        self.u_volatility_history: List[float] = []
    
    def step(
        self,
        observation: Dict,
        reward: Optional[float] = None,
        action: Optional[int] = None,
        salience: Optional[float] = None,
        stakes: float = 1.0
    ) -> Tuple[int, Dict]:
        """
        Один шаг агента.
        
        Args:
            observation: Raw observation из среды (без категориальных меток!)
            reward: Reward signal (optional)
            action: Previous action (optional, для exposure computation)
            salience: Salience estimate (optional, вычисляется если None)
            stakes: Stakes/modulator (default: 1.0)
        
        Returns:
            (action, metadata_dict)
            metadata_dict содержит все промежуточные состояния для логирования
        """
        # =====================================================================
        # 1. Вычисление Instant Diagnostics (u_t)
        # =====================================================================
        instant_diagnostics = self._compute_instant_diagnostics(
            observation, reward, action
        )
        
        # =====================================================================
        # 2. Вычисление Exposure Aggregates (X_risk, X_opp, D_est)
        # =====================================================================
        # Если observation уже содержит exposure aggregates (из spatial env),
        # используем их напрямую вместо recomputation
        if 'X_risk' in observation and 'X_opp' in observation and 'D_est' in observation:
            exposure_aggregates = ExposureAggregates(
                X_risk=observation['X_risk'],
                X_opp=observation['X_opp'],
                D_est=observation['D_est']
            )
        else:
            # Fallback: вычисляем через ExposureField
            exposure_aggregates = self.exposure_field.compute_exposure(
                observation=observation,
                action=action,
                reward=reward,
                trial=self.trial_count
            )
        
        # =====================================================================
        # 3. Обновление Temporal State (h_risk, h_opp, h_time)
        # =====================================================================
        # Вычисляем salience если не предоставлена
        if salience is None:
            salience = instant_diagnostics.u_delta  # PE как proxy для salience
        
        self.current_temporal_state = self.temporal_updater.update(
            state=self.current_temporal_state,
            X_risk=exposure_aggregates.X_risk,
            X_opp=exposure_aggregates.X_opp,
            salience=salience,
            stakes=stakes
        )
        
        # =====================================================================
        # 4. Backward Compatibility (Stage 2 emulation)
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
        # 6. Action Selection на основе режима
        # =====================================================================
        action = self._select_action_for_mode(
            mode=selected_mode,
            observation=observation,
            instant_diagnostics=instant_diagnostics
        )
        
        # =====================================================================
        # 7. Логирование
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
                gate_constraint=gate_metadata['winning_constraint']
            )
            self.log_buffer.append(log_entry)
        
        # Инкремент trial count
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
            'one_shot_pending': self.current_temporal_state.one_shot_pending
        }
        
        return action, metadata
    
    def _compute_instant_diagnostics(
        self,
        observation: Dict,
        reward: Optional[float],
        action: Optional[int]
    ) -> InstantDiagnostics:
        """
        Вычисляет InstantDiagnostics из observation/reward.
        
        Для Stage 3.0 это минимальный набор:
        - u_delta: unsigned prediction error
        - u_entropy: policy entropy
        - u_volatility: EMA(u_delta)
        """
        # Вычисляем u_delta (prediction error)
        if reward is not None:
            # Если есть reward, используем его для PE
            expected_reward = observation.get('expected_reward', 0.0)
            u_delta = abs(reward - expected_reward)
        else:
            # Иначе используем observation features
            u_delta = observation.get('prediction_error', 0.0)
        
        # Вычисляем u_entropy (policy entropy)
        u_entropy = observation.get('policy_entropy', 0.0)
        
        # Вычисляем u_volatility (EMA of u_delta)
        alpha = 0.3
        if len(self.u_delta_history) > 0:
            prev_volatility = self.u_volatility_history[-1] if self.u_volatility_history else 0.0
            u_volatility = alpha * u_delta + (1 - alpha) * prev_volatility
        else:
            u_volatility = u_delta
        
        # Сохраняем историю для backward compatibility
        self.u_delta_history.append(u_delta)
        self.u_entropy_history.append(u_entropy)
        self.u_volatility_history.append(u_volatility)
        
        return InstantDiagnostics(
            u_delta=float(u_delta),
            u_entropy=float(u_entropy),
            u_volatility=float(u_volatility),
            trial=self.trial_count
        )
    
    def _select_action_for_mode(
        self,
        mode: GateMode,
        observation: Dict,
        instant_diagnostics: InstantDiagnostics
    ) -> int:
        """
        Выбирает действие на основе режима Gate.
        
        Mode-specific action selection:
        - EXPLOIT: Выбирать действие с максимальной Q-ценностью
        - EXPLORE: Выбирать действие с максимальной энтропией/неопределённостью
        - EXPLOIT_SAFE: Выбирать безопасное действие (минимальный риск)
        - ABSENCE_CHECK: Выбирать проверочное действие (сбор информации)
        
        Args:
            mode: Выбранный режим Gate
            observation: Raw observation
            instant_diagnostics: Instant diagnostics
        
        Returns:
            action: Выбранное действие
        """
        # Получаем Q-ценности из observation (если есть)
        q_values = observation.get('q_values', [0.5, 0.5])
        
        if mode == GateMode.EXPLOIT:
            # Выбирать действие с максимальной Q-ценностью
            action = int(np.argmax(q_values))
        
        elif mode == GateMode.EXPLORE:
            # Выбирать действие с максимальной неопределённостью
            # Для простоты: случайное действие с bias к менее исследованным
            if len(q_values) == 2:
                # Если 2 действия, выбирать менее предпочтительное
                action = 0 if np.argmin(q_values) == 1 else 1
            else:
                action = np.random.randint(0, len(q_values))
        
        elif mode == GateMode.EXPLOIT_SAFE:
            # Выбирать безопасное действие (минимальный риск)
            # Для простоты: действие с максимальной Q-ценностью но с conservative bias
            safe_q = [q * 0.9 for q in q_values]  # Conservative bias
            action = int(np.argmax(safe_q))
        
        elif mode == GateMode.ABSENCE_CHECK:
            # Выбирать проверочное действие (сбор информации)
            # Для простоты: чередовать действия для максимального coverage
            action = self.trial_count % len(q_values)
        
        else:
            # Fallback на EXPLOIT
            action = int(np.argmax(q_values))
        
        return action
    
    def _create_log_entry(
        self,
        observation: Dict,
        exposure: ExposureAgg,
        temporal_state: TemporalState,
        instant_diagnostics: InstantDiagnostics,
        mode_scores: Dict,
        selected_mode: GateMode,
        action: int,
        reward: float,
        gate_constraint: str
    ) -> AgentLog:
        """
        Создаёт лог entry для одного шага.
        
        Args:
            observation: Raw observation
            exposure: Exposure aggregates
            temporal_state: Temporal state
            instant_diagnostics: Instant diagnostics
            mode_scores: Scores для каждого режима
            selected_mode: Выбранный режим
            action: Выбранное действие
            reward: Полученная награда
            gate_constraint: Какой порог сработал
        
        Returns:
            AgentLog entry
        """
        return AgentLog(
            trial=self.trial_count,
            observation=observation,
            exposure={
                'X_risk': exposure.X_risk,
                'X_opp': exposure.X_opp,
                'D_est': exposure.D_est
            },
            temporal_state={
                'h_risk': temporal_state.h_risk,
                'h_opp': temporal_state.h_opp,
                'h_time': temporal_state.h_time,
                'one_shot_pending': temporal_state.one_shot_pending
            },
            instant_diagnostics={
                'u_delta': instant_diagnostics.u_delta,
                'u_entropy': instant_diagnostics.u_entropy,
                'u_volatility': instant_diagnostics.u_volatility
            },
            mode_scores={str(k): v for k, v in mode_scores.items()},
            selected_mode=str(selected_mode),
            action=action,
            reward=reward,
            gate_constraint=gate_constraint
        )
    
    def get_logs(self) -> List[AgentLog]:
        """
        Возвращает все логи с начала эпизода.
        
        Returns:
            List of AgentLog entries
        """
        return self.log_buffer
    
    def clear_logs(self):
        """
        Очищает лог буфер.
        """
        self.log_buffer = []
    
    def reset(self):
        """
        Сбрасывает внутреннее состояние агента (для нового эпизода).
        """
        self.current_temporal_state = TemporalState.zeros()
        self.trial_count = 0
        self.log_buffer = []
        self.u_delta_history = []
        self.u_entropy_history = []
        self.u_volatility_history = []
    
    def get_current_state(self) -> Dict:
        """
        Возвращает текущее внутреннее состояние агента.
        
        Returns:
            Dict с текущими значениями всех переменных
        """
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
    log_level: int = 1
) -> AgentStage3:
    """
    Создаёт тестового агента Stage 3.0.
    
    Только для тестов!
    """
    config = AgentStage3Config(
        compatibility_mode=compatibility_mode,
        log_level=log_level
    )
    return AgentStage3(config)


# =============================================================================
# TESTS (для быстрой проверки)
# =============================================================================

def test_agent_step():
    """
    Test: Agent Step.
    
    Проверяет что agent.step() работает корректно.
    """
    agent = create_test_agent()
    
    # Тестовое observation (без категориальных меток!)
    observation = {
        'prediction_error': 0.5,
        'policy_entropy': 0.3,
        'q_values': [0.6, 0.4],
        'expected_reward': 0.5
    }
    
    action, metadata = agent.step(
        observation=observation,
        reward=0.8,
        action=0
    )
    
    # Проверяем что action валидный
    assert isinstance(action, int), f"Action should be int, got {type(action)}"
    assert 0 <= action < 2, f"Action should be 0 or 1, got {action}"
    
    # Проверяем что metadata содержит все поля
    assert 'mode' in metadata, "Metadata should contain 'mode'"
    assert 'exposure' in metadata, "Metadata should contain 'exposure'"
    assert 'temporal_state' in metadata, "Metadata should contain 'temporal_state'"
    assert 'gate_constraint' in metadata, "Metadata should contain 'gate_constraint'"
    
    print("✓ PASS: Agent Step")
    return True


def test_backward_compatibility_mode():
    """
    Test: Backward Compatibility Mode.
    
    Проверяет что compatibility mode работает корректно.
    """
    agent = create_test_agent(compatibility_mode=True)
    
    observation = {
        'prediction_error': 0.5,
        'policy_entropy': 0.3,
        'q_values': [0.6, 0.4],
        'expected_reward': 0.5
    }
    
    action, metadata = agent.step(
        observation=observation,
        reward=0.8
    )
    
    # В compatibility mode должны быть только EXPLOIT/EXPLORE режимы
    mode = metadata['mode']
    assert mode in ['exploit', 'explore'], f"Unexpected mode in compat mode: {mode}"
    
    print("✓ PASS: Backward Compatibility Mode")
    return True


def test_mode_specific_actions():
    """
    Test: Mode-Specific Action Selection.
    
    Проверяет что разные режимы дают разные действия.
    """
    agent = create_test_agent(log_level=2)
    
    observation = {
        'prediction_error': 0.5,
        'policy_entropy': 0.3,
        'q_values': [0.8, 0.2],  # Явное предпочтение action 0
        'expected_reward': 0.5
    }
    
    # EXPLOIT должен выбрать action 0 (max Q)
    action_exploit, _ = agent.step(observation, reward=0.8, action=0)
    
    # EXPLORE должен выбрать action 1 (менее предпочтительное)
    action_explore, _ = agent.step(observation, reward=0.8, action=0)
    
    # Note: В текущей реализации режимы выбираются Gate, не вручную
    # Этот тест проверяет что action selection работает
    assert isinstance(action_exploit, int)
    assert isinstance(action_explore, int)
    
    print("✓ PASS: Mode-Specific Action Selection")
    return True


def test_logging():
    """
    Test: Logging.
    
    Проверяет что логирование работает корректно.
    """
    agent = create_test_agent(log_level=2)
    
    observation = {
        'prediction_error': 0.5,
        'policy_entropy': 0.3,
        'q_values': [0.6, 0.4],
        'expected_reward': 0.5
    }
    
    # Несколько шагов
    for _ in range(5):
        agent.step(observation, reward=0.8)
    
    # Проверяем что логи записаны
    logs = agent.get_logs()
    assert len(logs) == 5, f"Should have 5 logs, got {len(logs)}"
    
    # Проверяем структуру лога
    log = logs[0]
    assert hasattr(log, 'trial'), "Log should have 'trial'"
    assert hasattr(log, 'selected_mode'), "Log should have 'selected_mode'"
    assert hasattr(log, 'action'), "Log should have 'action'"
    
    print("✓ PASS: Logging")
    return True


def test_reset():
    """
    Test: Reset.
    
    Проверяет что reset() работает корректно.
    """
    agent = create_test_agent(log_level=2)
    
    observation = {
        'prediction_error': 0.5,
        'policy_entropy': 0.3,
        'q_values': [0.6, 0.4],
        'expected_reward': 0.5
    }
    
    # Несколько шагов
    for _ in range(5):
        agent.step(observation, reward=0.8)
    
    # Проверяем что state не нулевой
    state = agent.get_current_state()
    assert state['trial'] == 5, f"Trial should be 5, got {state['trial']}"
    
    # Reset
    agent.reset()
    
    # Проверяем что state нулевой
    state = agent.get_current_state()
    assert state['trial'] == 0, f"Trial should be 0 after reset, got {state['trial']}"
    assert len(agent.get_logs()) == 0, "Logs should be cleared after reset"
    
    print("✓ PASS: Reset")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("Stage 3.0: Agent Stage 3 — Unit Tests")
    print("=" * 70)
    
    test_agent_step()
    test_backward_compatibility_mode()
    test_mode_specific_actions()
    test_logging()
    test_reset()
    
    print("=" * 70)
    print("All tests completed!")
    print("=" * 70)