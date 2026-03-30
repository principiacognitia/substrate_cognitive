"""
Stage 3.1A: Open/Covered Choice Maze Environment.

Пространственная задача с центральной развилкой для проверки:
- Exposure-sensitive gating
- VTE-like hesitation на choice point
- One-shot persistence после aversive event
- Backward compatibility со Stage 2

Design Principles:
- Graph-based topology (не grid) для контроля параметров
- Exposure/valence field как отдельный слой (не categorical labels)
- VTE proxies логируются для future IdPhi wrapper
- Downward compatibility со Stage 2 logging format
- No ready semions at port (Gate получает aggregates, не метки)

Usage:
    from stage3.envs.open_covered_choice_env import OpenCoveredChoiceEnv
    from stage3.configs.config_stage3_1a import CONFIG_3_1A
    
    env = OpenCoveredChoiceEnv(CONFIG_3_1A['env'])
    for trial in range(n_trials):
        observation, reward, done, info = env.step(action)
        env.reset()

Author: Alex Snow (Aleksey L. Snigirov)
License: MIT
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from pathlib import Path

from stage3.envs.maze_builder import MazeBuilder, MazeGraph, NodeType
from stage3.core.gate_inputs import ExposureAggregates


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EnvState:
    """
    Внутреннее состояние среды.
    
    Attributes:
        current_node: ID текущей ноды
        previous_node: ID предыдущей ноды (для backtrack detection)
        trial: Номер текущего триала
        tick: Номер тика внутри триала
        at_junction: Флаг нахождения в junction node
        candidate_path: Текущий candidate path (open/covered)
        committed_path: Выбранный path (после commit)
        path_choice: Финальный выбор пути ('open' или 'covered')
        junction_entry_tick: Тик входа в junction
        junction_pause_duration: Длительность паузы на junction
        reorientation_count: Количество смен candidate_path
        retreat_return_count: Количество retreat→return событий
        commit_latency: Тиков до окончательного commit
        trial_reward: Суммарная награда за триал
        trial_complete: Флаг завершения триала
    """
    current_node: str = "start"
    previous_node: str = ""
    trial: int = 0
    tick: int = 0
    at_junction: bool = False
    candidate_path: Optional[str] = None
    committed_path: Optional[str] = None
    path_choice: Optional[str] = None
    junction_entry_tick: int = 0
    junction_pause_duration: int = 0
    reorientation_count: int = 0
    retreat_return_count: int = 0
    commit_latency: int = 0
    trial_reward: float = 0.0
    trial_complete: bool = False
    
    # Pseudo-kinematic state (для future IdPhi wrapper)
    heading_state: float = 0.0
    heading_change: float = 0.0
    candidate_heading_switches: int = 0
    zone_entry_tick: int = 0
    zone_exit_tick: int = 0
    
    def reset_trial(self, trial: int) -> None:
        """Сбрасывает состояние для нового триала."""
        self.current_node = "start"
        self.previous_node = ""
        self.trial = trial
        self.tick = 0
        self.at_junction = False
        self.candidate_path = None
        self.committed_path = None
        self.path_choice = None
        self.junction_entry_tick = 0
        self.junction_pause_duration = 0
        self.reorientation_count = 0
        self.retreat_return_count = 0
        self.commit_latency = 0
        self.trial_reward = 0.0
        self.trial_complete = False
        self.heading_state = 0.0
        self.heading_change = 0.0
        self.candidate_heading_switches = 0
        self.zone_entry_tick = 0
        self.zone_exit_tick = 0


@dataclass
class TrialSummary:
    """
    Сводка триала (для trial log).
    
    Соответствует LOGGING_CONFIG['trial_log_fields'] из config_stage3_1a.py.
    """
    seed: int = 0
    trial: int = 0
    path_choice: str = ""
    reward_total: float = 0.0
    junction_pause_duration: int = 0
    reorientation_count: int = 0
    retreat_return_count: int = 0
    commit_latency: int = 0
    junction_deliberation_proxy: float = 0.0
    mode_at_junction: str = ""
    final_mode: str = ""
    one_shot_fired: bool = False
    config_name: str = "stage3_1a"
    ablation_name: str = "full"
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в dict для CSV logging."""
        return {
            'seed': self.seed,
            'trial': self.trial,
            'path_choice': self.path_choice,
            'reward_total': self.reward_total,
            'junction_pause_duration': self.junction_pause_duration,
            'reorientation_count': self.reorientation_count,
            'retreat_return_count': self.retreat_return_count,
            'commit_latency': self.commit_latency,
            'junction_deliberation_proxy': self.junction_deliberation_proxy,
            'mode_at_junction': self.mode_at_junction,
            'final_mode': self.final_mode,
            'one_shot_fired': self.one_shot_fired,
            'config_name': self.config_name,
            'ablation_name': self.ablation_name
        }


# =============================================================================
# ENVIRONMENT
# =============================================================================

class OpenCoveredChoiceEnv:
    """
    Open/Covered Choice Maze Environment (Stage 3.1A).
    
    Topology:
        start -- junction -- open_mid -- goal
                     |
                     -- covered_mid -- goal
    
    Agent должен выбрать между open path (высокая экспозиция) и
    covered path (низкая экспозиция) при одинаковой награде.
    
    Attributes:
        maze: MazeGraph с топологией и exposure field
        config: Конфигурация среды
        state: Текущее состояние EnvState
        trial_summaries: List[TrialSummary] для всех триалов
        rng: numpy random generator для воспроизводимости
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        seed: int = 42
    ):
        """
        Инициализирует среду.
        
        Args:
            config: Конфигурация из CONFIG_3_1A['env']
            seed: Random seed для воспроизводимости
        """
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Создаём maze через MazeBuilder
        self.builder = MazeBuilder()
        self.maze = self._build_maze()
        
        # Внутреннее состояние
        self.state = EnvState()
        self.trial_summaries: List[TrialSummary] = []
        
        # Agent state (для exposure computation)
        self.agent_exposure_history: List[Dict[str, float]] = []
        
        # VTE configuration
        self.vte_config = config.get('vte', {})
        self.junction_node = self.config['topology']['junction_node']
        self.goal_node = self.config['topology']['goal_node']
        self.start_node = self.config['topology']['start_node']
        
        # Temporal parameters
        self.junction_pause_min = self.config['temporal']['junction_pause_min']
        self.junction_pause_max = self.config['temporal']['junction_pause_max']
        
    def _build_maze(self) -> MazeGraph:
        """
        Строит maze topology из конфигурации.
        
        Returns:
            MazeGraph с настроенной топологией
        """
        paths = self.config['paths']
        
        maze = self.builder.create_open_covered_maze(
            open_exposure=paths['open']['exposure_profile']['X_risk'],
            covered_exposure=paths['covered']['exposure_profile']['X_risk'],
            path_length=paths['open']['length'],
            reward_equal=(paths['open']['base_reward'] == paths['covered']['base_reward']),
            reward_value=paths['open']['base_reward']
        )
        
        # Валидация maze
        is_valid, errors = self.builder.validate_maze()
        if not is_valid:
            raise ValueError(f"Maze validation failed: {errors}")
        
        return maze
    
    def reset(self, trial: Optional[int] = None) -> Dict[str, Any]:
        """
        Сбрасывает среду для нового триала.
        
        Args:
            trial: Номер триала (авто-инкремент если None)
        
        Returns:
            observation: Initial observation dict
        """
        if trial is None:
            trial = self.state.trial + 1
        
        self.state.reset_trial(trial)
        
        # Получаем начальное observation
        observation = self._get_observation()
        
        return observation
    
    def step(
        self,
        action: int,
        mode: str = "EXPLOIT",
        gate_trigger: str = "default"
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Один шаг среды.
        
        Args:
            action: Выбранное действие (0 = forward, 1 = left, 2 = right, etc.)
            mode: Текущий режим Gate (для logging)
            gate_trigger: Какой порог Gate сработал (для logging)
        
        Returns:
            observation: Observation dict
            reward: Reward signal
            done: Флаг завершения триала
            info: Dict с дополнительной информацией
        """
        self.state.tick += 1

        # Сохраняем previous_node в state ПЕРЕД движением
        self.state.previous_node = self.state.current_node
        
        # =====================================================================
        # 1. Движение по графу
        # =====================================================================
        self._move(action, mode)
        
        # =====================================================================
        # 2. VTE proxy detection (на junction)
        # =====================================================================
        self._update_vte_proxies(mode)
        
        # =====================================================================
        # 3. Вычисление observation
        # =====================================================================
        observation = self._get_observation()
        
        # =====================================================================
        # 4. Вычисление reward
        # =====================================================================
        reward = self._compute_reward()
        self.state.trial_reward += reward
        
        # =====================================================================
        # 5. Проверка завершения триала
        # =====================================================================
        done = self._check_trial_complete()
        
        if done:
            self._finalize_trial(mode, gate_trigger)
        
        # =====================================================================
        # 6. Info dict (для logging)
        # =====================================================================
        info = {
            'trial': self.state.trial,
            'tick': self.state.tick,
            'node_id': self.state.current_node,
            'previous_node': self.state.previous_node,
            'at_junction': self.state.at_junction,
            'candidate_path': self.state.candidate_path,
            'committed_path': self.state.committed_path,
            'mode': mode,
            'gate_trigger': gate_trigger,
            'vte_proxies': {
                'junction_pause_duration': self.state.junction_pause_duration,
                'reorientation_count': self.state.reorientation_count,
                'retreat_return_count': self.state.retreat_return_count,
                'commit_latency': self.state.commit_latency
            }
        }
        
        return observation, reward, done, info
    
    def _move(self, action: int, mode: str) -> None:
        """
        Обновляет позицию агента на графе.
        
        Args:
            action: Выбранное действие
            mode: Режим Gate (влияет на движение на junction)
        """
        current = self.state.current_node
        neighbors = self.maze.get_neighbors(current)
        
        # =====================================================================
        # Special handling для junction node
        # =====================================================================
        if current == self.junction_node:
            self.state.at_junction = True
            
            # Если ещё не committed, выбираем path на основе action
            if self.state.committed_path is None:
                # Action 0 = open path, Action 1 = covered path
                if action == 0:
                    self.state.candidate_path = "open"
                    self.state.committed_path = "open"
                    self.state.commit_latency = self.state.tick - self.state.junction_entry_tick
                elif action == 1:
                    self.state.candidate_path = "covered"
                    self.state.committed_path = "covered"
                    self.state.commit_latency = self.state.tick - self.state.junction_entry_tick
                
                # Перемещаем на следующую ноду выбранного пути
                if self.state.committed_path == "open":
                    self.state.current_node = "open_mid"
                else:
                    self.state.current_node = "covered_mid"
            else:
                # Уже committed, движемся по пути
                self._move_along_path(action)
        
        # =====================================================================
        # Start node → Junction
        # =====================================================================
        elif current == self.start_node:
            self.state.current_node = self.junction_node
            self.state.junction_entry_tick = self.state.tick
            self.state.at_junction = True
        
        # =====================================================================
        # Path mid → Goal
        # =====================================================================
        elif current in ["open_mid", "covered_mid"]:
            self.state.current_node = self.goal_node
        
        # =====================================================================
        # Goal node (триал завершён)
        # =====================================================================
        elif current == self.goal_node:
            pass  # Триал завершён, движение не происходит
        
    
    def _move_along_path(self, action: int) -> None:
        """
        Движение по выбранному пути (после commit).
        
        Args:
            action: Действие (игнорируется после commit, движемся вперёд)
        """
        if self.state.committed_path == "open":
            self.state.current_node = "goal"
        elif self.state.committed_path == "covered":
            self.state.current_node = "goal"
    
    def _update_vte_proxies(self, mode: str) -> None:
        """
        Обновляет VTE proxy метрики.
        
        Args:
            mode: Текущий режим Gate
        """
        # =====================================================================
        # Junction pause detection
        # =====================================================================
        if self.state.at_junction and self.state.committed_path is None:
            # Агент ещё не выбрал путь — считаем паузу
            self.state.junction_pause_duration = self.state.tick - self.state.junction_entry_tick
        
        # =====================================================================
        # Reorientation detection (смена candidate_path)
        # =====================================================================
        if self.state.at_junction:
            current_candidate = self.state.candidate_path
            
            # Если mode сменился и это влияет на candidate
            if mode == "EXPLORE" and self.state.candidate_path is not None:
                # EXPLORE mode может вызвать reorientation
                prev_candidate = self.state.candidate_path
                # В реальной реализации здесь была бы логика смены candidate
                # Для Stage 3.1A упрощённо:
                if self.rng.random() < 0.1:  # 10% chance reorientation в EXPLORE
                    self.state.reorientation_count += 1
                    self.state.candidate_heading_switches += 1
        
        # =====================================================================
        # Retreat-return detection (возврат в start после junction)
        # =====================================================================
        if self.state.previous_node == self.junction_node and self.state.current_node == self.start_node:
            self.state.retreat_return_count += 1
        
        # =====================================================================
        # Heading state update (pseudo-kinematic для IdPhi wrapper)
        # =====================================================================
        if self.state.at_junction:
            # Имитация heading change на junction
            if self.state.committed_path is None:
                # Ещё не выбрал — heading oscillates
                self.state.heading_change = self.rng.uniform(-0.5, 0.5)
            else:
                # Выбрал путь — heading stabilizes
                self.state.heading_change = 0.0
    
    def _get_observation(self) -> Dict[str, Any]:
        """
        Получает observation для агента.
        
        Design Constraint: No Ready Semions at Port.
        Observation не содержит categorical labels ("open", "covered").
        Только numeric aggregates.
        
        Returns:
            observation: Dict с numeric features
        """
        current = self.state.current_node
        
        # Получаем exposure profile из maze
        exposure = self.builder.get_exposure_at_node(current)
        
        # Вычисляем diagnostic variables
        u_delta = self._compute_prediction_error()
        u_entropy = self._compute_policy_entropy()
        u_volatility = self._compute_volatility()
        
        # Observation (только numeric, no strings!)
        observation = {
            # === Diagnostic Vector (u_t) ===
            'prediction_error': u_delta,
            'policy_entropy': u_entropy,
            'volatility': u_volatility,
            
            # === Exposure Aggregates (X_risk, X_opp, D_est) ===
            'X_risk': exposure['X_risk'],
            'X_opp': exposure['X_opp'],
            'D_est': exposure['D_est'],
            
            # === Spatial State (numeric encoding) ===
            'node_id_encoded': self._encode_node_id(current),
            'distance_to_goal': self._compute_distance_to_goal(),
            'at_junction': 1.0 if self.state.at_junction else 0.0,
            
            # === Q-values (для action selection) ===
            'q_values': self._get_q_values(),
            'expected_reward': self._get_expected_reward(),
            
            # === Temporal State (для agent) ===
            'tick': self.state.tick,
            'trial': self.state.trial,
            
            # === Pseudo-kinematic (для IdPhi wrapper) ===
            'heading_state': self.state.heading_state,
            'heading_change': self.state.heading_change
        }
        
        return observation
    
    def _compute_prediction_error(self) -> float:
        """Вычисляет prediction error (u_delta)."""
        expected = self._get_expected_reward()
        actual = self._compute_reward()
        return abs(actual - expected)
    
    def _compute_policy_entropy(self) -> float:
        """Вычисляет policy entropy (u_entropy)."""
        q_values = self._get_q_values()
        q_normalized = np.array(q_values) / (np.sum(q_values) + 1e-10)
        entropy = -np.sum(q_normalized * np.log(q_normalized + 1e-10))
        return float(entropy)
    
    def _compute_volatility(self) -> float:
        """Вычисляет volatility estimate (u_volatility)."""
        if len(self.agent_exposure_history) < 2:
            return 0.0
        
        # EMA prediction errors
        recent_errors = [abs(h.get('prediction_error', 0.0)) for h in self.agent_exposure_history[-10:]]
        volatility = float(np.std(recent_errors))
        
        return volatility
    
    def _compute_reward(self) -> float:
        """
        Вычисляет reward для текущего шага.
        
        Returns:
            reward: Reward signal
        """
        current = self.state.current_node
        
        if current == self.goal_node:
            # Награда в goal zone
            paths = self.config['paths']
            if self.state.path_choice == "open":
                return paths['open']['base_reward']
            elif self.state.path_choice == "covered":
                return paths['covered']['base_reward']
            else:
                return paths['open']['base_reward']  # Default
        else:
            return 0.0  # Нет награды в пути
    
    def _check_trial_complete(self) -> bool:
        """
        Проверяет завершение триала.
        
        Returns:
            done: True если триал завершён
        """
        if self.state.current_node == self.goal_node:
            self.state.trial_complete = True
            self.state.path_choice = self.state.committed_path
            return True
        
        return False
    
    def _finalize_trial(self, mode: str, gate_trigger: str) -> None:
        """
        Финализирует триал (создаёт TrialSummary).
        
        Args:
            mode: Финальный режим Gate
            gate_trigger: Какой порог сработал
        """
        # Вычисляем junction_deliberation_proxy (z-scored mean)
        vte_metrics = [
            np.log1p(self.state.junction_pause_duration),
            self.state.reorientation_count,
            self.state.retreat_return_count,
            np.log1p(self.state.commit_latency)
        ]
        
        # Z-score (упрощённо, без полноценной нормализации)
        z_metrics = [(x - np.mean(vte_metrics)) / (np.std(vte_metrics) + 1e-10) for x in vte_metrics]
        deliberation_proxy = float(np.mean(z_metrics))
        
        summary = TrialSummary(
            seed=self.seed,
            trial=self.state.trial,
            path_choice=self.state.path_choice or "unknown",
            reward_total=self.state.trial_reward,
            junction_pause_duration=self.state.junction_pause_duration,
            reorientation_count=self.state.reorientation_count,
            retreat_return_count=self.state.retreat_return_count,
            commit_latency=self.state.commit_latency,
            junction_deliberation_proxy=deliberation_proxy,
            mode_at_junction=mode,  # Упрощённо
            final_mode=mode,
            one_shot_fired=False,  # Будет установлено agent'ом
            config_name="stage3_1a",
            ablation_name="full"
        )
        
        self.trial_summaries.append(summary)
    
    def _encode_node_id(self, node_id: str) -> float:
        """
        Кодирует node_id в numeric value (no strings to Gate!).
        
        Args:
            node_id: ID ноды
        
        Returns:
            encoded: Numeric encoding
        """
        encoding = {
            'start': 0.0,
            'junction': 0.5,
            'open_mid': 0.75,
            'covered_mid': 0.25,
            'goal': 1.0
        }
        return encoding.get(node_id, 0.0)
    
    def _compute_distance_to_goal(self) -> float:
        """
        Вычисляет расстояние до goal (в тиках).
        
        Returns:
            distance: Расстояние
        """
        distances = {
            'start': 4.0,
            'junction': 3.0,
            'open_mid': 1.0,
            'covered_mid': 1.0,
            'goal': 0.0
        }
        return distances.get(self.state.current_node, 0.0)
    
    def _get_q_values(self) -> List[float]:
        """
        Получает Q-values для действий.
        
        Returns:
            q_values: List[float]
        """
        # Упрощённая реализация (для Stage 3.1A)
        if self.state.current_node == self.junction_node:
            # На junction два выбора: open или covered
            return [0.5, 0.5]  # Изначально равные
        else:
            return [1.0]  # Одно действие (вперёд к goal)
    
    def _get_expected_reward(self) -> float:
        """
        Получает ожидаемую награду.
        
        Returns:
            expected_reward: Float
        """
        paths = self.config['paths']
        return paths['open']['base_reward']  # Default expectation
    
    def get_trial_summaries(self) -> List[TrialSummary]:
        """
        Возвращает все TrialSummary.
        
        Returns:
            List[TrialSummary]
        """
        return self.trial_summaries
    
    def get_current_state(self) -> EnvState:
        """
        Возвращает текущее состояние.
        
        Returns:
            EnvState
        """
        return self.state
    
    def get_maze(self) -> MazeGraph:
        """
        Возвращает MazeGraph.
        
        Returns:
            MazeGraph
        """
        return self.maze
    
    def save_logs(self, output_dir: str, filename: str) -> None:
        """
        Сохраняет логи в CSV файл внутри директории.
        
        Args:
            output_dir: Директория для сохранения (существующая)
            filename: Имя файла (например, "stage3_1a_seed42_trials.csv")
        """
        dir_path = Path(output_dir)
        dir_path.mkdir(parents=True, exist_ok=True)  # Создаём директорию
        
        file_path = dir_path / filename  # Полный путь к файлу
        
        if self.trial_summaries:
            df_trials = pd.DataFrame([s.to_dict() for s in self.trial_summaries])
            df_trials.to_csv(file_path, index=False)
            print(f"  Logs saved: {file_path.name} ({len(df_trials)} trials)")
        else:
            print(f"  ⚠ No trial summaries to save")


# =============================================================================
# TESTS
# =============================================================================

def test_env_creation():
    """
    Test: Environment creation.
    """
    from stage3.configs.config_stage3_1a import CONFIG_3_1A
    
    env = OpenCoveredChoiceEnv(CONFIG_3_1A['env'], seed=42)
    
    # Check maze created
    assert env.maze is not None
    assert "start" in env.maze.nodes
    assert "junction" in env.maze.nodes
    assert "goal" in env.maze.nodes
    
    # Check state initialized
    assert env.state.current_node == "start"
    assert env.state.trial == 0
    
    print("✓ PASS: Environment creation")
    return True


def test_env_step():
    """
    Test: Environment step.
    """
    from stage3.configs.config_stage3_1a import CONFIG_3_1A
    
    env = OpenCoveredChoiceEnv(CONFIG_3_1A['env'], seed=42)
    
    # Reset
    observation = env.reset(trial=1)
    
    # Check observation has required fields
    assert 'X_risk' in observation
    assert 'X_opp' in observation
    assert 'D_est' in observation
    assert 'prediction_error' in observation
    assert 'q_values' in observation
    
    # Step to junction
    observation, reward, done, info = env.step(action=0, mode="EXPLOIT")
    assert env.state.current_node == "junction"
    assert env.state.at_junction == True
    
    # Step to goal (choose open path)
    observation, reward, done, info = env.step(action=0, mode="EXPLOIT")
    assert env.state.current_node in ["open_mid", "covered_mid"]
    
    # Step to goal
    observation, reward, done, info = env.step(action=0, mode="EXPLOIT")
    assert done == True
    
    print("✓ PASS: Environment step")
    return True


def test_vte_proxies():
    """
    Test: VTE proxy metrics.
    """
    from stage3.configs.config_stage3_1a import CONFIG_3_1A
    
    env = OpenCoveredChoiceEnv(CONFIG_3_1A['env'], seed=42)
    
    # Reset
    env.reset(trial=1)
    
    # Step to junction
    env.step(action=0, mode="EXPLOIT")
    assert env.state.at_junction == True
    
    # Check VTE proxies initialized
    assert env.state.junction_pause_duration >= 0
    assert env.state.reorientation_count >= 0
    
    print("✓ PASS: VTE proxies")
    return True


def test_no_ready_semions():
    """
    Test: No categorical labels in observation.
    """
    from stage3.configs.config_stage3_1a import CONFIG_3_1A
    
    env = OpenCoveredChoiceEnv(CONFIG_3_1A['env'], seed=42)
    env.reset(trial=1)
    
    # Несколько шагов
    for _ in range(5):
        observation = env._get_observation()
        
        # Проверяем что нет строковых меток
        for key, value in observation.items():
            if key == 'path_choice':
                continue  # Это internal state, не observation
            assert not isinstance(value, str), f"Observation field {key} is string: {value}"
        
        env.step(action=0, mode="EXPLOIT")
    
    print("✓ PASS: No ready semions in observation")
    return True


def test_trial_completion():
    """
    Test: Trial completion and summary.
    """
    from stage3.configs.config_stage3_1a import CONFIG_3_1A
    
    env = OpenCoveredChoiceEnv(CONFIG_3_1A['env'], seed=42)
    
    # Полный триал
    env.reset(trial=1)
    
    # Start → Junction
    env.step(action=0, mode="EXPLOIT")
    # Junction → Path mid
    env.step(action=0, mode="EXPLOIT")
    # Path mid → Goal
    observation, reward, done, info = env.step(action=0, mode="EXPLOIT")
    
    assert done == True
    assert len(env.trial_summaries) == 1
    
    # Check summary fields
    summary = env.trial_summaries[0]
    assert summary.trial == 1
    assert summary.path_choice in ["open", "covered"]
    assert summary.reward_total >= 0
    
    print("✓ PASS: Trial completion")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("Stage 3.1A: Open/Covered Choice Env — Unit Tests")
    print("=" * 70)
    
    test_env_creation()
    test_env_step()
    test_vte_proxies()
    test_no_ready_semions()
    test_trial_completion()
    
    print("=" * 70)
    print("All tests passed!")
    print("=" * 70)