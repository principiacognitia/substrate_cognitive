"""
Stage 3.1: Maze Builder Module.

Создаёт графовую топологию лабиринта + экспозиционно-валентное поле.

Design Principles:
- Graph-based topology (не grid) для контроля параметров
- Exposure/valence field как отдельный слой (не categorical labels)
- Поддержка расширения до Redish-style мазов (Box 1 из Redish 2016)
- Downward compatibility со Stage 2 logging format

Usage:
    builder = MazeBuilder()
    maze = builder.create_open_covered_maze()
    env = OpenCoveredChoiceEnv(maze)

Author: Alex Snow (Aleksey L. Snigirov)
License: MIT
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from enum import Enum


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class NodeType(Enum):
    """
    Типы нод в графе лабиринта.
    
    Note: Это internal representation для builder, не передаётся в Gate.
    Gate получает только numeric aggregates (X_risk, X_opp, D_est).
    """
    START = "start"
    JUNCTION = "junction"
    PATH = "path"
    GOAL = "goal"


class PathType(Enum):
    """
    Типы путей между нодами.
    
    Note: Используется для вычисления exposure profile, не передаётся в Gate.
    """
    OPEN = "open"
    COVERED = "covered"


@dataclass
class Node:
    """
    Нода в графе лабиринта.
    
    Attributes:
        node_id: Уникальный идентификатор
        node_type: Тип ноды (start, junction, path, goal)
        position: 2D координаты (x, y) для визуализации
        exposure_profile: Базовый exposure профиль ноды [X_risk, X_opp, D_est]
        valence_profile: Базовый valence профиль (для reward zones)
    """
    node_id: str
    node_type: NodeType
    position: Tuple[float, float] = (0.0, 0.0)
    exposure_profile: Dict[str, float] = field(default_factory=lambda: {
        'X_risk': 0.0,
        'X_opp': 0.0,
        'D_est': 0.0
    })
    valence_profile: Dict[str, float] = field(default_factory=lambda: {
        'nu': 0.0,
        'stakes': 1.0
    })


@dataclass
class Edge:
    """
    Ребро (путь) между нодами.
    
    Attributes:
        edge_id: Уникальный идентификатор
        from_node: ID ноды начала
        to_node: ID ноды конца
        path_type: Тип пути (open/covered)
        length: Длина пути (в тиках/шагах)
        exposure_profile: Exposure профиль пути
    """
    edge_id: str
    from_node: str
    to_node: str
    path_type: PathType = PathType.OPEN
    length: int = 1
    exposure_profile: Dict[str, float] = field(default_factory=lambda: {
        'X_risk': 0.0,
        'X_opp': 0.0,
        'D_est': 0.0
    })


@dataclass
class MazeGraph:
    """
    Полный граф лабиринта.
    
    Attributes:
        nodes: Dict[node_id, Node]
        edges: Dict[edge_id, Edge]
        adjacency: Dict[node_id, List[node_id]]
        start_node: ID стартовой ноды
        goal_node: ID целевой ноды
        junction_node: ID ноды выбора (если есть)
    """
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: Dict[str, Edge] = field(default_factory=dict)
    adjacency: Dict[str, List[str]] = field(default_factory=dict)
    start_node: Optional[str] = None
    goal_node: Optional[str] = None
    junction_node: Optional[str] = None
    
    def add_node(self, node: Node) -> None:
        """Добавляет ноду в граф."""
        self.nodes[node.node_id] = node
        if node.node_id not in self.adjacency:
            self.adjacency[node.node_id] = []
    
    def add_edge(self, edge: Edge) -> None:
        """Добавляет ребро в граф."""
        self.edges[edge.edge_id] = edge
        # Bidirectional adjacency
        if edge.from_node not in self.adjacency:
            self.adjacency[edge.from_node] = []
        if edge.to_node not in self.adjacency:
            self.adjacency[edge.to_node] = []
        
        self.adjacency[edge.from_node].append(edge.to_node)
        self.adjacency[edge.to_node].append(edge.from_node)
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Возвращает соседей ноды."""
        return self.adjacency.get(node_id, [])
    
    def get_edge_between(self, node1: str, node2: str) -> Optional[Edge]:
        """Возвращает ребро между двумя нодами."""
        for edge in self.edges.values():
            if (edge.from_node == node1 and edge.to_node == node2) or \
               (edge.from_node == node2 and edge.to_node == node1):
                return edge
        return None


# =============================================================================
# MAZE BUILDER
# =============================================================================

class MazeBuilder:
    """
    Builder для создания графов лабиринтов.
    
    Поддерживает:
    - Open/Covered Choice Maze (Stage 3.1A)
    - Threat/Reward Conflict Maze (Stage 3.1B)
    - Расширение до Redish-style мазов (future)
    """
    
    def __init__(self):
        self.graph = MazeGraph()
    
    def create_open_covered_maze(
        self,
        open_exposure: float = 0.7,
        covered_exposure: float = 0.2,
        path_length: int = 3,
        reward_equal: bool = True,
        reward_value: float = 1.0
    ) -> MazeGraph:
        """
        Создаёт минимальный Open/Covered Choice Maze (5 nodes).
        
        Topology:
            start -- junction -- open_mid -- goal
                         |
                         -- covered_mid -- goal
        
        Args:
            open_exposure: Exposure level для open path [0, 1]
            covered_exposure: Exposure level для covered path [0, 1]
            path_length: Длина каждого пути (в тиках)
            reward_equal: Если True, reward одинаковый на обоих путях
            reward_value: Базовое значение reward
        
        Returns:
            MazeGraph с настроенной топологией и exposure field
        """
        self.graph = MazeGraph()
        
        # =====================================================================
        # NODES (5 nodes minimum)
        # =====================================================================
        
        # Start zone
        start = Node(
            node_id="start",
            node_type=NodeType.START,
            position=(0.0, 0.0),
            exposure_profile={'X_risk': 0.1, 'X_opp': 0.0, 'D_est': 0.5},
            valence_profile={'nu': 0.0, 'stakes': 1.0}
        )
        self.graph.add_node(start)
        self.graph.start_node = "start"
        
        # Junction (choice point)
        junction = Node(
            node_id="junction",
            node_type=NodeType.JUNCTION,
            position=(1.0, 0.0),
            exposure_profile={'X_risk': 0.3, 'X_opp': 0.3, 'D_est': 0.8},
            valence_profile={'nu': 0.0, 'stakes': 1.0}
        )
        self.graph.add_node(junction)
        self.graph.junction_node = "junction"
        
        # Open path midpoint
        open_mid = Node(
            node_id="open_mid",
            node_type=NodeType.PATH,
            position=(2.0, 1.0),  # Above junction
            exposure_profile={
                'X_risk': open_exposure,
                'X_opp': 0.5,
                'D_est': 0.9
            },
            valence_profile={'nu': 0.0, 'stakes': 1.0}
        )
        self.graph.add_node(open_mid)
        
        # Covered path midpoint
        covered_mid = Node(
            node_id="covered_mid",
            node_type=NodeType.PATH,
            position=(2.0, -1.0),  # Below junction
            exposure_profile={
                'X_risk': covered_exposure,
                'X_opp': 0.5,
                'D_est': 0.3
            },
            valence_profile={'nu': 0.0, 'stakes': 1.0}
        )
        self.graph.add_node(covered_mid)
        
        # Goal zone
        goal = Node(
            node_id="goal",
            node_type=NodeType.GOAL,
            position=(3.0, 0.0),
            exposure_profile={'X_risk': 0.1, 'X_opp': 1.0, 'D_est': 0.5},
            valence_profile={'nu': 1.0 if reward_equal else 0.5, 'stakes': 1.0}
        )
        self.graph.add_node(goal)
        self.graph.goal_node = "goal"
        
        # =====================================================================
        # EDGES
        # =====================================================================
        
        # Start -> Junction
        edge_start_junction = Edge(
            edge_id="start_junction",
            from_node="start",
            to_node="junction",
            path_type=PathType.COVERED,  # Start zone обычно безопасный
            length=1,
            exposure_profile={'X_risk': 0.1, 'X_opp': 0.0, 'D_est': 0.5}
        )
        self.graph.add_edge(edge_start_junction)
        
        # Junction -> Open Mid
        edge_junction_open = Edge(
            edge_id="junction_open",
            from_node="junction",
            to_node="open_mid",
            path_type=PathType.OPEN,
            length=path_length,
            exposure_profile={
                'X_risk': open_exposure,
                'X_opp': 0.5,
                'D_est': 0.9
            }
        )
        self.graph.add_edge(edge_junction_open)
        
        # Junction -> Covered Mid
        edge_junction_covered = Edge(
            edge_id="junction_covered",
            from_node="junction",
            to_node="covered_mid",
            path_type=PathType.COVERED,
            length=path_length,
            exposure_profile={
                'X_risk': covered_exposure,
                'X_opp': 0.5,
                'D_est': 0.3
            }
        )
        self.graph.add_edge(edge_junction_covered)
        
        # Open Mid -> Goal
        edge_open_goal = Edge(
            edge_id="open_goal",
            from_node="open_mid",
            to_node="goal",
            path_type=PathType.OPEN,
            length=1,
            exposure_profile={
                'X_risk': open_exposure * 0.5,
                'X_opp': 1.0,
                'D_est': 0.8
            }
        )
        self.graph.add_edge(edge_open_goal)
        
        # Covered Mid -> Goal
        edge_covered_goal = Edge(
            edge_id="covered_goal",
            from_node="covered_mid",
            to_node="goal",
            path_type=PathType.COVERED,
            length=1,
            exposure_profile={
                'X_risk': covered_exposure * 0.5,
                'X_opp': 1.0,
                'D_est': 0.4
            }
        )
        self.graph.add_edge(edge_covered_goal)
        
        return self.graph
    
    def create_threat_conflict_maze(
        self,
        open_exposure: float = 0.8,
        covered_exposure: float = 0.2,
        predator_cue: bool = False,
        reward_premium_open: float = 0.0,
        path_length: int = 3
    ) -> MazeGraph:
        """
        Создаёт Threat/Reward Conflict Maze (Stage 3.1B).
        
        Отличия от 3.1A:
        - Predator cue на open path (повышает X_risk)
        - Reward premium на open path (повышает X_opp)
        
        Args:
            open_exposure: Base exposure для open path
            covered_exposure: Base exposure для covered path
            predator_cue: Если True, добавить predator cue на open path
            reward_premium_open: Дополнительный reward на open path [0, 1]
            path_length: Длина путей
        
        Returns:
            MazeGraph с threat/reward conflict
        """
        # Создаём базовый maze
        self.create_open_covered_maze(
            open_exposure=open_exposure,
            covered_exposure=covered_exposure,
            path_length=path_length,
            reward_equal=(reward_premium_open == 0.0),
            reward_value=1.0 + reward_premium_open
        )
        
        # Добавляем predator cue если нужно
        if predator_cue:
            open_mid = self.graph.nodes["open_mid"]
            open_mid.exposure_profile['X_risk'] = min(1.0, open_exposure + 0.3)
            open_mid.valence_profile['stakes'] = 2.0  # Повышенные stakes
        
        # Добавляем reward premium на open path
        if reward_premium_open > 0:
            goal = self.graph.nodes["goal"]
            goal.valence_profile['nu'] = 1.0 + reward_premium_open
            goal.exposure_profile['X_opp'] = min(1.0, 0.5 + reward_premium_open)
        
        return self.graph
    
    def get_exposure_at_node(self, node_id: str) -> Dict[str, float]:
        """
        Возвращает exposure profile для ноды.
        
        Args:
            node_id: ID ноды
        
        Returns:
            Dict с X_risk, X_opp, D_est
        """
        if node_id not in self.graph.nodes:
            raise ValueError(f"Node {node_id} not found in maze")
        
        return self.graph.nodes[node_id].exposure_profile.copy()
    
    def get_exposure_on_edge(self, edge_id: str) -> Dict[str, float]:
        """
        Возвращает exposure profile для ребра.
        
        Args:
            edge_id: ID ребра
        
        Returns:
            Dict с X_risk, X_opp, D_est
        """
        if edge_id not in self.graph.edges:
            raise ValueError(f"Edge {edge_id} not found in maze")
        
        return self.graph.edges[edge_id].exposure_profile.copy()
    
    def get_path_options(self, from_node: str) -> List[str]:
        """
        Возвращает доступные пути из ноды.
        
        Args:
            from_node: ID текущей ноды
        
        Returns:
            List соседних нод
        """
        return self.graph.get_neighbors(from_node)
    
    def get_junction_node(self) -> Optional[str]:
        """Возвращает ID ноды выбора (junction)."""
        return self.graph.junction_node
    
    def get_goal_node(self) -> Optional[str]:
        """Возвращает ID целевой ноды."""
        return self.graph.goal_node
    
    def get_start_node(self) -> Optional[str]:
        """Возвращает ID стартовой ноды."""
        return self.graph.start_node
    
    def validate_maze(self) -> Tuple[bool, List[str]]:
        """
        Валидирует корректность графа.
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check start/goal exist
        if not self.graph.start_node:
            errors.append("No start node defined")
        if not self.graph.goal_node:
            errors.append("No goal node defined")
        
        # Check junction exists (for choice tasks)
        if not self.graph.junction_node:
            errors.append("No junction node defined")
        
        # Check connectivity
        if self.graph.start_node:
            if not self.graph.get_neighbors(self.graph.start_node):
                errors.append("Start node has no neighbors")
        
        if self.graph.goal_node:
            if not self.graph.get_neighbors(self.graph.goal_node):
                errors.append("Goal node has no neighbors")
        
        # Check exposure profiles are valid
        for node in self.graph.nodes.values():
            for key in ['X_risk', 'X_opp', 'D_est']:
                val = node.exposure_profile.get(key, 0.0)
                if not 0.0 <= val <= 1.0:
                    errors.append(f"Node {node.node_id}: {key}={val} out of [0,1]")
        
        return len(errors) == 0, errors


# =============================================================================
# TESTS
# =============================================================================

def test_maze_builder_basic():
    """
    Test: Basic maze creation.
    """
    builder = MazeBuilder()
    maze = builder.create_open_covered_maze()
    
    # Check nodes exist
    assert "start" in maze.nodes
    assert "junction" in maze.nodes
    assert "open_mid" in maze.nodes
    assert "covered_mid" in maze.nodes
    assert "goal" in maze.nodes
    
    # Check junction is choice point
    junction_neighbors = maze.get_neighbors("junction")
    assert len(junction_neighbors) == 3  # start, open_mid, covered_mid
    
    # Check exposure difference
    open_exposure = maze.nodes["open_mid"].exposure_profile['X_risk']
    covered_exposure = maze.nodes["covered_mid"].exposure_profile['X_risk']
    assert open_exposure > covered_exposure, "Open path should have higher exposure"
    
    # Validate maze
    is_valid, errors = builder.validate_maze()
    assert is_valid, f"Maze validation failed: {errors}"
    
    print("✓ PASS: Basic maze creation")
    return True


def test_threat_conflict_maze():
    """
    Test: Threat/reward conflict maze.
    """
    builder = MazeBuilder()
    maze = builder.create_threat_conflict_maze(
        predator_cue=True,
        reward_premium_open=0.5
    )
    
    # Check predator cue increases risk
    open_risk = maze.nodes["open_mid"].exposure_profile['X_risk']
    assert open_risk > 0.7, "Predator cue should increase risk"
    
    # Check reward premium
    goal_nu = maze.nodes["goal"].valence_profile['nu']
    assert goal_nu > 1.0, "Reward premium should increase nu"
    
    print("✓ PASS: Threat conflict maze")
    return True


def test_exposure_retrieval():
    """
    Test: Exposure profile retrieval.
    """
    builder = MazeBuilder()
    maze = builder.create_open_covered_maze(
        open_exposure=0.8,
        covered_exposure=0.2
    )
    
    # Get exposure at node
    exposure = builder.get_exposure_at_node("open_mid")
    assert exposure['X_risk'] == 0.8
    assert exposure['X_opp'] == 0.5
    assert exposure['D_est'] == 0.9
    
    # Get exposure on edge
    edge_exposure = builder.get_exposure_on_edge("junction_open")
    assert edge_exposure['X_risk'] == 0.8
    
    print("✓ PASS: Exposure retrieval")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("Stage 3.1: Maze Builder — Unit Tests")
    print("=" * 70)
    
    test_maze_builder_basic()
    test_threat_conflict_maze()
    test_exposure_retrieval()
    
    print("=" * 70)
    print("All tests passed!")
    print("=" * 70)