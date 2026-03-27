"""
Stage 3.0: Gate Input Definitions.

Определяет три dataclass-слоя входа для Gate:
1. InstantDiagnostics — мгновенные диагностические переменные (Stage 2 наследие)
2. ExposureAggregates — агрегированные кривизны exposure поля
3. TemporalState — сжатая временная история (3 core trace)

Design Constraints:
- No Ready Semions at Port: сенсорный вход не содержит категориальных меток
- Ontology ≠ Engineering Interface: ν, O, X — аналитическое разложение для трассировки
- One-Shot как режим обновления (реализуется в temporal_state.py)

Примечание:
Этот файл содержит ТОЛЬКО интерфейсные определения (dataclass contract).
Вся логика (update, computation, routing) вынесена в отдельные модули:
- temporal_state.py — обновление h_risk, h_opp, h_time
- exposure_field.py — вычисление X_risk, X_opp, D_est
- gate_stage3.py — threshold cascade routing
- gate_modes.py — определение режимов GateMode

Author: Alex Snow (Aleksey L. Snigirov)
License: MIT
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# =============================================================================
# LAYER 1: INSTANT DIAGNOSTICS (u_t)
# =============================================================================

@dataclass
class InstantDiagnostics:
    """
    Мгновенные диагностические переменные (u_t).
    
    Минимальный набор, наследуемый из Stage 2 для backward compatibility.
    Все поля — скаляры, вычисляемые inter-trially.
    
    Design Constraint: No categorical object labels.
    u_t содержит только агрегированные метрики (PE, entropy, volatility),
    не семантические метки ("snake", "stick", "food").
    
    Attributes:
        u_delta: Unsigned prediction error |r - Q(s,a)|
        u_entropy: Policy entropy -Σ π(a) log π(a)
        u_volatility: EMA(u_delta, α=0.3) — volatility proxy
    """
    
    u_delta: float = 0.0
    u_entropy: float = 0.0
    u_volatility: float = 0.0
    
    # Metadata для логирования (не передается в Gate routing)
    trial: int = 0
    
    def __post_init__(self):
        """Валидация: все поля должны быть numeric (no strings, no categories)."""
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, str):
                raise ValueError(
                    f"InstantDiagnostics.{field_name} cannot be string "
                    f"(No Ready Semions constraint violated): {field_value}"
                )
            if field_name == 'trial':
                continue  # trial может быть int
            if not isinstance(field_value, (int, float, np.number)):
                raise ValueError(
                    f"InstantDiagnostics.{field_name} must be numeric: {field_value}"
                )
    
    @classmethod
    def zeros(cls) -> 'InstantDiagnostics':
        """Возвращает нулевой вектор (для инициализации / backward compatibility)."""
        return cls()
    
    def validate_no_semions(self) -> bool:
        """
        Проверяет что вход не содержит готовых семионов.
        
        Returns:
            True если валидно (нет строк/категорий кроме trial)
        """
        for field_name, field_value in self.__dict__.items():
            if field_name == 'trial':
                continue
            if isinstance(field_value, str):
                return False
        return True


# =============================================================================
# LAYER 2: EXPOSURE AGGREGATES (from exposure_field.py)
# =============================================================================

@dataclass
class ExposureAggregates:
    """
    Агрегированные кривизны exposure поля.
    
    Design Principle: Ontology ≠ Engineering Interface.
    
    Для агента: единое prospective field ожидаемого изменения энтропии.
    Для исследователя: аналитическое разложение на ν, O, X для трассировки.
    
    Формула (аналитическая, не онтологическая):
        X = O · ν
    где:
        ν = валентность (positive/negative)
        O = детектируемость/наблюдаемость
        X = integrated exposure
    
    Attributes:
        X_risk: Current risky exposure (агрегат из exposure_field.py)
        X_opp: Current opportunity exposure (агрегат из exposure_field.py)
        D_est: Detectability / visibility estimate (агрегат из exposure_field.py)
    
    Note:
        Вычисление ν, O, X происходит в exposure_field.py.
        Этот dataclass только получает готовые агрегаты.
    """
    
    X_risk: float = 0.0
    X_opp: float = 0.0
    D_est: float = 0.0
    
    # Metadata для логирования (не передается в Gate routing)
    region_id: Optional[str] = None
    trial: int = 0
    
    def __post_init__(self):
        """Валидация: aggregates должны быть numeric."""
        aggregate_fields = ['X_risk', 'X_opp', 'D_est']
        for field_name in aggregate_fields:
            value = getattr(self, field_name)
            if not isinstance(value, (int, float, np.number)):
                raise ValueError(
                    f"ExposureAggregates.{field_name} must be numeric: {value}"
                )
    
    @classmethod
    def zeros(cls) -> 'ExposureAggregates':
        """Возвращает нулевые агрегаты (для backward compatibility)."""
        return cls()
    
    def validate_no_semions(self) -> bool:
        """
        Проверяет что aggregates не содержат категориальных меток.
        
        Returns:
            True если валидно (region_id может быть str для логирования)
        """
        for field_name, field_value in self.__dict__.items():
            if field_name == 'region_id':
                continue  # region_id — исключение (для логирования, не для агента)
            if isinstance(field_value, str):
                return False
        return True


# =============================================================================
# LAYER 3: TEMPORAL STATE (h_t)
# =============================================================================

@dataclass
class TemporalState:
    """
    Сжатая временная история (h_t).
    
    Минимальный набор для Stage 3.0 core (3 trace):
    - h_risk: exponentially smoothed risky exposure trace
    - h_opp: exponentially smoothed opportunity trace
    - h_time: time trace since last high-salience event
    
    Design Principle: One-shot = amplitude-dependent update regime.
    Реализация обновления — в temporal_state.py (не здесь!).
    
    Attributes:
        h_risk: Exponentially smoothed risky exposure trace
        h_opp: Exponentially smoothed opportunity trace
        h_time: Time steps since last high-salience event
    
    Note:
        Update логика вынесена в temporal_state.py.
        Этот dataclass только хранит состояние.
    """
    
    h_risk: float = 0.0
    h_opp: float = 0.0
    h_time: int = 0
    
    # Debug metadata (не часть core state, для логирования)
    one_shot_pending: bool = False
    one_shot_amplitude: float = 0.0
    
    def __post_init__(self):
        """Валидация: trace должны быть numeric."""
        if not isinstance(self.h_risk, (int, float, np.number)):
            raise ValueError(f"TemporalState.h_risk must be numeric: {self.h_risk}")
        if not isinstance(self.h_opp, (int, float, np.number)):
            raise ValueError(f"TemporalState.h_opp must be numeric: {self.h_opp}")
        if not isinstance(self.h_time, (int, np.integer)):
            raise ValueError(f"TemporalState.h_time must be int: {self.h_time}")
    
    @classmethod
    def zeros(cls) -> 'TemporalState':
        """Возвращает нулевое состояние (для инициализации / backward compatibility)."""
        return cls()


# =============================================================================
# COMBINED INPUT (для Gate)
# =============================================================================

@dataclass
class GateInput:
    """
    Комбинированный вход для Gate.
    
    Объединяет три слоя в один интерфейс для Gate routing.
    Gate читает именованные поля из трех dataclass-слоев.
    
    Note:
        No to_combined_array() — Gate не требует объединенного массива.
        Gate читает именованные поля через threshold cascade.
    """
    
    instant: InstantDiagnostics = field(default_factory=InstantDiagnostics)
    exposure: ExposureAggregates = field(default_factory=ExposureAggregates)
    temporal: TemporalState = field(default_factory=TemporalState)
    
    def __post_init__(self):
        """Валидация всех трех слоев."""
        self.instant.__post_init__()
        self.exposure.__post_init__()
        self.temporal.__post_init__()
    
    def validate_all_layers(self) -> tuple:
        """
        Валидирует все три слоя на отсутствие семионов.
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        if not self.instant.validate_no_semions():
            errors.append("InstantDiagnostics contains semions (string fields)")
        
        if not self.exposure.validate_no_semions():
            errors.append("ExposureAggregates contains semions (string fields)")
        
        # TemporalState не имеет string полей (кроме debug metadata)
        
        return len(errors) == 0, errors
    
    @classmethod
    def zeros(cls) -> 'GateInput':
        """
        Возвращает нулевой вход (для backward compatibility с Stage 2).
        
        При нулевых exposure и temporal trace, Gate должен деградировать
        в Stage 2 behavior (EXPLOIT/EXPLORE только).
        """
        return cls(
            instant=InstantDiagnostics.zeros(),
            exposure=ExposureAggregates.zeros(),
            temporal=TemporalState.zeros()
        )


# =============================================================================
# CONVENIENCE FUNCTIONS (для тестирования, не для production)
# =============================================================================

def create_test_input(
    u_delta: float = 0.5,
    u_entropy: float = 0.3,
    u_volatility: float = 0.2,
    X_risk: float = 0.1,
    X_opp: float = 0.1,
    D_est: float = 0.5,
    h_risk: float = 0.0,
    h_opp: float = 0.0,
    h_time: int = 0
) -> GateInput:
    """
    Создает тестовый GateInput с заданными параметрами.
    
    Только для тестов! Не использовать в production code.
    
    Args:
        u_delta: Unsigned prediction error
        u_entropy: Policy entropy
        u_volatility: Volatility proxy
        X_risk: Risky exposure
        X_opp: Opportunity exposure
        D_est: Detectability estimate
        h_risk: Risk trace
        h_opp: Opportunity trace
        h_time: Time since last salient event
    
    Returns:
        GateInput с заданными параметрами
    """
    return GateInput(
        instant=InstantDiagnostics(
            u_delta=u_delta,
            u_entropy=u_entropy,
            u_volatility=u_volatility
        ),
        exposure=ExposureAggregates(
            X_risk=X_risk,
            X_opp=X_opp,
            D_est=D_est
        ),
        temporal=TemporalState(
            h_risk=h_risk,
            h_opp=h_opp,
            h_time=h_time
        )
    )