"""
Stage 3.0: Gate Input Definitions.

Определяет три dataclass-слоя входа для Gate:
1. InstantDiagnostics (u_t) — мгновенные диагностические переменные
2. TemporalState (h_t) — сжатая временная история
3. ExposureAggregates — агрегированные кривизны exposure поля

Принципы:
- No Ready Semions at Port: сенсорный вход не содержит категориальных меток
- Ontology ≠ Engineering Interface: ν, O, X — аналитическое разложение для трассировки
- One-Shot как режим обновления h_t (amplitude-dependent update)

Author: Alex Snow (Aleksey L. Snigirov)
License: MIT
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from enum import Enum


# =============================================================================
# ENUM: MODES (для трассировки, не для агента)
# =============================================================================

class GateMode(Enum):
    """
    Режимы Gate для Stage 3.0.
    
    Note: Это инженерная декомпозиция для логирования и абляций.
    Для агента это единое prospective field ожидаемого изменения энтропии.
    """
    EXPLOIT = "exploit"
    EXPLORE = "explore"
    EXPLOIT_SAFE = "exploit_safe"      # Avoidance-biased default при high threat
    ABSENCE_CHECK = "absence_check"    # Costly verification при high stakes + no evidence


# =============================================================================
# LAYER 1: INSTANT DIAGNOSTICS (u_t)
# =============================================================================

@dataclass
class InstantDiagnostics:
    """
    Мгновенные диагностические переменные (u_t).
    
    Это прямой наследник u_t из Stage 2, расширенный для Stage 3.0.
    Все поля — скаляры, вычисляемые inter-trially.
    
    Design Constraint: No categorical object labels.
    u_t содержит только агрегированные метрики (PE, entropy, volatility),
    не семантические метки ("snake", "stick").
    """
    
    # === Stage 2 Core (сохраняется для backward compatibility) ===
    u_delta: float = 0.0          # Unsigned prediction error |r - Q(s,a)|
    u_entropy: float = 0.0        # Policy entropy -Σ π(a) log π(a)
    u_volatility: float = 0.0     # EMA(u_delta, α=0.3)
    u_stakes: float = 0.0         # Stakes (currently fixed at 0)
    
    # === Stage 3.0 Extensions (derived from exposure_field) ===
    u_risk: float = 0.0           # Current risky exposure (from exposure_field)
    u_opportunity: float = 0.0    # Current opportunity exposure (from exposure_field)
    
    # === Metadata (для логирования, не для Gate scoring) ===
    trial: int = 0
    timestamp: float = 0.0
    
    def __post_init__(self):
        """Валидация: все поля должны быть numeric (no strings, no categories)."""
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, str):
                raise ValueError(
                    f"InstantDiagnostics.{field_name} cannot be string "
                    f"(No Ready Semions constraint violated): {field_value}"
                )
    
    def to_array(self) -> np.ndarray:
        """Конвертирует в numpy array для Gate scoring."""
        return np.array([
            self.u_delta,
            self.u_entropy,
            self.u_volatility,
            self.u_stakes,
            self.u_risk,
            self.u_opportunity
        ])
    
    @classmethod
    def zeros(cls) -> 'InstantDiagnostics':
        """Возвращает нулевой вектор (для инициализации)."""
        return cls()
    
    def validate_no_semions(self) -> bool:
        """
        Проверяет что вход не содержит готовых семионов.
        
        Returns:
            True если валидно (нет строк/категорий)
        """
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, (str, Enum)):
                return False
        return True


# =============================================================================
# LAYER 2: TEMPORAL STATE (h_t)
# =============================================================================

@dataclass
class TemporalState:
    """
    Сжатая временная история (h_t).
    
    Это отдельный слой (не расширение u_t), реализующий temporal smear / 
    compressed history / local jet memory.
    
    Design Principle: One-shot = amplitude-dependent update regime, не отдельный модуль.
    
    Fields:
        time_since_last_salient: Триалов с последнего значимого события
        cue_recency_trace: EMA trace недавних cue (decay=0.95)
        cue_rate_recent: Частота cue за последние N триалов
        safe_window_estimate: Оценка безопасного временного горизонта
        action_momentum: Инерция последних действий (EMA)
        valence_trace_neg: Trace негативной валентности (для avoidance)
        valence_trace_pos: Trace позитивной валентности (для approach)
        exposure_trace: Cumulative exposure trace
        one_shot_pending: Флаг high-amplitude update (не отдельный модуль!)
    """
    
    # === Temporal Integration ===
    time_since_last_salient: int = 0
    cue_recency_trace: float = 0.0
    cue_rate_recent: float = 0.0
    safe_window_estimate: float = 100.0  # Триалов
    action_momentum: float = 0.0
    
    # === Valence Traces (analytical decomposition for logging) ===
    valence_trace_neg: float = 0.0
    valence_trace_pos: float = 0.0
    exposure_trace: float = 0.0
    
    # === One-Shot Update Regime (не отдельный модуль!) ===
    one_shot_pending: bool = False
    one_shot_amplitude: float = 0.0
    
    # === Metadata ===
    last_update_trial: int = 0
    
    def __post_init__(self):
        """Валидация: все trace должны быть numeric."""
        trace_fields = [
            'cue_recency_trace', 'cue_rate_recent', 'safe_window_estimate',
            'action_momentum', 'valence_trace_neg', 'valence_trace_pos',
            'exposure_trace', 'one_shot_amplitude'
        ]
        for field_name in trace_fields:
            value = getattr(self, field_name)
            if not isinstance(value, (int, float, np.number)):
                raise ValueError(
                    f"TemporalState.{field_name} must be numeric: {value}"
                )
    
    def update(
        self,
        observation: Dict,
        action: int,
        reward: float,
        salience: float,
        stakes: float,
        exposure: float,
        one_shot_threshold: float = 5.0,
        decay_rate: float = 0.95
    ) -> 'TemporalState':
        """
        Обновляет temporal state.
        
        One-shot реализован как high-amplitude update regime:
        Если salience * stakes > threshold → one_shot_pending = True
        
        Args:
            observation: Словарь с raw observation (не категоризировано!)
            action: Выбранное действие
            reward: Полученная награда
            salience: Салиентность события (unsigned PE или novelty)
            stakes: Значимость ситуации
            exposure: Текущая exposure
            one_shot_threshold: Порог для one-shot update
            decay_rate: Коэффициент затухания trace
        
        Returns:
            Обновлённый TemporalState
        """
        # === One-Shot Detection (amplitude-dependent regime) ===
        surprise_amplitude = salience * stakes
        self.one_shot_pending = surprise_amplitude > one_shot_threshold
        self.one_shot_amplitude = surprise_amplitude
        
        # === Temporal Integration ===
        self.time_since_last_salient += 1
        
        if salience > 0.5:  # Salient event
            self.cue_recency_trace = 1.0
            self.time_since_last_salient = 0
        else:
            self.cue_recency_trace *= decay_rate
        
        # Cue rate (скользящее среднее за 10 триалов)
        self.cue_rate_recent = (
            0.9 * self.cue_rate_recent + 
            0.1 * (1.0 if salience > 0.5 else 0.0)
        )
        
        # Action momentum (EMA действий)
        self.action_momentum = 0.9 * self.action_momentum + 0.1 * action
        
        # === Valence Traces (analytical decomposition) ===
        # Note: Для агента это единое поле, но для исследователя — раздельно
        if reward < 0:
            self.valence_trace_neg = 0.9 * self.valence_trace_neg + 0.1 * abs(reward)
            self.valence_trace_pos *= 0.9
        elif reward > 0:
            self.valence_trace_pos = 0.9 * self.valence_trace_pos + 0.1 * reward
            self.valence_trace_neg *= 0.9
        
        # Exposure trace
        self.exposure_trace = 0.95 * self.exposure_trace + 0.05 * exposure
        
        # === Safe Window Estimate ===
        # Если exposure низкий → safe_window растёт
        if exposure < 0.2:
            self.safe_window_estimate = min(200, self.safe_window_estimate + 5)
        else:
            self.safe_window_estimate = max(10, self.safe_window_estimate - 10)
        
        self.last_update_trial += 1
        
        return self
    
    def to_array(self) -> np.ndarray:
        """Конвертирует в numpy array для Gate scoring."""
        return np.array([
            self.time_since_last_salient / 100.0,  # Нормализация
            self.cue_recency_trace,
            self.cue_rate_recent,
            self.safe_window_estimate / 200.0,
            self.action_momentum / 10.0,
            self.valence_trace_neg,
            self.valence_trace_pos,
            self.exposure_trace
        ])
    
    @classmethod
    def zeros(cls) -> 'TemporalState':
        """Возвращает нулевое состояние."""
        return cls()
    
    def validate_action_basin_equivalence(
        self,
        other: 'TemporalState',
        tolerance: float = 0.1
    ) -> bool:
        """
        Action-Basin Equivalence Test.
        
        Проверяет что два temporal state с близкими exposure профилями
        дают схожий mode bias.
        
        Args:
            other: Другой TemporalState для сравнения
            tolerance: Допустимое отклонение
        
        Returns:
            True если state эквивалентны для action selection
        """
        self_arr = self.to_array()
        other_arr = other.to_array()
        
        # Нормализованная L2 distance
        distance = np.linalg.norm(self_arr - other_arr) / np.sqrt(len(self_arr))
        
        return distance < tolerance


# =============================================================================
# LAYER 3: EXPOSURE AGGREGATES
# =============================================================================

@dataclass
class ExposureAggregates:
    """
    Агрегированные кривизны exposure поля.
    
    Design Principle: Ontology ≠ Engineering Interface.
    
    Для агента: единое prospective field ожидаемого изменения энтропии.
    Для исследователя: аналитическое разложение на ν, O, X для трассировки.
    
    Формула (аналитическая, не онтологическая):
        X_{r,m}(t,a) = O_{r,m}(t,a) · ν_{r,m}(t)
    
    где:
        ν = валентность (positive/negative)
        O = детектируемость/наблюдаемость
        X = integrated exposure
    """
    
    # === Analytical Decomposition (для логирования/абляций) ===
    valence_raw: float = 0.0          # ν: raw valence signal
    observability_raw: float = 0.0    # O: raw observability signal
    exposure_integrated: float = 0.0  # X = O · ν
    
    # === Exported Aggregates (видит Gate) ===
    risk_agg: float = 0.0             # Aggregated negative exposure
    opportunity_agg: float = 0.0      # Aggregated positive exposure
    detectability_agg: float = 0.0    # Agent visibility estimate
    safe_window_estimate: float = 100.0  # Temporal safety horizon
    diagnosticity_estimate: float = 0.0  # Information gain potential
    
    # === Metadata ===
    region_id: Optional[str] = None   # Для трассировки (не для агента!)
    trial: int = 0
    
    def __post_init__(self):
        """Валидация: aggregates должны быть numeric."""
        aggregate_fields = [
            'risk_agg', 'opportunity_agg', 'detectability_agg',
            'safe_window_estimate', 'diagnosticity_estimate',
            'valence_raw', 'observability_raw', 'exposure_integrated'
        ]
        for field_name in aggregate_fields:
            value = getattr(self, field_name)
            if not isinstance(value, (int, float, np.number)):
                raise ValueError(
                    f"ExposureAggregates.{field_name} must be numeric: {value}"
                )
    
    @classmethod
    def from_valence_observability(
        cls,
        valence: float,
        observability: float,
        region_id: Optional[str] = None,
        trial: int = 0
    ) -> 'ExposureAggregates':
        """
        Создаёт aggregates из valence/observability.
        
        Note: Это инженерный интерфейс для исследователя.
        Агент получает уже агрегированные risk_agg/opportunity_agg.
        
        Args:
            valence: Валентность [-1, 1]
            observability: Наблюдаемость [0, 1]
            region_id: ID региона (для логирования)
            trial: Номер триала
        
        Returns:
            ExposureAggregates
        """
        # X = O · ν (аналитическое разложение)
        exposure = observability * valence
        
        # Агрегация для Gate
        risk_agg = max(0, -exposure) if valence < 0 else 0.0
        opportunity_agg = max(0, exposure) if valence > 0 else 0.0
        detectability_agg = observability
        
        return cls(
            valence_raw=valence,
            observability_raw=observability,
            exposure_integrated=exposure,
            risk_agg=risk_agg,
            opportunity_agg=opportunity_agg,
            detectability_agg=detectability_agg,
            safe_window_estimate=100.0,  # Будет обновлено из TemporalState
            diagnosticity_estimate=abs(exposure),
            region_id=region_id,
            trial=trial
        )
    
    def to_array(self) -> np.ndarray:
        """Конвертирует в numpy array для Gate scoring."""
        return np.array([
            self.risk_agg,
            self.opportunity_agg,
            self.detectability_agg,
            self.safe_window_estimate / 100.0,
            self.diagnosticity_estimate
        ])
    
    @classmethod
    def zeros(cls) -> 'ExposureAggregates':
        """Возвращает нулевые агрегаты."""
        return cls()
    
    def validate_no_semions(self) -> bool:
        """
        Проверяет что aggregates не содержат категориальных меток.
        
        Returns:
            True если валидно (region_id может быть str для логирования)
        """
        # region_id — исключение (для логирования, не для агента)
        for field_name, field_value in self.__dict__.items():
            if field_name == 'region_id':
                continue
            if isinstance(field_value, str):
                return False
        return True


# =============================================================================
# COMBINED INPUT (для Gate)
# =============================================================================

@dataclass
class GateInput:
    """
    Комбинированный вход для Gate.
    
    Объединяет три слоя в один интерфейс для Gate scoring.
    """
    
    instant: InstantDiagnostics = field(default_factory=InstantDiagnostics)
    temporal: TemporalState = field(default_factory=TemporalState)
    exposure: ExposureAggregates = field(default_factory=ExposureAggregates)
    
    def to_combined_array(self) -> np.ndarray:
        """
        Конвертирует все три слоя в один array для Gate scoring.
        
        Returns:
            numpy array [u_t, h_t, exposure]
        """
        return np.concatenate([
            self.instant.to_array(),
            self.temporal.to_array(),
            self.exposure.to_array()
        ])
    
    def validate_all_layers(self) -> Tuple[bool, List[str]]:
        """
        Валидирует все три слоя.
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        if not self.instant.validate_no_semions():
            errors.append("InstantDiagnostics contains semions")
        
        if not self.exposure.validate_no_semions():
            errors.append("ExposureAggregates contains semions")
        
        # TemporalState валидируется в __post_init__
        
        return len(errors) == 0, errors
    
    @classmethod
    def zeros(cls) -> 'GateInput':
        """Возвращает нулевой вход."""
        return cls(
            instant=InstantDiagnostics.zeros(),
            temporal=TemporalState.zeros(),
            exposure=ExposureAggregates.zeros()
        )


# =============================================================================
# TESTS (для быстрой проверки)
# =============================================================================

def test_no_ready_semions():
    """
    Test: No Ready Semions at Port.
    
    Проверяет что InstantDiagnostics не принимает строки.
    """
    try:
        # Это должно выбросить ошибку
        bad_input = InstantDiagnostics(u_delta="snake")  # type: ignore
        print("❌ FAIL: String accepted in InstantDiagnostics")
        return False
    except ValueError:
        print("✓ PASS: No Ready Semions constraint enforced")
        return True


def test_action_basin_equivalence():
    """
    Test: Action-Basin Equivalence.
    
    Проверяет что два state с близкими exposure дают схожий bias.
    """
    state1 = TemporalState(
        cue_recency_trace=0.8,
        valence_trace_neg=0.2,
        exposure_trace=0.3
    )
    
    state2 = TemporalState(
        cue_recency_trace=0.82,  # Близко к state1
        valence_trace_neg=0.18,
        exposure_trace=0.32
    )
    
    if state1.validate_action_basin_equivalence(state2, tolerance=0.1):
        print("✓ PASS: Action-Basin Equivalence test")
        return True
    else:
        print("❌ FAIL: Action-Basin Equivalence test")
        return False


def test_one_shot_as_update_regime():
    """
    Test: One-Shot as Amplitude-Dependent Update Regime.
    
    Проверяет что one-shot — это не отдельный модуль, а флаг.
    """
    state = TemporalState.zeros()
    
    # Low amplitude (many-shot)
    state.update(
        observation={},
        action=0,
        reward=0.1,
        salience=0.2,
        stakes=0.5,
        exposure=0.1,
        one_shot_threshold=5.0
    )
    
    assert not state.one_shot_pending, "Low amplitude should not trigger one-shot"
    
    # High amplitude (one-shot)
    state.update(
        observation={},
        action=0,
        reward=5.0,
        salience=0.9,
        stakes=10.0,
        exposure=0.1,
        one_shot_threshold=5.0
    )
    
    assert state.one_shot_pending, "High amplitude should trigger one-shot"
    assert state.one_shot_amplitude > 5.0, "Amplitude should be recorded"
    
    print("✓ PASS: One-Shot as Update Regime")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("Stage 3.0: Gate Inputs — Unit Tests")
    print("=" * 70)
    
    test_no_ready_semions()
    test_action_basin_equivalence()
    test_one_shot_as_update_regime()
    
    print("=" * 70)
    print("All tests completed!")
    print("=" * 70)