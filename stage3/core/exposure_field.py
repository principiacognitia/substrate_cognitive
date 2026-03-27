"""
Stage 3.0: Exposure Field Computation.

Вычисляет агрегированные кривизны exposure поля из raw observation.

Design Constraints:
- No Ready Semions at Port: сенсорный вход не содержит категориальных меток
- Ontology ≠ Engineering Interface: ν, O, X — аналитическое разложение для трассировки
- For the agent: единое prospective field ожидаемого изменения энтропии

Примечание:
Этот файл содержит ТОЛЬКО вычисление exposure aggregates.
Вся логика Gate routing находится в gate_stage3.py.

Author: Alex Snow (Aleksey L. Snigirov)
License: MIT
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union
import numpy as np


# =============================================================================
# DATA CLASS: ExposureAggregates (только для чтения Gate)
# =============================================================================

@dataclass
class ExposureAggregates:
    """
    Агрегированные кривизны exposure поля (видит Gate).
    
    Design Principle: Ontology ≠ Engineering Interface.
    
    Для агента: единое prospective field ожидаемого изменения энтропии.
    Для исследователя: аналитическое разложение на ν, O, X для трассировки.
    
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
        """Возвращает нулевые агрегаты (для инициализации / backward compatibility)."""
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
# EXPOSURE FIELD COMPUTATION
# =============================================================================

class ExposureField:
    """
    Вычисляет exposure aggregates из raw observation.
    
    Design Constraint: No Ready Semions at Port.
    
    Sensory preprocessing yields valence-/exposure-weighted embeddings or aggregates,
    not pre-classified object semions. A semion is not an input label delivered by
    the port; it is a downstream stabilization/compression in the S-O-R cycle.
    
    For the agent, valence and observability are one prospective field of expected
    future entropy change. For implementation, logging, ablation, and tests, the
    field may be analytically decomposed as:
        X = O · ν
    where:
        ν = valence-weighted signal profile
        O = detectability / observability
        X = integrated exposure
    
    This decomposition is engineering-facing, not an ontological claim about
    separate primitives inside the agent.
    """
    
    def __init__(
        self,
        valence_scale: float = 1.0,
        observability_scale: float = 1.0,
        risk_threshold: float = 0.5,
        opportunity_threshold: float = 0.5
    ):
        """
        Инициализирует exposure field.
        
        Args:
            valence_scale: Масштаб для валентности [-1, 1]
            observability_scale: Масштаб для наблюдаемости [0, 1]
            risk_threshold: Порог для X_risk агрегации
            opportunity_threshold: Порог для X_opp агрегации
        """
        self.valence_scale = valence_scale
        self.observability_scale = observability_scale
        self.risk_threshold = risk_threshold
        self.opportunity_threshold = opportunity_threshold
    
    def compute_exposure(
        self,
        observation: Dict,
        action: Optional[int] = None,
        reward: Optional[float] = None,
        region_id: Optional[str] = None,
        trial: int = 0
    ) -> ExposureAggregates:
        """
        Вычисляет exposure aggregates из raw observation.
        
        Design Constraint: No Ready Semions at Port.
        
        The sensory interface may only return:
        - valence-/exposure-weighted embeddings
        - aggregated field quantities
        - compressed traces
        
        NOT: pre-classified object labels ("snake", "stick", "food", "predator").
        
        Args:
            observation: Raw observation dict (no categorical labels!)
            action: Optional action for detectability computation
            reward: Optional reward for valence computation
            region_id: Optional region ID for logging (not for agent!)
            trial: Trial number for logging
        
        Returns:
            ExposureAggregates with X_risk, X_opp, D_est
        
        Raises:
            ValueError: If observation contains categorical string labels
        """
        # Валидация: observation не должен содержать строковых меток
        self._validate_no_semions(observation)
        
        # Вычисляем ν (valence) из reward или observation features
        nu = self._compute_valence(observation, reward)
        
        # Вычисляем O (observability) из observation features
        O = self._compute_observability(observation, action)
        
        # Вычисляем X = O · ν (аналитическое разложение)
        X = O * nu
        
        # Агрегируем для Gate (только numeric aggregates)
        X_risk = max(0.0, -X) if nu < 0 else 0.0
        X_opp = max(0.0, X) if nu > 0 else 0.0
        D_est = O
        
        return ExposureAggregates(
            X_risk=X_risk,
            X_opp=X_opp,
            D_est=D_est,
            region_id=region_id,  # Только для логирования!
            trial=trial
        )
    
    def _compute_valence(
        self,
        observation: Dict,
        reward: Optional[float] = None
    ) -> float:
        """
        Вычисляет валентность ν из observation/reward.
        
        Design Principle: ν is not a categorical label, but a continuous
        valence-weighted signal profile shaped by selection on consequences.
        
        Args:
            observation: Raw observation dict
            reward: Optional reward signal
        
        Returns:
            Valence in range [-1, 1]
        """
        if reward is not None:
            # Нормализуем reward в [-1, 1]
            nu = np.clip(reward * self.valence_scale, -1.0, 1.0)
        else:
            # Вычисляем из observation features (без категориальных меток!)
            # Пример: используем numeric features только
            nu = 0.0
            
            # Если есть numeric features в observation
            for key, value in observation.items():
                if isinstance(value, (int, float, np.number)):
                    # Нормализуем и агрегируем
                    nu += float(value)
            
            # Нормализуем в [-1, 1]
            nu = np.tanh(nu * self.valence_scale)
        
        return float(nu)
    
    def _compute_observability(
        self,
        observation: Dict,
        action: Optional[int] = None
    ) -> float:
        """
        Вычисляет наблюдаемость O из observation/action.
        
        Design Principle: O is not a categorical label, but a continuous
        detectability signal profile.
        
        Args:
            observation: Raw observation dict
            action: Optional action for detectability computation
        
        Returns:
            Observability in range [0, 1]
        """
        # Вычисляем из observation features (без категориальных меток!)
        O = 0.0
        
        # Если есть numeric features в observation
        for key, value in observation.items():
            if isinstance(value, (int, float, np.number)):
                # Агрегируем magnitude features
                O += abs(float(value))
        
        # Нормализуем в [0, 1] через sigmoid
        O = float(1.0 / (1.0 + np.exp(-O * self.observability_scale)))
        
        return O
    
    def _validate_no_semions(self, observation: Dict) -> None:
        """
        Проверяет что observation не содержит готовых семионов.
        
        Design Constraint: No Ready Semions at Port.
        
        Args:
            observation: Raw observation dict
        
        Raises:
            ValueError: Если observation содержит строковые метки
        """
        for key, value in observation.items():
            # Ключи могут быть строками (это нормально)
            if isinstance(value, str):
                # Значения не должны быть строками (категориальные метки!)
                raise ValueError(
                    f"Observation value '{key}' contains string label '{value}'. "
                    f"No Ready Semions constraint violated: sensory input must not "
                    f"deliver pre-classified object labels."
                )
    
    def update_aggregates(
        self,
        prev_aggregates: ExposureAggregates,
        new_aggregates: ExposureAggregates,
        decay_rate: float = 0.9
    ) -> ExposureAggregates:
        """
        Обновляет aggregates с экспоненциальным затуханием.
        
        Args:
            prev_aggregates: Предыдущие aggregates
            new_aggregates: Новые aggregates
            decay_rate: Коэффициент затухания
        
        Returns:
            Updated ExposureAggregates
        """
        X_risk = decay_rate * prev_aggregates.X_risk + (1 - decay_rate) * new_aggregates.X_risk
        X_opp = decay_rate * prev_aggregates.X_opp + (1 - decay_rate) * new_aggregates.X_opp
        D_est = decay_rate * prev_aggregates.D_est + (1 - decay_rate) * new_aggregates.D_est
        
        return ExposureAggregates(
            X_risk=X_risk,
            X_opp=X_opp,
            D_est=D_est,
            region_id=new_aggregates.region_id,
            trial=new_aggregates.trial
        )


# =============================================================================
# CONVENIENCE FUNCTIONS (для тестирования)
# =============================================================================

def create_test_observation(
    reward: float = 0.5,
    noise: float = 0.1,
    visibility: float = 0.8
) -> Dict:
    """
    Создает тестовое observation без категориальных меток.
    
    Только для тестов! Не использовать в production code.
    
    Args:
        reward: Reward value
        noise: Noise level
        visibility: Visibility estimate
    
    Returns:
        Dict с numeric features только
    """
    return {
        'reward_signal': reward,
        'noise_level': noise,
        'visibility_estimate': visibility
    }


# =============================================================================
# TESTS (для быстрой проверки)
# =============================================================================

def test_no_ready_semions():
    """
    Test: No Ready Semions at Port.
    
    Проверяет что exposure field отвергает строковые метки.
    """
    field = ExposureField()
    
    # Это должно выбросить ошибку
    try:
        bad_observation = {'object_label': 'snake'}  # Строковая метка!
        field.compute_exposure(bad_observation)
        print("❌ FAIL: String label accepted in observation")
        return False
    except ValueError:
        print("✓ PASS: No Ready Semions constraint enforced")
        return True


def test_exposure_aggregates():
    """
    Test: Exposure Aggregates computation.
    
    Проверяет что aggregates вычисляются корректно.
    """
    field = ExposureField()
    
    # Тестовое observation без меток
    observation = create_test_observation(reward=0.5, noise=0.1, visibility=0.8)
    
    aggregates = field.compute_exposure(observation, reward=0.5, trial=1)
    
    # Проверяем что aggregates numeric
    assert isinstance(aggregates.X_risk, (int, float, np.number))
    assert isinstance(aggregates.X_opp, (int, float, np.number))
    assert isinstance(aggregates.D_est, (int, float, np.number))
    
    # Проверяем что нет строк
    assert aggregates.validate_no_semions()
    
    print("✓ PASS: Exposure Aggregates computation")
    return True


def test_valence_computation():
    """
    Test: Valence computation.
    
    Проверяет что валентность вычисляется в [-1, 1].
    """
    field = ExposureField()
    
    # Positive reward
    nu_pos = field._compute_valence({}, reward=1.0)
    assert -1.0 <= nu_pos <= 1.0
    assert nu_pos > 0
    
    # Negative reward
    nu_neg = field._compute_valence({}, reward=-1.0)
    assert -1.0 <= nu_neg <= 1.0
    assert nu_neg < 0
    
    # Zero reward
    nu_zero = field._compute_valence({}, reward=0.0)
    assert -1.0 <= nu_zero <= 1.0
    
    print("✓ PASS: Valence computation")
    return True


def test_observability_computation():
    """
    Test: Observability computation.
    
    Проверяет что наблюдаемость вычисляется в [0, 1].
    """
    field = ExposureField()
    
    observation = create_test_observation(reward=0.5, noise=0.1, visibility=0.8)
    O = field._compute_observability(observation)
    
    assert 0.0 <= O <= 1.0
    
    print("✓ PASS: Observability computation")
    return True


def test_action_basin_equivalence():
    """
    Test: Action-Basin Equivalence.
    
    Проверяет что два перцептивно разных стимула с близким X-profile
    дают схожие aggregates.
    """
    field = ExposureField()
    
    # Стимул 1: высокая награда, высокая видимость
    obs1 = create_test_observation(reward=0.8, noise=0.1, visibility=0.9)
    agg1 = field.compute_exposure(obs1, reward=0.8, trial=1)
    
    # Стимул 2: чуть другая награда, чуть другая видимость
    obs2 = create_test_observation(reward=0.75, noise=0.15, visibility=0.85)
    agg2 = field.compute_exposure(obs2, reward=0.75, trial=2)
    
    # Проверяем что aggregates близки (action basin equivalence)
    risk_diff = abs(agg1.X_risk - agg2.X_risk)
    opp_diff = abs(agg1.X_opp - agg2.X_opp)
    det_diff = abs(agg1.D_est - agg2.D_est)
    
    # Допустимое отклонение
    tolerance = 0.2
    
    assert risk_diff < tolerance, f"X_risk difference {risk_diff} > {tolerance}"
    assert opp_diff < tolerance, f"X_opp difference {opp_diff} > {tolerance}"
    assert det_diff < tolerance, f"D_est difference {det_diff} > {tolerance}"
    
    print("✓ PASS: Action-Basin Equivalence")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("Stage 3.0: Exposure Field — Unit Tests")
    print("=" * 70)
    
    test_no_ready_semions()
    test_exposure_aggregates()
    test_valence_computation()
    test_observability_computation()
    test_action_basin_equivalence()
    
    print("=" * 70)
    print("All tests completed!")
    print("=" * 70)