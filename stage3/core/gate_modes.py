"""
Stage 3.0: Gate Mode Definitions.

Определяет доступные режимы Gate для Stage 3.0.

Design Constraints:
- One Gate only: все режимы обрабатываются единым Gate
- No argmax arbitration: режимы выбираются через threshold cascade (в gate_stage3.py)
- No ready semions: режимы не являются категориальными метками объектов

Примечание:
Этот файл содержит ТОЛЬКО определения режимов (enum).
Вся логика выбора режима (threshold cascade) находится в gate_stage3.py.

Author: Alex Snow (Aleksey L. Snigirov)
License: MIT
"""

from enum import Enum


class GateMode(Enum):
    """
    Режимы Gate для Stage 3.0.
    
    Design Principle: Ontology ≠ Engineering Interface.
    
    Для исследователя: это аналитическое разделение для логирования и абляций.
    Для агента: это единый prospective field ожидаемого изменения энтропии,
    где разные режимы — это различные точки на континууме response basins.
    
    Modes:
        EXPLOIT: Standard exploitation of learned values (MF-dominated).
                 Default mode when uncertainty is low and exposure is acceptable.
        
        EXPLORE: Directed exploration for information gain (MB-dominated).
                 Activated when uncertainty is high but negative exposure is not blocking.
        
        EXPLOIT_SAFE: Avoidance-biased exploitation under high threat.
                      Not a new "emotional module" — safe default when risk is critical.
                      Bypasses standard explore barrier via threat override.
        
        ABSENCE_CHECK: Costly verification when stakes are high and evidence is absent.
                       Requires: high stakes + poor visibility + sufficient safe window.
    
    Note:
        Mode selection is implemented as threshold cascade in gate_stage3.py,
        not as global argmax scoring. Stronger signals (threat) override weaker ones.
    """
    
    EXPLOIT = "exploit"
    EXPLORE = "explore"
    EXPLOIT_SAFE = "exploit_safe"
    ABSENCE_CHECK = "absence_check"
    
    @classmethod
    def default(cls) -> 'GateMode':
        """
        Возвращает режим по умолчанию (для инициализации).
        
        Returns:
            GateMode.EXPLOIT
        """
        return cls.EXPLOIT
    
    @classmethod
    def is_safe_mode(cls, mode: 'GateMode') -> bool:
        """
        Проверяет является ли режим "безопасным" (avoidance-biased).
        
        Args:
            mode: Режим для проверки
        
        Returns:
            True если EXPLOIT_SAFE или ABSENCE_CHECK
        """
        return mode in (cls.EXPLOIT_SAFE, cls.ABSENCE_CHECK)
    
    @classmethod
    def is_exploratory(cls, mode: 'GateMode') -> bool:
        """
        Проверяет является ли режим исследовательским.
        
        Args:
            mode: Режим для проверки
        
        Returns:
            True если EXPLORE или ABSENCE_CHECK
        """
        return mode in (cls.EXPLORE, cls.ABSENCE_CHECK)
    
    def __str__(self) -> str:
        """
        Возвращает строковое представление режима.
        
        Returns:
            Значение enum (например, "exploit")
        """
        return self.value


# =============================================================================
# CONSTANTS (для быстрого доступа без импорта enum)
# =============================================================================

MODE_EXPLOIT = GateMode.EXPLOIT
MODE_EXPLORE = GateMode.EXPLORE
MODE_EXPLOIT_SAFE = GateMode.EXPLOIT_SAFE
MODE_ABSENCE_CHECK = GateMode.ABSENCE_CHECK

# Все доступные режимы (для итерации в тестах)
ALL_MODES = [
    GateMode.EXPLOIT,
    GateMode.EXPLORE,
    GateMode.EXPLOIT_SAFE,
    GateMode.ABSENCE_CHECK
]

# Режимы по приоритету override (для threshold cascade)
# Более сильные сигналы (threat) переопределяют более слабые
MODE_PRIORITY = {
    GateMode.ABSENCE_CHECK: 4,    # Highest priority (high stakes + no evidence)
    GateMode.EXPLOIT_SAFE: 3,     # High priority (critical threat)
    GateMode.EXPLORE: 2,          # Medium priority (high uncertainty)
    GateMode.EXPLOIT: 1           # Default (low uncertainty)
}


# =============================================================================
# HELPER FUNCTIONS (для логирования, не для Gate routing)
# =============================================================================

def mode_to_string(mode: GateMode) -> str:
    """
    Конвертирует GateMode в строку.
    
    Args:
        mode: GateMode enum
    
    Returns:
        Строковое представление
    """
    return str(mode)


def string_to_mode(mode_str: str) -> GateMode:
    """
    Конвертирует строку в GateMode.
    
    Args:
        mode_str: Строка ("exploit", "explore", etc.)
    
    Returns:
        GateMode enum
    
    Raises:
        ValueError: Если строка не соответствует ни одному режиму
    """
    for mode in ALL_MODES:
        if str(mode) == mode_str.lower():
            return mode
    raise ValueError(f"Invalid mode string: {mode_str}")


def get_mode_priority(mode: GateMode) -> int:
    """
    Возвращает приоритет режима для threshold cascade.
    
    Args:
        mode: GateMode enum
    
    Returns:
        Приоритет (1-4, где 4 = highest)
    """
    return MODE_PRIORITY.get(mode, 0)


# =============================================================================
# TESTS (для быстрой проверки)
# =============================================================================

def test_gate_modes():
    """
    Test: GateMode enum basic functionality.
    """
    # Test enum values
    assert GateMode.EXPLOIT.value == "exploit"
    assert GateMode.EXPLORE.value == "explore"
    assert GateMode.EXPLOIT_SAFE.value == "exploit_safe"
    assert GateMode.ABSENCE_CHECK.value == "absence_check"
    
    # Test default
    assert GateMode.default() == GateMode.EXPLOIT
    
    # Test is_safe_mode
    assert GateMode.is_safe_mode(GateMode.EXPLOIT_SAFE) == True
    assert GateMode.is_safe_mode(GateMode.ABSENCE_CHECK) == True
    assert GateMode.is_safe_mode(GateMode.EXPLOIT) == False
    assert GateMode.is_safe_mode(GateMode.EXPLORE) == False
    
    # Test is_exploratory
    assert GateMode.is_exploratory(GateMode.EXPLORE) == True
    assert GateMode.is_exploratory(GateMode.ABSENCE_CHECK) == True
    assert GateMode.is_exploratory(GateMode.EXPLOIT) == False
    assert GateMode.is_exploratory(GateMode.EXPLOIT_SAFE) == False
    
    # Test priority
    assert get_mode_priority(GateMode.ABSENCE_CHECK) == 4
    assert get_mode_priority(GateMode.EXPLOIT_SAFE) == 3
    assert get_mode_priority(GateMode.EXPLORE) == 2
    assert get_mode_priority(GateMode.EXPLOIT) == 1
    
    # Test string conversion
    assert mode_to_string(GateMode.EXPLORE) == "explore"
    assert string_to_mode("exploit") == GateMode.EXPLOIT
    
    print("✓ PASS: GateMode enum tests")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("Stage 3.0: Gate Modes — Unit Tests")
    print("=" * 70)
    
    test_gate_modes()
    
    print("=" * 70)
    print("All tests completed!")
    print("=" * 70)