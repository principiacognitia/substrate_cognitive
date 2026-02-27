"""
Stage 2 Core Module — общие компоненты для всех задач.
"""

# Импорты будут работать только после создания соответствующих файлов
try:
    from .gate import gate_select, compute_diagnostic_vector
except ImportError:
    pass

try:
    from .rheology import update_rheology, eta_to_V
except ImportError:
    pass

try:
    from .baselines import MFAgent, MBAgent, HybridAgent, NoVGAgent
except ImportError:
    pass

try:
    from .logger import TrialLogger, ExperimentLogger
except ImportError:
    pass

__all__ = [
    'gate_select',
    'compute_diagnostic_vector',
    'update_rheology',
    'eta_to_V',
    'MFAgent',
    'MBAgent',
    'HybridAgent',
    'NoVGAgent',
    'TrialLogger',
    'ExperimentLogger'
]