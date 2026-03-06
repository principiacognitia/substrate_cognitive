"""
Stage 2 Core Module — общие компоненты для всех задач.
"""

from .gate import gate_select, sigmoid
from .rheology import update_rheology, eta_to_V, ETA_0
from .baselines import (
    MFAgent,
    MBAgent,
    HybridAgent,
    RheologicalAgent,
    RheologicalAgent_NoVG,
    RheologicalAgent_NoVp,
    RheologicalAgent_NoReology
)
from .logger import TrialLogger, ExperimentLogger
from .args import parse_args, print_debug, print_always, get_default_parser

__all__ = [
    # Gate
    'gate_select',
    'sigmoid',
    
    # Rheology
    'update_rheology',
    'eta_to_V',
    'ETA_0',
    
    # Agents
    'MFAgent',
    'MBAgent',
    'HybridAgent',
    'RheologicalAgent',
    'RheologicalAgent_NoVG',
    'RheologicalAgent_NoVp',
    'RheologicalAgent_NoReology',
    
    # Logger
    'TrialLogger',
    'ExperimentLogger',
    
    # Args
    'parse_args',
    'print_debug',
    'print_always',
    'get_default_parser'
]