"""
Модуль логирования для Stage 2 экспериментов.
Единый формат для всех задач (Two-Step, Reversal, etc.).
"""

import csv
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np


def get_git_commit() -> str:
    """Возвращает текущий git commit hash."""
    import subprocess
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
    except:
        return 'unknown'


def get_config_hash(config_dict: Dict) -> str:
    """Вычисляет хэш конфигурации для отслеживания версий."""
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


# ============================================================================
# TRIAL LOGGER (построчное логирование)
# ============================================================================

class TrialLogger:
    """
    Логирует trial-level данные в CSV.
    """
    
    SCHEMA = [
        'trial', 'seed', 'config_version', 'git_commit',
        's1', 'a1', 's2', 'a2', 'trans_type', 'reward',
        'u_delta', 'u_s', 'u_v', 'u_c',
        'V_G', 'V_p', 'eta_G', 'eta_p', 'mode'
    ]
    
    def __init__(self, log_dir: str = 'logs/twostep/', experiment_id: str = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_id is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_id = f'exp_{timestamp}'
        
        self.experiment_id = experiment_id
        self.filepath = self.log_dir / f'{experiment_id}_trials.csv'
        
        self._write_header()
    
    def _write_header(self):
        with open(self.filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.SCHEMA)
    
    def log_trial(self, **kwargs):
        """
        Записывает один триал. Все поля из SCHEMA обязательны.
        """
        with open(self.filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            row = [kwargs.get(field, '') for field in self.SCHEMA]
            writer.writerow(row)
    
    def log_metadata(self, metadata: Dict):
        """Записывает метаданные эксперимента в JSON."""
        meta_path = self.log_dir / f'{self.experiment_id}_meta.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)


# ============================================================================
# EXPERIMENT LOGGER (управление экспериментом)
# ============================================================================

class ExperimentLogger:
    """
    Управляет логированием на уровне всего эксперимента.
    Создает директорию, сохраняет метаданные, агрегирует результаты.
    """
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = 'logs/',
        config: Dict = None,
        description: str = ''
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or {}
        self.description = description
        self.start_time = datetime.now()
        self.git_commit = get_git_commit()
        self.config_hash = get_config_hash(self.config)
        
        # Сохраняем метаданные сразу
        self._save_metadata()
    
    def _save_metadata(self):
        """Сохраняет метаданные эксперимента."""
        metadata = {
            'experiment_name': self.experiment_name,
            'description': self.description,
            'start_time': self.start_time.isoformat(),
            'git_commit': self.git_commit,
            'config_hash': self.config_hash,
            'config': self.config
        }
        
        meta_path = self.log_dir / 'experiment_meta.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def get_trial_logger(self, suffix: str = '') -> TrialLogger:
        """
        Создает TrialLogger для под-эксперимента.
        
        Args:
            suffix: Суффикс для имени файла (например, 'Full', 'NoVG', 'NoVp')
        
        Returns:
            TrialLogger instance
        """
        experiment_id = f'{self.experiment_name}_{suffix}' if suffix else self.experiment_name
        return TrialLogger(
            log_dir=str(self.log_dir),
            experiment_id=experiment_id
        )
    
    def save_results(self, results_df, filename: str = 'results.csv'):
        """
        Сохраняет агрегированные результаты (DataFrame) в CSV.
        
        Args:
            results_df: pandas DataFrame с результатами
            filename: Имя файла
        """
        filepath = self.log_dir / filename
        results_df.to_csv(filepath, index=False, encoding='utf-8')
    
    def save_figure(self, fig, filename: str):
        """
        Сохраняет фигуру (matplotlib) в директорию эксперимента.
        
        Args:
            fig: matplotlib Figure
            filename: Имя файла (с расширением .png, .pdf, .svg)
        """
        filepath = self.log_dir / 'figures' / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
    
    def finalize(self, end_time: datetime = None):
        """
        Завершает эксперимент, записывает время окончания и статус.
        """
        end_time = end_time or datetime.now()
        
        metadata = {
            'experiment_name': self.experiment_name,
            'description': self.description,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': (end_time - self.start_time).total_seconds(),
            'git_commit': self.git_commit,
            'config_hash': self.config_hash,
            'config': self.config,
            'status': 'completed'
        }
        
        meta_path = self.log_dir / 'experiment_meta.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def log_message(self, message: str, level: str = 'INFO'):
        """
        Записывает сообщение в лог эксперимента.
        
        Args:
            message: Текст сообщения
            level: Уровень (INFO, WARNING, ERROR)
        """
        log_path = self.log_dir / 'experiment.log'
        timestamp = datetime.now().isoformat()
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f'[{timestamp}] [{level}] {message}\n')