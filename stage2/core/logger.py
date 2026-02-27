"""
Единый формат логирования для всех экспериментов.
"""

import csv
import json
import hashlib
from datetime import datetime
from pathlib import Path

class TrialLogger:
    """Логирует trial-level данные в CSV."""
    
    SCHEMA = [
        'trial', 'seed_env', 'seed_agent', 'config_version', 'git_commit',
        'a1', 'a2', 's2', 'trans_type', 'reward',
        'u_delta', 'u_s', 'u_v', 'u_c',
        'V_G', 'V_p', 'mode', 'belief_entropy',
        'eta_G', 'eta_p'
    ]
    
    def __init__(self, log_dir='logs/twostep/', experiment_id=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_id is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_id = f'exp_{timestamp}'
        
        self.experiment_id = experiment_id
        self.filepath = self.log_dir / f'{experiment_id}.csv'
        
        self._write_header()
    
    def _write_header(self):
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.SCHEMA)
    
    def log_trial(self, **kwargs):
        """Записывает один триал. Все поля из SCHEMA обязательны."""
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [kwargs.get(field, '') for field in self.SCHEMA]
            writer.writerow(row)
    
    def log_metadata(self, metadata: dict):
        """Записывает метаданные эксперимента в JSON."""
        meta_path = self.log_dir / f'{self.experiment_id}_meta.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

def get_git_commit():
    """Возвращает текущий git commit hash."""
    import subprocess
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except:
        return 'unknown'