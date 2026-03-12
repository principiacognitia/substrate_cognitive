"""
Загрузчики данных для Stage 2 экспериментов.
Читает CSV логи и агрегирует по seeds.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def find_latest_experiment(task: str, base_dir: str = 'logs/') -> Optional[str]:
    """
    Ищет последний эксперимент указанного типа по timestamp.
    
    Args:
        task: 'twostep' или 'reversal'
        base_dir: Базовая директория логов
        
    Returns:
        Путь к директории последнего эксперимента или None
    """
    search_dir = Path(base_dir) / task
    
    if not search_dir.exists():
        return None
    
    # Ищем все папки с экспериментами
    exp_dirs = [d for d in search_dir.iterdir() 
                if d.is_dir() and d.name.startswith(f'{task}_')]
    
    if not exp_dirs:
        return None
    
    # Сортируем по имени (timestamp в имени)
    exp_dirs.sort(key=lambda x: x.name, reverse=True)
    
    return str(exp_dirs[0])


def find_experiment_by_id(experiment_id: str, base_dir: str = 'logs/') -> Optional[str]:
    """
    Ищет эксперимент по точному ID.
    """
    # Пробуем в twostep
    path = Path(base_dir) / 'twostep' / experiment_id
    if path.exists():
        return str(path)
    
    # Пробуем в reversal
    path = Path(base_dir) / 'reversal' / experiment_id
    if path.exists():
        return str(path)
    
    return None

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Стандартизирует имена колонок в DataFrame.
    
    Разные эксперименты используют разные имена для одних и тех же данных:
    - trans_factor / transition / trans_type
    - mode_numeric / mode
    - и т.д.
    """
    df = df.copy()
    
    # Если есть 'trans_type' (строка 'common'/'rare'), конвертируем в 'trans_factor' (1/-1)
    if 'trans_type' in df.columns and 'trans_factor' not in df.columns:
        df['trans_factor'] = df['trans_type'].apply(lambda x: 1 if x == 'common' else -1)
    
    # Если есть 'transition' (число 0/1), конвертируем в 'trans_factor' (1/-1)
    if 'transition' in df.columns and 'trans_factor' not in df.columns:
        df['trans_factor'] = df['transition'].apply(lambda x: 1 if x == 1 else -1)
    
    # Если есть 'mode' (строка), создаём 'mode_numeric'
    if 'mode' in df.columns and 'mode_numeric' not in df.columns:
        df['mode_numeric'] = df['mode'].apply(lambda x: 1 if x == 'EXPLORE' else 0)

    # ← ДОБАВЛЕНО: Если нет 'stay', но есть 'a1' и 'prev_a1', вычисляем
    if 'stay' not in df.columns and 'a1' in df.columns:
        df['stay'] = df['a1'].eq(df['a1'].shift(1)).astype(int)

    return df

def load_experiment_data(exp_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Загружает все CSV логи из директории эксперимента.
    
    Args:
        exp_dir: Путь к директории эксперимента
        
    Returns:
        Dict {agent_name: DataFrame с агрегированными данными}
    """
    exp_path = Path(exp_dir)
    
    if not exp_path.exists():
        raise FileNotFoundError(f"Эксперимент не найден: {exp_dir}")
    
    # Ищем все CSV файлы с триалами
    csv_files = list(exp_path.glob('*_trials.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"CSV файлы не найдены в {exp_dir}")
    
    # Группируем по агентам
    agent_data = {}
    
    for csv_file in csv_files:
        # Извлекаем имя агента из имени файла
        # Форматы: {exp_id}_{agent}_seed{N}_trials.csv
        parts = csv_file.stem.split('_')
        
        # Ищем часть с именем агента
        agent_name = None
        for part in parts:
        # Поддержка всех форматов имён агентов
            if part in ['Full', 'NoVG', 'NoVp', 'MF', 'MB', 
                        'RheologicalAgent', 'MFAgent', 'MBAgent',
                        'RheologicalAgent_NoVG', 'RheologicalAgent_NoVp']:
                agent_name = part
                break
        
        # Если не нашли явное имя, пробуем найти по индексу
        if agent_name is None:
            for i, part in enumerate(parts):
                if part.startswith('seed') and i > 0:
                    potential_agent = parts[i-1]
                    if potential_agent in ['Full', 'NoVG', 'NoVp', 'MF', 'MB', 
                        'RheologicalAgent', 'MFAgent', 'MBAgent',
                        'RheologicalAgent_NoVG', 'RheologicalAgent_NoVp']:
                        agent_name = potential_agent
                        break
        
        if agent_name is None:
            continue
        
        # Загружаем CSV
        df = pd.read_csv(csv_file)

        df = standardize_columns(df)
        
        if agent_name not in agent_data:
            agent_data[agent_name] = []
        agent_data[agent_name].append(df)
    
    # Агрегируем по seeds
    aggregated = {}
    for agent_name, dfs in agent_data.items():
        aggregated[agent_name] = pd.concat(dfs, ignore_index=True)
    
    return aggregated


def load_meta_data(exp_dir: str) -> Optional[Dict]:
    """
    Загружает метаданные эксперимента из JSON.
    """
    meta_path = Path(exp_dir) / 'experiment_meta.json'
    
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    return None