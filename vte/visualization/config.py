"""Patch 21A: YAML config loader."""
from __future__ import annotations
import yaml
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config_regime_selection.yaml"

def load_regime_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Regime config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)