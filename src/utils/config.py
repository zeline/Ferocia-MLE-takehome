"""Configuration loader utility."""

import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        Dictionary containing configuration parameters.
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to the project root.
    """
    return Path(__file__).parent.parent.parent

