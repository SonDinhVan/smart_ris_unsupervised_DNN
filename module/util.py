"""
This module suppports the utilities, e.g. Load data from yaml, etc.
"""
import yaml
from typing import Dict, Any

def load_config(filename: str) -> Dict[str, Any]:
    """
    Parse the config from the given YAML file
    
    Args:
        filename (str): [Path to the config file]

    Returns:
        Dict[str, Any]: [Configuration contained in a dictionary]
    """
    with open(filename, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)
