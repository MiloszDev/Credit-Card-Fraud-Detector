"""Utility functions for reading YAML files with optional verbosity."""

import yaml
import logging

from pathlib import Path
from typing import Union, Optional
from ensure import ensure_annotations


@ensure_annotations
def read_yaml(path_to_yaml_file: Path, verbose: Optional[bool] = True) -> Union[dict, None]:
    """
    Reads a YAML file and returns its contents.

    Args:
        path_to_yaml_file (Path): Path to the YAML file.
        verbose (bool, optional): Whether to print log messages. Defaults to True.
    
    Returns:
        dict | None: The content of the YAML file as a dictionary, or None on failure.
    
    Raises:
        OSError: If the file cannot be opened or read.
        yaml.YAMLError: If the YAML is invalid.
    """
    try:
        with open(path_to_yaml_file, "r") as file:
            content = yaml.safe_load(file)

            if content is None:
                logging.error(f"Config file is empty or malformed.")
            elif verbose:
                logging.info(f"Successfully read YAML file: {path_to_yaml_file}")
            
            return content

    except (OSError, yaml.YAMLError) as e:
        logging.error(f"Error reading {path_to_yaml_file} file: {e}")
        
        return None