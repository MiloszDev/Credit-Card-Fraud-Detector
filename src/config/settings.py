"""Configuration manager for loading and accessing YAML configuration files."""

from pathlib import Path
from typing import Type, TypeVar, Any, Optional

from src.utils.functions import read_yaml
from src.entities.models import (
    GeneralConfig,
    DataConfig,
    EnvironmentConfig,
    DqnAgentConfig,
    TrainingConfig,
    EvaluationConfig,
    LoggingConfig
)

T = TypeVar('T')

class ConfigurationManager:
    """
    Loads configuration from a YAML file and provides typed access to config sections.
    """

    def __init__(self, config_file_path: Path) -> None:
        """
        Initialize ConfigurationManager by loading the YAML configuration file.

        Args:
            config_file_path (Path): Path to the configuration YAML file.
        """
        self.config = self._load_yaml(config_file_path)

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        """
        Load YAML configuration from a file.

        Args:
            path (Path): Path to the YAML config file.

        Returns:
            dict[str, Any]: Parsed YAML content.

        Raises:
            RuntimeError: If the file cannot be read or parsed.
        """
        try:
            return read_yaml(path)
        except Exception as e:
            raise RuntimeError(f"Error loading config file '{path}': {e}") from e

    def _get_config_section(
        self,
        section_name: str,
        config_type: Type[T],
        key_map: Optional[dict[str, str]] = None
    ) -> T:
        """
        Retrieve and parse a specific configuration section into a dataclass.

        Args:
            section_name (str): Key of the config section in YAML.
            config_type (Type[T]): Dataclass type to parse the section into.
            key_map (Optional[dict[str, str]]): Optional key remapping from YAML keys to dataclass fields.

        Returns:
            T: An instance of the config dataclass.

        Raises:
            KeyError: If the config section is missing.
            ValueError: If the config section is not a dict or parsing fails.
        """
        try:
            section = self.config[section_name]
        except KeyError as e:
            raise KeyError(f"Missing required config section '{section_name}'") from e

        if not isinstance(section, dict):
            raise ValueError(f"Config section '{section_name}' must be a dictionary")

        if key_map:
            section = {key_map.get(k, k): v for k, v in section.items()}

        try:
            section = self._convert_paths(section)
            return config_type(**section)
        except Exception as e:
            raise ValueError(f"Error parsing config section '{section_name}': {e}") from e

    def _convert_paths(self, config_section: dict[str, Any]) -> dict[str, Any]:
        """
        Convert string paths in the config section to pathlib.Path objects.

        Args:
            config_section (dict[str, Any]): Config section dictionary.

        Returns:
            dict[str, Any]: Config section with path strings converted to Path objects.
        """
        converted = {}
        for key, value in config_section.items():
            if isinstance(value, str) and (key.endswith('_path') or key.endswith('_dir')):
                converted[key] = Path(value)
            else:
                converted[key] = value
        return converted

    def get_general_config(self) -> GeneralConfig:
        return self._get_config_section('general', GeneralConfig)

    def get_data_config(self) -> DataConfig:
        return self._get_config_section('data', DataConfig)

    def get_environment_config(self) -> EnvironmentConfig:
        return self._get_config_section('env', EnvironmentConfig)

    def get_dqn_config(self) -> DqnAgentConfig:
        return self._get_config_section('dqn', DqnAgentConfig)

    def get_training_config(self) -> TrainingConfig:
        return self._get_config_section('training', TrainingConfig)

    def get_evaluation_config(self) -> EvaluationConfig:
        return self._get_config_section(
            'evaluation',
            EvaluationConfig,
            key_map={'visualize_confusion_matrix': 'visualize'}
        )

    def get_logging_config(self) -> LoggingConfig:
        return self._get_config_section('logging', LoggingConfig)
