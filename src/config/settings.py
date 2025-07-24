"""Manages loading configurations for the data pipeline."""

from pathlib import Path
from src.logger import handlers
from src.utils.functions import read_yaml
from src.entities.models import (
    DataConfig,
    EnvironmentConfig,
    DqnAgentConfig,
    TrainingConfig,
    EvaluationConfig,
    LoggingConfig
)


class ConfigurationManager:
    """
    Loads configuration from the configuration file.
    """

    def __init__(self, config_file_path: Path) -> None:
        """
        Initialize the ConfigurationManager by loading the YAML configuration file.

        Args:
            config_file_path (Path): Path to the configuration YAML file.

        Raises:
            RuntimeError: If the configuration file cannot be read or parsed.
        """
        try:
            self.config = read_yaml(config_file_path)

        except Exception as e:
            raise RuntimeError(f"Error loading the configuration file or creating directories: {e}")

    @staticmethod
    def _convert_to_path(path: str) -> Path:
        """
        Converts a string path to a Path object.

        Args:
            path (str): Path as a string.

        Returns:
            Path: Path object.
        """
        return Path(path)

    def get_data_config(self) -> DataConfig:
        """
        Retrieves data configuration.

        Returns:
            DataConfig: Parsed data configuration.

        Raises:
            KeyError, ValueError: If required data keys are missing or invalid.
        """
        try:
            config = self.config["data"]

            return DataConfig(
                raw_path=self._convert_to_path(config["raw_path"]),
                processed_path=self._convert_to_path(config["processed_path"]),
                test_split=float(config["test_split"]),
                scaler=config["scaler"]
            )
        
        except Exception as e:
            raise ValueError(f"Error loading data config: {e}")

    def get_environment_config(self) -> EnvironmentConfig:
        """
        Retrieves environment configuration.

        Returns:
            EnvironmentConfig: Parsed environment configuration.

        Raises:
            KeyError, ValueError: If required environment keys are missing or invalid.
        """
        try:
            config = self.config["env"]

            return EnvironmentConfig(
                type=config["type"],
                steps=int(config["steps"])
            )
        
        except Exception as e:
            raise ValueError(f"Error loading environment config: {e}")

    def get_dqn_config(self) -> DqnAgentConfig:
        """
        Retrieves DQN agent configuration.

        Returns:
            DqnAgentConfig: Parsed DQN configuration.

        Raises:
            KeyError, ValueError: If required DQN keys are missing or invalid.
        """
        try:
            config = self.config["dqn"]

            return DqnAgentConfig(
                state_dim=int(config["state_dim"]),
                action_dim=int(config["action_dim"]),
                hidden_dim=int(config["hidden_dim"]),
                gamma=float(config["gamma"]),
                epsilon_start=float(config["epsilon_start"]),
                epsilon_min=float(config["epsilon_min"]),
                epsilon_decay=float(config["epsilon_decay"]),
                learning_rate=float(config["learning_rate"]),
                target_update_freq=int(config["target_update_freq"]),
                memory_size=int(config["memory_size"]),
                batch_size=int(config["batch_size"])
            )
        
        except Exception as e:
            raise ValueError(f"Error loading DQN config: {e}")

    def get_training_config(self) -> TrainingConfig:
        """
        Retrieves training configuration.

        Returns:
            TrainingConfig: Parsed training configuration.

        Raises:
            KeyError, ValueError: If required training keys are missing or invalid.
        """
        try:
            config = self.config["training"]

            return TrainingConfig(
                num_episodes=int(config["num_episodes"]),
                max_steps_per_episode=int(config["max_steps_per_episode"]),
                log_interval=int(config["log_interval"]),
                save_model=bool(config["save_model"]),
                model_save_path=self._convert_to_path(config["model_save_path"])
            )
        
        except Exception as e:
            raise ValueError(f"Error loading training config: {e}")

    def get_evaluation_config(self) -> EvaluationConfig:
        """
        Retrieves evaluation configuration.

        Returns:
            EvaluationConfig: Parsed evaluation configuration.

        Raises:
            KeyError, ValueError: If required evaluation keys are missing or invalid.
        """
        try:
            config = self.config["evaluation"]

            return EvaluationConfig(
                metrics=config["metrics"],
                threshold=float(config["threshold"]),
                visualize=bool(config.get("visualize_confusion_matrix", False))
            )
        
        except Exception as e:
            raise ValueError(f"Error loading evaluation config: {e}")

    def get_logging_config(self) -> LoggingConfig:
        """
        Retrieves logging configuration.

        Returns:
            LoggingConfig: Parsed logging configuration.

        Raises:
            KeyError, ValueError: If required logging keys are missing or invalid.
        """
        try:
            config = self.config["logging"]
            
            return LoggingConfig(
                use_wandb=bool(config["use_wandb"]),
                wandb_project=config["wandb_project"],
                wandb_entity=config["wandb_entity"],
                logger_name=config["logger_name"],
                log_dir=self._convert_to_path(config["log_dir"])
            )
        
        except Exception as e:
            raise ValueError(f"Error loading logging config: {e}")
