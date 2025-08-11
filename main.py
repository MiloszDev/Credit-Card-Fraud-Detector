from pathlib import Path

from src.config.settings import ConfigurationManager
from pipelines.training_pipeline import train_pipeline

CONFIG_FILE_PATH = Path("configs/config.yaml")

configuration_manager = ConfigurationManager(CONFIG_FILE_PATH)

raw_data_path = str(configuration_manager.get_data_config().raw_path)

train_pipeline(raw_data_path, verbose=True)