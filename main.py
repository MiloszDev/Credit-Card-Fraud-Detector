from pathlib import Path

from src.logger.handlers import get_logger
from src.config.settings import ConfigurationManager

CONFIG_FILE_PATH = Path("configs/config.yaml")

configuration_manager = ConfigurationManager(CONFIG_FILE_PATH)

logger = get_logger(configuration_manager.get_logging_config(), verbose=True)
logger.info("Logger initialized successfully.")