"""
Logger utility for initializing colored and file-based logging with verbosity control.
"""

import os
import sys
import logging
import colorlog

from src.config.settings import ConfigurationManager


def get_logger(config: ConfigurationManager, verbose: bool = False) -> logging.Logger:
    """
    Initializes and returns a logger with color-coded console output and file logging.

    Args:
        config (ConfigurationManager): The configuration manager containing logging paths and names.
        verbose (bool, optional): If True, sets logging level to DEBUG; otherwise INFO. Defaults to False.

    Returns:
        logging.Logger: Configured logger instance.

    Raises:
        OSError: If the log directory or file cannot be created.
    """

    log_file = os.path.join(config.log_dir, "running_logs.log")
    log_level = logging.DEBUG if verbose else logging.INFO

    log_format = "%(log_color)s[%(levelname)s]%(reset)s - %(message)s"
    log_colors = {
        "DEBUG": "blue",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "magenta",
    }

    logger = logging.getLogger(config.logger_name)
    logger.setLevel(log_level)
    logger.propagate = False

    if not logger.handlers:
        console_handler = colorlog.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(
            colorlog.ColoredFormatter(fmt=log_format, log_colors=log_colors)
        )
        console_handler.setLevel(log_level)

        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(
            logging.Formatter(fmt="[%(levelname)s]: %(asctime)s - %(message)s")
        )
        file_handler.setLevel(log_level)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
