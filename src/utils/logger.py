# src/utils/logger.py

import logging


def setup_logger(name="EyeMonitor", level=logging.INFO):
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    logger.addHandler(console)
    logger.propagate = False

    return logger
