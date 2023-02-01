"""
Global logging configuration.
"""

import logging
import sys


def setup_custom_logger(name):
    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s")

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)

    # https://docs.python.org/3/library/logging.html#levels
    logger.setLevel(logging.DEBUG)

    if not logger.hasHandlers():
        logger.addHandler(handler)

    logging.info("Setup logger.")
    return logger
