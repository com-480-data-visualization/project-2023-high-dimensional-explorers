# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Kyle Matoba <kyle.matoba@idiap.ch>
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: MIT
"""Miscellaneous functions."""
import logging
import math
import os
import time
from datetime import datetime, timezone


def _create_filename(log_dir):
    """Create a filename for the log file."""
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{datetime.now(timezone.utc):%Y%m%d%H%M%S%f}.log"
    return os.path.join(log_dir, log_filename)


def add_filehandler_to_root_logger(log_dir: str) -> str:
    """Add a file handler to the logger."""
    log_fullfilename = _create_filename(log_dir)
    f_handler = logging.FileHandler(log_fullfilename)
    f_handler.setLevel(logging.NOTSET)
    f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    f_handler.setFormatter(f_format)
    logging.root.addHandler(f_handler)
    return log_fullfilename


def get_logger():
    """Set default params for logger."""
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.setLevel(level=logging.DEBUG)
    logging.Formatter.converter = time.gmtime
    return logger

