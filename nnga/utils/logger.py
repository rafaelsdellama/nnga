import logging
import os
import sys
from nnga.utils.data_io import create_dir


def setup_logger(name, save_dir, filename="log.txt"):
    """
        Parameters
        ----------
        name : str
            Logger name
        save_dir : str|path:
            dir to save log files
        filename :
             (Default value = "log.txt")
            file name for log files
        Returns
        -------
            Logger
                Logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        create_dir(save_dir)
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_logger(name):
    """

    Parameters
    ----------
    name : str
        Logger name

    Returns
    -------
        Logger
            Logger object
    """
    return logging.getLogger(name)