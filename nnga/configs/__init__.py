""" Module to create a defaul experiment config as a Singleton
    to be access in all package
"""

from .defaults import _C as cfg
from .defaults import export_config

__all__ = ["cfg", "export_config"]
