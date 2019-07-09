# Licensed under a 3-clause BSD style license - see LICENSE.rst

import abc

from ._base import Config
from ..logger import logger


class Stage(abc.ABC):
    """Stage process (sub-part) of a pipeline."""
    config = Config()  # stage default config
    name = None  # stage name
    _enabled = True  # stage enabled
    _requested_functions = []  # Instrument needed functions
    _requirements = []  # Product needed variables
    _provided = []  # Product variables provided by stage
    logger = logger

    def __init__(self, processor):
        self.processor = processor

    @property
    def enabled(self):
        return self._enabled

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    @abc.abstractmethod
    def run(self, product, config=None):
        """Run the stage"""

    def __call__(self, product, config=None):
        self.logger = product.logger
        self.run(product, config)
