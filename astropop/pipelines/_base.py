# Licensed under a 3-clause BSD style license - see LICENSE.rst

import abc
import yaml
from collections import OrderedDict

from ..logger import logger


# TODO: __str__, __repr__, print functions for all classes


__all__ = ['Config']


class IncompatibilityError(RuntimeError):
    """Error to be raised for come incompatibility."""


def info_dumper(infos):
    """Dump a dictionary information to a formated string.

    Now, it's just a wrapper to yaml.dump, put here to customize if needed.
    """
    return yaml.dump(infos)


class _GenericConfigClass(abc.ABC):
    """Class for generic sotring configs. Like a powered dict."""
    _frozen = False
    _prop_dict = OrderedDict()
    _mutable_vars = ['_frozen', 'logger']
    logger = logger

    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            self.__setitem__(name, value)

    def __getattr__(self, name):
        if name in self._prop_dict.keys():
            return self.__getitem__(name)
        else:
            return super().__getattribute__(name)

    def __getitem__(self, name):
        if name in self._prop_dict:
            return self._prop_dict[name]
        else:
            # TODO: think if it is better to return None or raise error
            return None

    def __setattr__(self, name, value):
        if self._frozen:
            self.logger.warn('Tried to change `{}` with value `{}` while'
                             ' {} is frozen. Skipping.'
                             .format(name, value, self.__class__.__name__))
            return
        if name not in self.__class__.__dict__.keys():
            self.__setitem__(name, value)
        elif name in self._mutable_vars:
            # mutable vars
            super().__setattr__(name, value)
        else:
            raise KeyError('{} is a protected variable.'.format(name))

    def __setitem__(self, name, value):
        self._prop_dict[name] = value

    def __delattr__(self, name):
        if name in self._prop_dict.keys():
            del self._prop_dict[name]
        else:
            super().__delattr__(name)

    def __repr__(self):
        info = self.__class__.__name__ + "\n\n"
        info += info_dumper({'Properties': self.properties})
        return info

    def get(self, key, value):
        return self._prop_dict.get(key, value)

    def freeze(self):
        self._frozen = True

    def unfreeze(self):
        self._frozen = False

    @property
    def properties(self):
        return self._prop_dict.copy()

    @property
    def frozen(self):
        return self._frozen

    def update(self, config):
        for k, v in config.items():
            self.__setitem__(k, v)

    def items(self):
        return self._prop_dict.items()


class Config(_GenericConfigClass):
    """Store the config of a stage."""
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
