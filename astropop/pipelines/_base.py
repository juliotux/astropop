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


class Instrument(_GenericConfigClass):
    """Store all the informations and needed functions of a instrument."""
    _frozen = False
    _prop_dict = {}
    _mutable_vars = ['_frozen']
    _identifier = 'dummy_instrument'

    def __init__(self, *args, **kwargs):
        super(Instrument, self).__init__(*args, **kwargs)

    def list_functions(self):
        """List the class functions."""
        # Pass any callable object that do not start with '_' may cause problem
        # This may pass unwanted functions. Commented out.
        # l = [i.__name__ for i in self.__class__.__dict__.values()
        #      if callable(i) and i.__name__[0] != '_']

        funcs = [i.__name__ for i in self.__class__.__dict__.values()
                 if type(i) in ['function', 'builtin_function_or_method'] and
                 i.__name__[0] != '_']

        # If needed to remove another class function, put here
        for i in ['list_functions']:
            if i in funcs:
                funcs.remove(i)
        return funcs

    def __str__(self):
        info = "{} ({})\n\n".format(self.__class__.__name__, self._identifier)
        info += info_dumper({'Properties': self.properties,
                             'Functions': self.list_functions()})
        return info

    def __repr__(self):
        return "{} ({})".format(self.__class__.__name__, self._identifier)


class Product():
    """Store all informations and data of a product."""
    _product_manager = None  # Parent product manager
    _capabilities = []  # The product capabilities
    _destruct_callbacks = []  # List of destruction callbacks
    _log_list = []  # Store the logs
    _infos = OrderedDict()  # Custom product informations
    _mutable_vars = ['_product_manager']  # Variables that can be changed
    _instrument = None  # Product instrument

    # Product do not subclass _GenericConfigClass to keep variables
    # acessing more customized.
    def __init__(self, product_manager=None, instrument=None, index=None,
                 **kwargs):
        if product_manager is None:
            raise ValueError("A product has to be created with a"
                             " product_manager.")
        # if not isinstance(product_manager, ProductManager):
        #     raise ValueError("product_manager is not a valid ProductManager "
        #                      "instance.")
        self._product_manager = product_manager
        self._instrument = instrument
        self.index = index

        # Setup the logger conveniently
        self._logger = self._product_manager.logger.getChild('product')
        log_to_list(self._logger, self._log_list)

        for name, value in kwargs.items():
            self.__setattr__(name, value)

    @property
    def logger(self):
        return self.logger

    @property
    def product_manager(self):
        return self._product_manager

    @property
    def capabilities(self):
        return self._capabilities

    @property
    def instrument(self):
        return self._instrument

    @property
    def info(self):
        """Print general informations about the product, in a yaml format."""
        info_dict = self._infos.copy()

        info_dict['capabilities'] = self._capabilities

        inst = OrderedDict()
        inst['class'] = self._instrument.__class__.__name__
        inst['id'] = self._instrument._identifier
        inst['properties'] = self._instrument.properties
        info_dict['instrument'] = inst

        dest = OrderedDict()
        dest['number'] = len(self._destruct_callbacks)
        dest['functions'] = [i.__name__ for i in self._destruct_callbacks]
        info_dict['destruction_callbacks'] = dest

        dest['history'] = self._log_list

        return info_dumper(info_dict)

    @property
    def log(self):
        """Print the log of the product."""
        return "\n".join(self._log_list)

    def add_info(self, session, info_dict):
        """Custom add information dictionaries to the product."""
        if session in self._infos.keys():
            self.logger.warn('Session {} already exists in this product infos.'
                             ' Overwriting it.'
                             .format(session))
        if session in ['history', 'capabilities', 'destruction_callbacks']:
            self.logger.warn('{} is a protected name of session. Skipping.'
                             .format(session))
            return
        self._infos[session] = info_dict

    def add_capability(self, capability):
        if capability not in self._capabilities:
            self._capabilities.append(capability)

    def del_capability(self, capability):
        if capability in self._capabilities:
            self._capabilities.remove(capability)

    def add_destruct_callback(self, callback, *args, **kwargs):
        """Add a destruction callback. First argument must be a class slot,
        like self."""
        func = partial(callback, self, *args, **kwargs)
        self._destruct_callbacks.append(func)

    def destruct(self):
        """Execute the destruction callbacks sequentially."""
        for i, func in enumerate(self._destruct_callbacks):
            try:
                func()
            except Exception as e:
                logger.debug("Destruction callback {} problem. Error: {}"
                             .format(i, e))

    def update(self, config):
        for k, v in config.items():
            self.__setitem__(k, v)

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, name):
        return self.__dict__.__getitem__(name)

    def __getattr__(self, name):
        if name in self.__dict__.keys():
            return self.__getitem__(name)
        else:
            super().__getattribute__(name)

    def __setitem__(self, name, value):
        self.__dict__.__setattr__(name, value)

    def __setattr__(self, name, value):
        if name not in self.__class__.__dict__.keys():
            self.__setitem__(name, value)
        elif name not in self._mutable_vars:
            super().__setattr__(name, value)
        else:
            raise KeyError('{} is a protected variable.'.format(name))

    def __delattr__(self, name):
        if name in self.__dict__.keys():
            del self.__dict__[name]
        else:
            super().__delattr__(name)

    def __str__(self):
        return "Product: {}\n{}".format(self.__class__.__name__, self.info)

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


class Manager():
    _stages = OrderedDict()  # Stages list
    _register = OrderedDict()  # Variables registering
    _config = Config()
    _targets = []  # Stage targets for pipeline execution
    _active_product = None


class Boss(abc.ABC):
    _config = Config()
    _products = OrderedDict()

    """Class to handle the general pipeline management."""
    def __init__(self, config_file=None):
        if config_file is not None:
            with open(config_file, 'r') as stream:
                self._config.update(yaml.load(stream))
