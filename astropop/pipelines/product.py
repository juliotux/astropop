# Licensed under a 3-clause BSD style license - see LICENSE.rst

from collections import OrderedDict
from functools import partial

from ._base import info_dumper
from ..logger import logger, log_to_list


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
