# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Control a pipeline, execution flow and configurations.

The basic design is:
    - The Processor manages all the inputs, configs and the running;
    - Processor has a ProductManager instance, that store all Products;
    - Processor has a lot of Stages, that get a Config, a Instrument and
      a Product and process it;
    - Stages modify one product per time;
    - All these things are objects that can be frozen for run.
"""

import yaml
from functools import partial
from collections import OrderedDict

from ..logger import logger, log_to_list


# TODO: implement logging
# __str__, print and to_header functions for all classes


__all__ = ['Product', 'ProductManager', 'Config', 'Stage', 'Instrument',
           'Processor']


class Config:
    """Store the config of a stage."""
    frozen = False
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            self[name] = value

    def __getattr__(self, name):
        return self[name]

    def __getitem__(self, name):
        if name in self.__dict__.keys():
            return self.__dict__[name]
        else:
            # TODO: think if it is better to return None or raise error
            return None

    def __setattr__(self, name, value):
        self[name] = value

    def __setitem__(self, name, value):
        if not self.frozen:
            if isinstance(value, dict):
                value = Config(**value)
            self.__dict__[name] = value

    def update(self, config):
        for k, v in config.items():
            self.__setitem__(k, v)

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()


class Product:
    """Store all informations and data of a product."""
    _product_manager = None
    _capabilities = []
    _destruct_callbacks = []
    _log_list = []
    _infos = OrderedDict()

    def __init__(self, product_manager=None, **kwargs):
        if product_manager is None:
            raise ValueError("A product has to be created with a"
                             " product_manager.")
        if not isinstance(product_manager, ProductManager):
            raise ValueError("product_manager is not a valid ProductManager "
                             "instance.")
        self._product_manager = product_manager

        # Setup the logger conveniently
        self._logger = self._product_manager.logger.getLogger()
        log_to_list(self._logger, self._log_list)

        for name, value in kwargs.items():
            self[name] = value

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
    def info(self):
        """Print general informations about the product, in a yaml format."""
        info_dict = self._infos.copy()

        info_dict['capabilities'] = self._capabilities

        dest = OrderedDict()
        dest['number'] = len(self._destruct_callbacks)
        dest['functions'] = [i.__name__ for i in self._destruct_callbacks]
        info_dict['destruction_callbacks'] = dest

        dest['history'] = self._log_list

        return yaml.dump(info_dict)

    @property
    def log(self):
        """Print the log of the product."""
        return "\n".join(self._log_list)

    def add_info(self, session, info_dict):
        """Custom add information dictionaries to the product. (to info prop)"""
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
            self._capability.remove(capability)

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

    def __getattr__(self, name):
        return self[name]

    def __getitem__(self, name):
        if name in self.__dict__.keys():
            return self.__dict__[name]
        else:
            # TODO: think if it is better to return None or raise error
            return None

    def __setattr__(self, name, value):
        self[name] = value

    def __setitem__(self, name, value):
        self.__dict__[name] = value

    def update(self, config):
        for k, v in config.items():
            self.__setitem__(k, v)

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()


class Instrument:
    """Store all the informations and needed functions of a instrument."""
    _frozen = False
    _prop_dict = {}
    # TODO: I know it is almost equal Config, think if it is better to
    # inherite Config or keep separated
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            self._prop_dict[name] = value

    def __getattr__(self, name):
        if name == '_prop_dict':
            return self.__dict__['_prop_dict']
        elif name == '_frozen':
            return self.__dict__['_frozen']
        else:
            return self._prop_dict[name]

    def __getitem__(self, name):
        return self.__getattr__(name)

    def __setattr__(self, name, value):
        if name == '_frozen':
            self._frozen = value
        elif name == '_prop_dict':
            raise ValueError('_prop_dict is a protected name')
        elif not self._frozen:
            self._prop_dict[name] = value

    def __setitem__(self, name, value):
        self.__setattr__(name, value)

    def freeze(self):
        self._frozen = True

    def unfreeze(self):
        self._frozen = False

    @property
    def frozen(self):
        return self._frozen

    def update(self, config):
        for k, v in config.items():
            self.__setitem__(k, v)

    def items(self):
        return self.prop_dict.items()


class ProductManager:
    """Manage a bunch of products."""
    _log_list = []
    _products = []
    _iterating = False

    def __init__(self, processor):
        if not isinstance(processor, Processor):
            raise ValueError("porcessor has to be a Processor instance.")
        self.processor = processor

        # Setup the logger conveniently
        self._logger = self.processor.logger.getLogger()
        log_to_list(self._logger, self._log_list)

    @property
    def number_of_products(self):
        return len(self.products)

    @property
    def iterating(self):
        return self._iterating

    @property
    def products(self):
        return self._products

    def add_product(self, product, index=None):
        """Add a product to the manager."""
        if product in self._products:
            raise ValueError('Product already in the manager.')
        else:
            if self.iterating:
                raise RuntimeError('Insert product in index while iteratin'
                                    'g not allowed.')
            elif index is not None:
                self._products.insert(index, product)
                product.product_manager = self
            else:
                self._products.append(product)
                product.product_manager = self

    def del_product(self, product):
        """Delete a product from the manager."""
        self.products.remove(product)

    def create_product(self, **kwargs):
        """Create a product using kwargs."""
        prod = Product(**kwargs)
        self.add_product(prod)
        return prod

    def iterate_products(self):
        """Iterate over all products."""
        if self.number_of_products == 0:
            raise ValueError('No products available in this product manager.')

        i = 0
        self._iterating = True
        while i < self.number_of_products:
            yield self.products[i]
            i += 1

        self._iterating = False

    def product(self, index):
        """Get one specific product."""
        return self.products[index]


class Stage:
    """Stage process (sub-part) of a pipeline."""
    config = Config()
    _children = []
    processor = None
    parent = None
    name = None
    _enabled = True

    def __init__(self, processor):
        self.processor = processor

    @property
    def enabled(self):
        return self._enabled

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = True

    def add_children(self, name, stage):
        """Add a children stage to this stage."""
        if stage not in self._children:
            self._children.append(stage)
            stage.parent = self
            stage.name = name

    def remove_children(self, stage):
        """Remove a children stage from this stage."""
        self._children.remove(stage)
        stage.parent = None

    def run(self, product, config=None, instrument=None):
        """Run the stage"""
        raise NotImplementedError('Stage not implemented.')


class Processor:
    """Master class of a pipeline"""
    def __init__(self, config_file=None, instrument=None, product_manager=None):
        self.product_manager = product_manager or ProductManager(self)
        self.instrument = instrument
        self.config = Config()
        self._stages = []
        self._processing_stage = None
        self.running = False

        if config_file is not None:
            with open(config_file, 'r') as stream:
                self.config.update(yaml.load(stream))

    @property
    def number_of_stages(self):
        return len(self._stages)

    def setup(self, **kwargs):
        """Set a special stage that populate the ProductManager.

        This function also can handle other things before properly run the
        pipeline.
        """
        raise NotImplementedError('setup is not implementated to this pipeline.'
                                  ' Required.')

    @property
    def processing_index(self):
        return self._processing_stage

    def set_current_stage(self, index):
        """Set the index of processing, useful for loops iteration loops."""
        if not self.running:
            raise ValueError('Current stage set only available when pipeline '
                             'is running.')
        self._processing_stage = index

    def add_stage(self, name, stage, index=None):
        """Add a stage to the pipeline."""
        if self.number_of_stages != 0:
            if name in self._stages[:][0]:
                raise ValueError('Stage {} already in the pipeline.'
                                 .format(name))
        if self.running:
            raise RuntimeError('Pipeline running, cannot add stages.')
        if not isinstance(stage, Stage):
            raise ValueError('Not a valid Stage.')

        if index is not None:
            self._stages.insert(index, (name, stage))
            stage.processor = self
        else:
            self._stages.append((name, stage))
            stage.processor = self

    def remove_stage(self, name):
        """Remove a stage from the pipeline."""
        if self.running:
            raise RuntimeError('Pipeline running, cannot remove stages.')
        index = self.get_index(name)
        self._stages[index].processor = None
        self._stages.pop(index)

    def get_index(self, name):
        """Get the index of a stage."""
        for i in self.number_of_stages:
            if self._stages[i][0] == name:
                return i
        return None

    def run(self, **runargs):
        """Run the pipeline."""

        self.setup(**runargs)

        if self.number_of_stages == 0:
            raise ValueError('This pipeline has no stages.')
        if self.product_manager.number_of_products == 0:
            raise ValueError('This pipeline has no products.')

        self.running = True
        for prod in self.product_manager.products:
            self._processing_stage = 0
            # Allow stages to set the current processing stage,
            # like for iterating
            while self._processing_stage < self.number_of_stages:
                stage_name, stage = self._stages[self._processing_stage]
                self._processing_stage += 1
                if stage_name in self.config.keys():
                    config = self.config[stage_name]
                else:
                    config = None
                stage.run(prod, config, instrument=self.instrument)
        self.running = False
        self._processing_stage = None
