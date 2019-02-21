# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Control a pipeline, execution flow and configurations.

The basic design is:
    - The Processor manages all the inputs, configs and the running;
    - Processor has a ProductManager instance, that store all Products;
    - Processor has a lot of Stages, that get a Config, a Instrument and
      a Product and process it;
    - Stages modify one product per time;
    - Stage can have Stage children, Config can have Config children. In the
      same logic.
"""


__all__ = ['Product', 'ProductManager', 'Config', 'Stage', 'Instrument',
           'Processor']


class Config:
    """Store the config of a stage."""
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            self.__dict__[name] = value

    def __getattribute__(self, name):
        if name in self.__dict__.keys():
            return object.__getattribute__(self, name)
        else:
            # TODO: think if it is better to return None or raise error
            return None

    def __setattr__(self, name, value):
        self.__dict__[name] = value


class Product:
    """Store all informations and data of a product."""
    # TODO: inherite Config?


class Instrument:
    """Store all the informations and needed functions of a instrument."""
    # TODO: I know it is almost equal Config, think if it is better to
    # inherite Config or keep separated
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            self.__dict__[name] = value

    def __getattribute__(self, name):
        if name in self.__dict__.keys():
            return object.__getattribute__(self, name)
        else:
            # TODO: think if it is better to return None or raise error
            return None

    def __setattr__(self, name, value):
        self.__dict__[name] = value


class ProductManager:
    """Manage a bunch of products."""
    def __init__(self, processor):
        self.processor = processor
        self.products = []

    def add_product(self, product, index=None):
        """Add a product to the manager."""
        if product in self.products:
            raise ValueError('Product already in the manager.')
        else:
            if index is not None:
                self.products.insert(index, product)
            else:
                self.products.append(product)

    def del_product(self, product):
        """Delete a product from the manager."""
        raise NotImplementedError

    def iterate_products(self):
        """Iterate over all products."""
        for i in range(len(self.products)):
            yield self.products[i]

    def product(self, index):
        """Get one specific product."""
        return self.products[index]


class Stage:
    """Stage process (sub-part) of a pipeline."""
    config = Config()
    _children = []

    def add_children(self, stage):
        """Add a children stage to this stage."""

    def remove_children(self, stage):
        """Remove a children stage from this stage."""

    def run(self, product, config=None, instrument=None):
        """Run the stage"""


class Processor:
    """Master class of a pipeline"""
    def __init__(self, config_file=None, iterate_over='stages'):
        self.prod_manager = ProductManager()
        self.instrument = None
        self.config_dict = {}
        self._group_stages = []

    def add_stage(self, name, stage, group=None, index=None):
        """Add a stage to the pipeline."""

    def remove_stage(self, name):
        """Remove a stage from the pipeline."""

    def run(self, **runargs):
        """Run the pipeline."""
