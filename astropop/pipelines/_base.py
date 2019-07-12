# Licensed under a 3-clause BSD style license - see LICENSE.rst

import abc
import yaml
from functools import partial
from collections.abc import Mapping
import copy

from ..logger import logger, log_to_list
from ..py_utils import IndexedDict, check_iterable


# TODO: __str__, __repr__, print functions for all classes
# TODO: More logging and debuging


__all__ = ['Config', 'Product', 'Instrument', 'Manager', 'Factory',
           'Stage']


def info_dumper(infos):
    """Dump a dictionary information to a formated string.

    Now, it's just a wrapper to yaml.dump, put here to customize if needed.
    """
    return yaml.dump(infos)


class Config(dict):
    """Class for generic sotring configs. Like a powered dict."""
    _mutable_vars = ['_frozen', 'logger']

    def __init__(self, *args, **kwargs):
        self._frozen = False
        super().__init__()

        if len(args) > 1:
            raise ValueError('Just one positional argument supported.')
        elif len(args) == 1:
            up = args[0]
            if isinstance(up, Mapping):
                self.update(up)
            elif check_iterable(up):
                for i, v in up:
                    self.__setitem__(i, v)
            else:
                raise TypeError('Argument not supported for dict'
                                ' initialization.')

        for i, v in kwargs:
            self.__setitem__(i, v)

    def __setitem__(self, name, value):
        if self._frozen:
            raise ValueError('Tried to change `{}` with value `{}` while'
                             ' {} is frozen.'
                             .format(name, value, self.__class__.__name__))

        if isinstance(value, Mapping):
            # Convert to this class if not
            value = self.__class__(value)

        super().__setitem__(name, value)

    def __delitem__(self, name):
        if self._frozen:
            raise ValueError('Tried to delete `{}` while'
                             ' {} is frozen.'
                             .format(name, self.__class__.__name__))
        return super().__delitem__(name)

    def update(self, dictlike):
        for k, v in dictlike.items():
            self.__setitem__(k, v)

    def freeze(self):
        self._frozen = True
        for v in self.values():
            if isinstance(v, Config):
                v.freeze()
        return self

    def unfreeze(self):
        self._frozen = False
        for v in self.values():
            if isinstance(v, Config):
                v.unfreeze()
        return self

    @property
    def frozen(self):
        return self._frozen


class Instrument(abc.ABC):
    """Store all the informations and needed functions of a instrument."""
    _frozen = False
    _mutable = ['_frozen']
    _identifier = ''

    def __init__(self, *args, **kwargs):
        super(Instrument, self).__init__(*args, **kwargs)

    def __setattr__(self, name, value):
        if self._frozen and name not in self._mutable:
            raise ValueError('Trying to change an attr while frozen.')
        return super().__setattr__(name, value)

    def __delattr__(self, name):
        if self._frozen:
            raise ValueError('Trying to delete an attr while frozen.')
        return super().__delattr__(name)

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

    def freeze(self):
        self._frozen = True
        return self

    def unfreeze(self):
        self._frozen = False
        return self

    @property
    def frozen(self):
        return self._frozen


class Product():
    """Store all informations and data of a product."""
    _destruct_callbacks = []  # List of destruction callbacks
    _log_list = []  # Store the logs
    _variables = {}
    _infos = IndexedDict()
    _instrument = None  # Product instrument
    _logger = None
    _targets = []

    def __init__(self, manager=None, instrument=None, targets=[],
                 **kwargs):
        if manager is None:
            raise ValueError("A product has to be created with a"
                             " manager.")
        self._manager = manager
        self._instrument = instrument
        self._targets = targets

        for name, value in kwargs.items():
            self.__setattr__(name, value)

    @property
    def index(self):
        return self._manager.get_product_index(self.name)

    @property
    def name(self):
        return self._manager.get_product_name(self)

    @property
    def logger(self):
        if self._logger is None:
            self._logger = self._manager.logger.getChild(self.name)
            self._logger.setLevel(self._manager.logger.getEffectiveLevel())
            log_to_list(self._logger, self._log_list)
        return self._logger

    @property
    def manager(self):
        return self._manager

    @property
    def instrument(self):
        return self._instrument

    @property
    def info(self):
        """Print general informations about the product, in a yaml format."""
        info_dict = self._infos.copy()

        info_dict['variables'] = self._variables

        inst = dict()
        inst['class'] = self._instrument.__class__.__name__
        inst['id'] = self._instrument._identifier
        inst['properties'] = self._instrument.properties
        info_dict['instrument'] = inst

        dest = dict()
        dest['number'] = len(self._destruct_callbacks)
        dest['functions'] = [i.__name__ for i in self._destruct_callbacks]
        info_dict['destruction_callbacks'] = dest

        dest['history'] = self._log_list

        return info_dumper(info_dict)

    @property
    def log(self):
        """Print the log of the product."""
        return "\n".join(self._log_list)

    @property
    def targets(self):
        return copy.deepcopy(self._targets)

    def get_value(self, name):
        """Get a variable value."""
        return self._variables[name]

    def set_value(self, name, value):
        """Set a value to a variable."""
        self._variables[name] = value

    def add_info(self, session, info_dict):
        """Custom add information dictionaries to the product."""
        if session in self._infos.keys():
            self.logger.warn('Session {} already exists in this product infos.'
                             ' Overwriting it.'
                             .format(session))
        if session in ['variables', 'history', 'destruction_callbacks']:
            self.logger.warn('{} is a protected name of session. Skipping.'
                             .format(session))
            return
        self._infos[session] = info_dict

    def add_target(self, name):
        """Add a stage to the target list."""
        if name not in self._targets:
            self._targets.append(name)
        else:
            self.logger.debug('Stage {} already in product targets.'
                              .format(name))

    def del_target(self, name):
        """Remove a stage from the target list."""
        if name in self._targets:
            self._targets.remove(name)
        else:
            self.logger.debug('Stage {} not in the target list.'
                              .format(name))

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


class Stage(abc.ABC):
    """Stage process (sub-part) of a pipeline."""
    _default_config = Config()  # stage default config
    _requested_functions = []  # Instrument needed functions
    _required_variables = []  # Product needed variables
    _optional_variables = []  # Optional variables for product ruunning
    _provided = []  # Product variables provided by stage
    _status = 'idle'
    _raise_error = True
    # TODO: Implement instrument compatibility checking

    def __init__(self, factory):
        self._factory = factory

    @property
    def status(self):
        return self._status

    @property
    def name(self):
        return self.factory.get_stage_name(self)

    @property
    def index(self):
        return self.factory.get_stage_index(self.name)

    @property
    def factory(self):
        return self._factory

    @property
    def logger(self):
        return self.factory.logger

    @property
    def defaults(self):
        return Config(**self._default_config)

    @status.setter
    def status(self, status):
        if status not in ['idle', 'running', 'done', 'error']:
            raise ValueError('Status {} not allowed.'
                             .format(status))
        self._status = status

    def wait(self):
        """Wait process to finish."""
        raise RuntimeError('No processing running in parallel now. '
                           'This should not be possible!')

    def get_variables(self):
        """Return a dict of all needed and optional variables."""
        variables = {}
        for v in self._required_variables:
            variables[v] = self.factory.get_value(self, v)
        # Optional variables will get None if not available
        for v in self._optional_variables:
            try:
                variables[v] = self.factory.get_value(self, v)
            except:
                variables[v] = None

        return variables

    def get_instrument(self):
        """Get the instrument class of active product."""
        return self.factory.get_instrument()

    @abc.abstractmethod
    def callback(self, instrument, variables, config):
        """Run the stage. Return a dict of processed variables."""
        # Avoiding access to class stuff make possible parallelizing

    def _call_pipeline(self, instrument, config):
        try:

            # Ensure get all variables before running
            variables = self.get_variables()
            result = {i: None for i in self.factory.owned_variables(self)}
            conf = copy.deepcopy(self.defaults)
            conf.update(conf)

            self.status = 'running'
            instrument = self.get_instrument()
            # TODO: Async apply this method?
            result.update(self.callback(instrument, variables, config))
            self.status = 'done'
            for i, v in result.items():
                if i in self.factory.owned_variables(self):
                    self.factory.set_value(self, i, v)
        except Exception as e:
            if self._raise_error:
                raise e
            self.logger.error('Stage {} error: {}'
                              .format(self.name, e))
            self.status = 'error'


class Factory():
    _stages = IndexedDict()  # Stages list
    _register = IndexedDict()  # Variables registering
    _active_prod = None
    _active_config = Config()
    _logger = None

    # Targets are defined by product
    """Class to handle execution and product/stage interface."""
    def __init__(self, manager):
        self._manager = manager

    @property
    def logger(self):
        if self._logger is None:
            self._logger = self._manager.logger.getChild('factory')
            self._logger.setLevel(self._manager.logger.getEffectiveLevel())
        return self._logger

    def get_value(self, stage, variable):
        """Get a value from a registered variable."""
        if variable not in self._register.keys():
            raise KeyError('Variable {} not registered.'
                           .format(variable))

        if stage is None or self._register[variable] is None:
            # Non-stage variable access
            return copy.deepcopy(self._active_prod.get_value(variable))

        if stage.status != 'idle':
            raise RuntimeError('Only idle stages can access variables. '
                               'Current status of {} stage: {}'
                               .format(stage.name, stage.status))

        # Check the status of registered stage for this variable
        var_stage = self._stages[self._register[variable]]

        if var_stage.status == 'idle':
            # If not started yet, run it
            self.run_stage(var_stage.name)

        if var_stage.status == 'running':
            # Wait if already running
            var_stage.wait()

        # Done results
        if var_stage.status == 'done':
            return copy.deepcopy(self._active_prod.get_value(variable))
        elif var_stage.status == 'error':
            raise RuntimeError('Variable {} not accessible due to '
                               '{} stage processing error.')
        else:
            raise ValueError('This should be impossible! Stage status: {}'
                             .format(var_stage.status))

    def set_value(self, stage, variable, value):
        """Set a variable value to product."""
        if isinstance(stage, Product):
            stage.set_value(variable, copy.deepcopy(value))
            self._register[variable] = None
            return

        if variable not in self._register.keys():
            raise ValueError('Variable {} not registered.'.format(variable))

        if isinstance(stage, Stage):
            # If a stage instance is passed, get its name for checking
            stage = stage.name

        if stage != self._register[variable]:
            raise ValueError('Stage {} not own {} variable.'
                             .format(stage, variable))

        var_stage = self._stages(stage)
        if var_stage != 'done':
            raise RuntimeError('Only done stages can set variables. '
                               'Current {} stage status: {}'
                               .format(var_stage.name, var_stage.status))

        self._active_prod.set_value(variable, copy.deepcopy(value))

    def get_instrument(self):
        """Get the product instrument."""
        # Return a copy, so it cannot be changed
        return copy.deepcopy(self._active_prod.instrument)

    def register_stage(self, name, stage, disable_variables=[]):
        """Register a stage."""
        if stage in self._stages.values():
            raise ValueError('Stage already registered.')

        if name in self._stages.keys():
            raise ValueError('Stage name {} already in use.'.format(name))

        var = [i for i in stage._provided if i not in disable_variables]
        for i in var:
            self._register[i] = name

        self._stages[name] = stage

    def unregister_stage(self, stage):
        """Remove a stage from the registers."""
        if isinstance(stage, Stage):
            # If a stage instance is passed, get its name for checking
            stage = stage.name

        for i, v in self._register.items():
            if v == stage:
                self._register.popitem(v)

        self._stages.popitem(stage)

    def get_stage_name(self, instance):
        """Get the name of a stage instance."""
        if instance not in self._stages.values():
            if instance.factory == self:
                return 'unregistered_stage'
            else:
                raise ValueError('Stage not associated to '
                                 'this manager.')

        for i, v in self._stages.items():
            if v == instance:
                return i

    def owned_variables(self, name):
        """Return the variables owned by a stage."""
        return [k for k, v in self._register.items() if v == name]

    def activate_product(self, product):
        """Activate a product to this factory."""
        self.reset()
        self.logger.info('Actiavting {} product.'.format(product.name))
        self._active_prod = product

    def reset(self):
        """Cleanup all needed informations from this factory."""
        self.logger.info("Reseting factory")
        self._active_prod = None
        self._active_config = Config()
        for v in self._stages.values():
            v.status = 'idle'
        self._logger = self._manager.logger.getChild('factory')

    def dump_defaults(self):
        """Dump the default configurations in yaml format."""
        conf = {}
        for i, v in self._stages:
            conf[i] = dict(v.config)

        return yaml.dump(conf)

    def run_stage(self, stage):
        """Run a single stage for the active product."""
        if self._active_prod is None:
            raise ValueError("No product has been activated!")

        self.logger.info('Executing {} stage for {} product.'
                         .format(self._stages[stage].name,
                                 self._active_prod.name))

        stage_conf = self._active_config.get(stage, {})
        stage_conf = Config(stage_conf)
        instrument = self._active_prod.instrument

        logger.debug('Freezing instrument and config.')
        stage_conf.freeze()
        instrument.freeze()

        # Execute!
        self._stages[stage]._call_pipeline(instrument, stage_conf)

        logger.debug('Unfreezing instrument and config')
        stage_conf.unfreeze()
        instrument.unfreeze()

    def run(self, config):
        """Run te factory to the product."""

        self._active_config = Config(config)
        targets = self._active_prod.targets
        self.logger.info('Executing pipeline with {} targets.'
                         .format(targets))
        for i in targets:
            self.run_stage(i)

        self._active_config = {}


class Manager(abc.ABC):
    _config = Config()
    _products = IndexedDict()
    _factory = None
    _logger = None

    # TODO: handle configs
    """Class to handle the general pipeline management."""
    def __init__(self):
        self._factory = Factory(self)
        self._config['stages'] = Config()

    @property
    def factory(self):
        return self._factory

    @property
    def logger(self):
        if self._logger is None:
            self._logger = logger.getChild('manager')
            self._logger.setLevel(logger.getEffectiveLevel())
        return self._logger

    @property
    def config(self):
        return copy.deepcopy(self._config)

    @abc.abstractmethod
    def setup_pipeline(self, *args, **kwargs):
        """Setup the pipeline, register stages, etc.

        Can be used to read a config.
        """

    @abc.abstractmethod
    def setup_products(self, *args, **kwargs):
        """Setup products based on specified parameters.

        Examples:
            - Read files from a folder, with filters
            - List files
            - Load a configure file
        """

    def add_product(self, name, product, index=None, requires=[]):
        """Add a new product to this Manager.

        Parameters:
            name : string
                Indentifying name for this product. Uniqueness checked.
            product : `Product`
                Product valid instance.
            index : int
                Insert the product in a given index.
            requires : list(string)
                List of products to be processed first.
        """
        index1 = None

        inds = [self._products.index(r)
                for r in requires]
        if len(inds) > 0:
            index1 = max(inds)

        if index1 is not None and index is not None:
            if index1 < index:
                logger.warn('Cannot insert product {} before its requirements.'
                            ' Overwriting to {}'.format(index, index1))
                index = index1

        if index is None:
            # If indexes are not set, just append
            self._products[name] = product
        else:
            self._products.insert_at(index, name, product)

    def get_product_index(self, name):
        """Return the current index of a given product."""
        for i, k in enumerate(self._products.keys()):
            if k == name:
                return i

        self.logger.warn("Product {} not found.".format(name))

    def get_product_name(self, instance):
        """Return the name of a product based on its instance."""
        if instance not in self._products.values():
            if instance.manager == self:
                return 'unregistered_product'
            else:
                raise ValueError('Product not associated to '
                                 'this manager.')
        for i, v in self._products.items():
            if v == instance:
                return i

        self.logger.warn("No product associated to {} instance."
                         .format(str(instance)))

    def del_product(self, name):
        """Remove and clean a product."""
        try:
            del self._products[name]
        except KeyError:
            logger.debug("Product {} not in this factory.".format(name))
            pass

    def register_stage(self, name, stage, disable_variables=[]):
        """Register a stage"""
        self.factory.register_stage(name, stage, disable_variables)

    def unregister_stage(self, name):
        """Remove a stage from the registers."""
        self.factory.unregister_stage(name)

    def set_value(self, product, variable, value):
        """Set a default value to a variable not owned by a stage."""
        self.factory.set_value(product, variable, value)

    def get_value(self, product, variable):
        """Get a default value from a variable not owned by a stage."""
        return self.factory.get_value()

    def show_products(self):
        """Show the created products in a list."""
        if len(self._products) == 0:
            print('# No products on this manager.')
        else:
            for i, n in enumerate(self._products.keys()):
                print("{}\t{}".format(i, n))

    def run(self, index=None):
        if index is not None:
            if not check_iterable(index):
                index = [index]
        else:
            index = list(range(len(self._products)))

        n = len(index)
        self.logger.info('Processing {} products.'.format(n))

        for i in index:
            self.logger.info("Processing product {} from {}".format(i+1, n))
            name = list(self._products.keys())[i]
            self.factory.activate_product(self._products[name])
            self.factory.run(self.config['stages'])
