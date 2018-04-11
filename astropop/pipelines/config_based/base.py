# Licensed under a 3-clause BSD style license - see LICENSE.rst
import json
import copy
from collections import OrderedDict

from ...logger import logger
from ...py_utils import batch_key_replace, check_iterable


class ReducePipeline():
    """Simple class to process pipeline scripts with configuration files."""
    # TODO: introduce parameters checking
    def __init__(self, config=None):
        """Dummy entry for the class"""
        self._default_kwargss = OrderedDict()

        if config is not None:
            if isinstance(config, (dict, OrderedDict)):
                self._default_kwargss.update(config)
            elif isinstance(config, str):
                self.load_default_file(config)
            else:
                raise ValueError('Config not supported.')

    def load_default_file(self, filename):
        """Loads a json file with default variables for the reduction.
        If the configuration shares a variable with detaful values, it will be
        overrided."""
        # TODO: make it safer
        j = json.load(open(filename, 'r'))
        self._default_kwargss.update(j)

    def clear(self):
        self._default_kwargss = OrderedDict()

    def process_product(self, filename, dataset='all', raise_error=False,
                        **kwargs):
        """Process the products of a json file.
        If the configuration shares a variable with detaful values, it will be
        overrided."""
        prod_file = json.load(open(filename, 'r'))
        default = copy.copy(self._default_kwargss)
        if '__preload__' in prod_file.keys():
            preload = copy.copy(self._default_kwargss)
            preload.update(prod_file.pop('__preload__'))
            batch_key_replace(preload)
            if 'preload_config_files' in preload.keys():
                files = preload.pop('preload_config_files')
                other = {}
                if check_iterable(files):
                    for f in files:
                        other.update(json.load(open(f, 'r')))
                else:
                    other.update(json.load(open(files, 'r')))
                other.update(preload)
                preload = other
            default.update(preload)

        if dataset not in ['all', None]:
            valid = dataset
        else:
            valid = prod_file.keys()
        for i, v in prod_file.items():
            if i in valid:
                prod = copy.copy(default)
                prod.update(v)
                prod.update(kwargs)
                batch_key_replace(prod)
                logger.info('Reducing {} from {}'.format(i, filename))
                if raise_error:
                    self.run(i, **prod)
                else:
                    try:
                        self.run(i, **prod)
                    except Exception as e:
                        logger.error('Problem in the process of {} product'
                                     ' from {} file. Passing it.'
                                     .format(i, filename) +
                                     '\nError: {}'.format(e))

    def run(self, name, **config):
        """Run a single product. Config is the dictionary of needed
        parameters."""
        raise NotImplementedError('This pipeline is not a valid implemented'
                                  ' pipeline!')

    @property
    def parameters(self):
        '''Print the acceptable parameters of the pipeline.'''
        raise NotImplementedError('This pipeline has no parameters.')

    def pprint(self):
        for i,v in self.parameters.items():
            print('{}: {}'.format(i, v))

    def __call__(self, name, **config):
        return self.run(name, **config)
