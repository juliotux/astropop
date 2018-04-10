# Licensed under a 3-clause BSD style license - see LICENSE.rst
# Important: this is the default keys of the configuration files!

import os
import glob

from .base import ReducePipeline
from .photometry import PhotometryPipeline
from .polarimetry import PolarimetryPipeline
from .image_processing import CalibPipeline
from ...logger import logger


parameters = PolarimetryPipeline().parameters


class MasterPipeline(ReducePipeline):
    def __init__(self, config=None):
        super(MasterPipeline, self).__init__(config=config)

    @property
    def parameters(self):
        parameters.update(dict(pipeline="Pipeline to use: 'calib', "
                                        "'photometry' or 'polarimetry'"))
        return parameters

    def run(self, name, **config):
        if 'sources' in config.keys() and 'source_ls_pattern' in config.keys():
            logger.warn('sources and sources_ls_pattern given. Using sources.')
        elif 'sources_ls_pattern' in config.keys():
            ls_pat = os.path.join(config['raw_dir'],
                                  config['sources_ls_pattern'])
            fs = glob.glob(ls_pat)
            config['sources'] = [os.path.basename(i) for i in sorted(fs)]
            if len(config['sources']) == 0:
                raise FileNotFoundError("Could not determine sources."
                                        " glob pattern: {}".format(ls_pat))

        config['sources'] = [i for i in config['sources']
                             if i not in config['exclude_images']]

        logger.debug("Product {} config:{}".format(name, str(config)))
        if 'pipeline' not in config.keys():
            raise ValueError('The config must specify what pipeline will be'
                             ' used!')
        if config['pipeline'] == 'photometry':
            p = PhotometryPipeline()
        elif config['pipeline'] == 'polarimetry':
            p = PolarimetryPipeline()
        elif config['pipeline'] == 'calib':
            p = CalibPipeline()
        elif config['pipeline'] == 'lightcurve':
            logger.error('lightcurve pipeline not implemented yet')
        else:
            raise ValueError('Pipeline {} not'
                             ' supported.'.format(config['pipeline']))

        p.run(name, **config)
