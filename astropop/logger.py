# Licensed under a 3-clause BSD style license - see LICENSE.rst

import logging


__all__ = ['logger', 'ListHandler', 'log_to_list']

        
logging.basicConfig(format='%(asctime)-15s %(name)s - %(levelname)s -'
                           ' %(message)s  [%(module)s]')
logger = logging.getLogger('astropop')
logger.setLevel('INFO')


class ListHandler(logging.Handler):
    """Logging handler to save messages in a list. No thread safe!"""
    def __init__(self, log_list):
        logging.Handler.__init__(self)
        self.log_list = log_list

    def emit(self, record):
        self.log_list.append(record.msg)


def log_to_list(logger, log_list):
    """Add a ListHandler and a log_list to a Logger."""
    lh = ListHandler(log_list)
    logger.addHandler(lh)
