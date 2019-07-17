# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropop.logger import logger, log_to_list


def test_logger_list():
    l = logger.getChild('testing')
    logs = []
    log_to_list(l, logs)
    l.setLevel('INFO')
    l.warn('Warning test')
    l.error('Error test')
    l.info('Info test')
    l.debug('Debug test')
    assert l.name == 'astropop.testing'
    assert len(logs) == 3


def test_logger_list_debug():
    l = logger.getChild('testing')
    logs = []
    log_to_list(l, logs)
    l.setLevel('DEBUG')
    l.warn('Warning test')
    l.error('Error test')
    l.info('Info test')
    l.debug('Debug test')
    assert l.name == 'astropop.testing'
    assert len(logs) == 4
