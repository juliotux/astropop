# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
from astropop.logger import logger, log_to_list, resolve_level_string


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


@pytest.mark.parametrize('val, res', [('DEBUG', 10),
                                      ('INFO', 20),
                                      ('WARN', 30),
                                      ('WARNING', 30),
                                      ('ERROR', 40),
                                      ('CRITICAL', 50),
                                      ('FATAL', 50),
                                      (50, 50)])
def test_resolve_string(val, res):
    assert resolve_level_string(val) == res


def test_invalid_levels_invalid_string():
    with pytest.raises(AttributeError):
        resolve_level_string('NOT_A_LOGLEVEL')


def test_invalid_levels_none():
    with pytest.raises(TypeError):
        resolve_level_string(None)
