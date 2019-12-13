# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
from astropop.logger import logger, log_to_list, resolve_level_string, \
                            ListHandler


@pytest.mark.parametrize('level, expected', [('WARN', 2), ('INFO', 3),
                                             ('DEBUG', 4)])
def test_logger_list_defaults(level, expected):
    l = logger.getChild('testing')
    logs = []
    log_to_list(l, logs)
    l.setLevel(level)
    l.error('Error test')
    l.warn('Warning test')
    l.info('Info test')
    l.debug('Debug test')
    assert l.name == 'astropop.testing'
    assert len(logs) == expected
    for i, k in zip(['Error test', 'Warning test', 'Info test',
                     'Debug test'][0:expected],
                    logs):
        assert i == k


@pytest.mark.parametrize('level, expected', [('WARN', 2), ('INFO', 3),
                                             ('DEBUG', 4)])
def test_logger_list_only_messagens(level, expected):
    l = logger.getChild('testing')
    logs = []
    log_to_list(l, logs, full_record=False)
    l.setLevel(level)
    l.error('Error test')
    l.warn('Warning test')
    l.info('Info test')
    l.debug('Debug test')
    assert l.name == 'astropop.testing'
    assert len(logs) == expected
    for i, k in zip(['Error test', 'Warning test', 'Info test',
                     'Debug test'][0:expected],
                    logs):
        assert i == k


@pytest.mark.parametrize('level, expected', [('WARN', 2), ('INFO', 3),
                                             ('DEBUG', 4)])
def test_logger_list_full_record(level, expected):
    l = logger.getChild('testing')
    logs = []
    log_to_list(l, logs, full_record=True)
    l.setLevel(level)
    l.error('Error test')
    l.warn('Warning test')
    l.info('Info test')
    l.debug('Debug test')
    assert l.name == 'astropop.testing'
    assert len(logs) == expected
    for i, k, n in zip(['Error', 'Warning', 'Info',
                        'Debug'][0:expected],
                       logs,
                       [40, 30, 20, 10][0:expected]):
        assert f'{i} test' == k.msg
        assert k.name == 'astropop.testing'
        assert k.levelno == n
        assert k.levelname == i.upper()


def test_logger_remove_handler():
    l = logger.getChild('testing')
    msg = 'Some error happend here.'
    logs = []
    lh = log_to_list(l, logs)
    l.setLevel('DEBUG')
    l.error(msg)
    assert isinstance(lh, ListHandler)
    assert lh in l.handlers
    l.removeHandler(lh)
    assert lh not in l.handlers
    assert logs[0] == msg
    assert lh.log_list[0] == msg
    assert lh.log_list == logs


def test_logger_no_loglist():
    l = logger.getChild('testing')
    msg = 'Some error happend here.'
    lh = ListHandler()
    assert isinstance(lh.log_list, list)
    l.addHandler(lh)
    l.error(msg)
    assert lh.log_list[0] == msg


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
