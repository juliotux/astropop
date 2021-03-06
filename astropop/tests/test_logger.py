# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
from astropop.logger import logger, log_to_list, resolve_level_string, \
                            ListHandler
from astropop.testing import assert_equal, assert_is_instance, assert_in, \
                             assert_not_in


@pytest.mark.parametrize('level, expected', [('WARN', 2), ('INFO', 3),
                                             ('DEBUG', 4)])
def test_logger_list_defaults(level, expected):
    mylog = logger.getChild('testing')
    logs = []
    log_to_list(mylog, logs)
    mylog.setLevel(level)
    mylog.error('Error test')
    mylog.warning('Warning test')
    mylog.info('Info test')
    mylog.debug('Debug test')
    assert_equal(mylog.name, 'astropop.testing')
    assert_equal(len(logs), expected)
    for i, k in zip(['Error test', 'Warning test', 'Info test',
                     'Debug test'][0:expected],
                    logs):
        assert_equal(i, k)


@pytest.mark.parametrize('level, expected', [('WARN', 2), ('INFO', 3),
                                             ('DEBUG', 4)])
def test_logger_list_only_messagens(level, expected):
    mylog = logger.getChild('testing')
    logs = []
    log_to_list(mylog, logs, full_record=False)
    mylog.setLevel(level)
    mylog.error('Error test')
    mylog.warning('Warning test')
    mylog.info('Info test')
    mylog.debug('Debug test')
    assert_equal(mylog.name, 'astropop.testing')
    assert_equal(len(logs), expected)
    for i, k in zip(['Error test', 'Warning test', 'Info test',
                     'Debug test'][0:expected],
                    logs):
        assert_equal(i, k)


@pytest.mark.parametrize('level, expected', [('WARN', 2), ('INFO', 3),
                                             ('DEBUG', 4)])
def test_logger_list_full_record(level, expected):
    mylog = logger.getChild('testing')
    logs = []
    log_to_list(mylog, logs, full_record=True)
    mylog.setLevel(level)
    mylog.error('Error test')
    mylog.warning('Warning test')
    mylog.info('Info test')
    mylog.debug('Debug test')
    assert_equal(mylog.name, 'astropop.testing')
    assert_equal(len(logs), expected)
    for i, k, n in zip(['Error', 'Warning', 'Info',
                        'Debug'][0:expected],
                       logs,
                       [40, 30, 20, 10][0:expected]):
        assert_equal(f'{i} test', k.msg)
        assert_equal(k.name, 'astropop.testing')
        assert_equal(k.levelno, n)
        assert_equal(k.levelname, i.upper())


def test_logger_remove_handler():
    mylog = logger.getChild('testing')
    msg = 'Some error happend here.'
    logs = []
    lh = log_to_list(mylog, logs)
    mylog.setLevel('DEBUG')
    mylog.error(msg)
    assert_is_instance(lh, ListHandler)
    assert_in(lh, mylog.handlers)
    mylog.removeHandler(lh)
    assert_not_in(lh, mylog.handlers)
    assert_equal(logs[0], msg)
    assert_equal(lh.log_list[0], msg)
    assert_equal(lh.log_list, logs)


def test_logger_no_loglist():
    mylog = logger.getChild('testing')
    msg = 'Some error happend here.'
    lh = ListHandler()
    assert_is_instance(lh.log_list, list)
    mylog.addHandler(lh)
    mylog.error(msg)
    assert_equal(lh.log_list[0], msg)


def test_logger_list_debug():
    mylog = logger.getChild('testing')
    logs = []
    log_to_list(mylog, logs)
    mylog.setLevel('DEBUG')
    mylog.warning('Warning test')
    mylog.error('Error test')
    mylog.info('Info test')
    mylog.debug('Debug test')
    assert_equal(mylog.name, 'astropop.testing')
    assert_equal(len(logs), 4)


@pytest.mark.parametrize('val, res', [('DEBUG', 10),
                                      ('INFO', 20),
                                      ('WARN', 30),
                                      ('WARNING', 30),
                                      ('ERROR', 40),
                                      ('CRITICAL', 50),
                                      ('FATAL', 50),
                                      (50, 50)])
def test_resolve_string(val, res):
    assert_equal(resolve_level_string(val), res)


def test_invalid_levels_invalid_string():
    with pytest.raises(AttributeError):
        resolve_level_string('NOT_A_LOGLEVEL')


def test_invalid_levels_none():
    with pytest.raises(TypeError):
        resolve_level_string(None)
