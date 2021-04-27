# Licensed under a 3-clause BSD style license - see LICENSE.rst

import sys
import os
import pytest
from astropop.py_utils import mkdir_p, string_fix, process_list, \
                              check_iterable, batch_key_replace, \
                              run_command, IndexedDict, check_number
import numpy as np

from astropop.testing import assert_true, assert_equal, assert_in, \
                             assert_false, assert_is_instance


def test_mkdir(tmpdir):
    p = tmpdir.join('level1/level2').strpath
    mkdir_p(p)
    assert_true(os.path.isdir(p))
    # mkdir a existent dir should not raise error
    mkdir_p(p)


def test_check_number():
    assert_true(check_number(1))
    assert_true(check_number(1.5))
    assert_false(check_number('2'))
    assert_false(check_number('2.5'))
    assert_false(check_number(1+3j))
    assert_false(check_number(5j))
    assert_false(check_number('A'))
    assert_false(check_number('AB'))
    assert_false(check_number([1, 2, 3]))
    assert_false(check_number(np.array([1, 2])))
    assert_true(check_number(np.float(3)))
    assert_true(check_number(np.int('3')))
    assert_false(check_number(False))
    assert_false(check_number((1, 2, 3)))


def test_mkdir_oserror():
    p = '/bin/bash'
    with pytest.raises(OSError):
        mkdir_p(p)


def test_run_command():
    com = ["python", "-c", "print(__import__('sys').version)"]
    stdout = []
    stderr = []
    _, out, err = run_command(com, stdout=stdout, stderr=stderr,
                              stdout_loglevel='WARN')
    for o in out:
        assert_in(o, stdout)
    for e in err:
        assert_in(e, stderr)
    assert_equal('\n'.join(stdout), sys.version)


def test_run_command_string():
    com = "python -c \"print(__import__('sys').version)\""
    stdout = []
    stderr = []
    _, out, err = run_command(com, stdout=stdout, stderr=stderr,
                              stdout_loglevel='WARN')
    for o in out:
        assert_in(o, stdout)
    for e in err:
        assert_in(e, stderr)
    assert_equal('\n'.join(stdout), sys.version)


def test_process_list():
    def dummy_func(i):
        i = 1
        return i
    a = np.zeros(20)
    b = np.ones(20)
    c = process_list(dummy_func, a)
    assert_equal(b, c)
    assert_false(np.array_equal(a, c))


def test_process_list_with_args():
    def dummy_func(i, a, b):
        return (i+a)*b
    i_array = np.arange(20)
    a_val = 2
    b_val = 3
    res = process_list(dummy_func, i_array, a_val, b=b_val)
    assert_equal((i_array+a_val)*b_val, res)


def test_check_iterabel_array():
    a = [1, 2, 3, 4, 5]
    assert_true(check_iterable(a))


def test_check_iterabel_string():
    a = '12345'
    assert_false(check_iterable(a))


def test_check_iterabel_nparray():
    a = np.zeros(20)
    assert_true(check_iterable(a))


def test_check_iterabel_number():
    a = 10
    assert_false(check_iterable(a))


def test_check_iterabel_range():
    a = range(10)
    assert_true(check_iterable(a))


def test_check_iterabel_dict():
    a = dict(a=1, b=2, c=3, d=4)
    assert_true(check_iterable(a))
    assert_true(check_iterable(a.items()))
    assert_true(check_iterable(a.keys()))
    assert_true(check_iterable(a.values()))


def test_batch_key_replace():
    dic1 = {'a': '{b} value', 'b': '6{c}', 'c': 2}
    batch_key_replace(dic1)
    assert_equal(dic1['a'], '62 value')


def test_batch_key_replace_list():
    dic1 = {'a': '{b} value', 'b': ['6{c}', '4{d}'], 'c': 1, 'd': 2}
    batch_key_replace(dic1)
    assert_equal(dic1['a'], "['61', '42'] value")


@pytest.mark.parametrize("inp, enc, res", [("a!^1ö~[😀", None, "a!^1ö~[😀"),
                                           ("a!^1ö~[😀", "utf-8", "a!^1ö~[😀"),
                                           ("a!1[", 'latin-1', "a!1["),
                                           (b'bytes', None, 'bytes'),
                                           (42, None, "42")])
def test_string_fix(inp, enc, res):
    if enc is not None:
        inp = inp.encode(enc)
    assert_equal(string_fix(inp, enc), res)


def test_indexeddict_create():
    d = dict(a=1, b=2, c=3)
    i = IndexedDict(a=1, b=2, c=3)
    assert_is_instance(i, dict)
    assert_equal(len(d), len(i))
    # Python 3.6 and above ensure items order
    assert_equal(list(d.keys()), list(i.keys()))
    assert_equal(list(d.values()), list(i.values()))
    assert_equal(i, d)


def test_indexeddict_insert_at():
    a = IndexedDict(a=1, b=2, c=3, d=4)
    a.insert_at(2, 'e', 5)
    assert_equal(a, {'a': 1, 'b': 2, 'e': 5, 'c': 3, 'd': 4})


def test_indexeddict_insert_at_first():
    a = IndexedDict(a=1, b=2, c=3, d=4)
    a.insert_at(0, 'e', 5)
    assert_equal(a, {'e': 5, 'a': 1, 'b': 2, 'c': 3, 'd': 4})


def test_indexeddict_insert_at_last():
    a = IndexedDict(a=1, b=2, c=3, d=4)
    a.insert_at(4, 'e', 5)
    assert_equal(a, {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5})


def test_indexeddict_insert_at_away():
    a = IndexedDict(a=1, b=2, c=3, d=4)
    a.insert_at(42, 'e', 5)
    assert_equal(a, {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5})


def test_indexeddict_insert_at_negative():
    a = IndexedDict(a=1, b=2, c=3, d=4)
    a.insert_at(-2, 'e', 5)
    assert_equal(a, {'a': 1, 'b': 2, 'c': 3, 'e': 5, 'd': 4})


def test_indexeddict_after():
    a = IndexedDict(a=1, b=2, c=3, d=4)
    a.insert_after('b', 'e', 5)
    assert_equal(a, {'a': 1, 'b': 2, 'e': 5, 'c': 3, 'd': 4})


def test_indexeddict_before():
    a = IndexedDict(a=1, b=2, c=3, d=4)
    a.insert_before('b', 'e', 5)
    assert_equal(a, {'a': 1, 'e': 5, 'b': 2, 'c': 3, 'd': 4})


def test_indexeddict_existing_before_before():
    a = IndexedDict(a=1, b=2, c=3, d=4)
    a.insert_before('b', 'c', 3)
    assert_equal(a, {'a': 1, 'c': 3, 'b': 2, 'd': 4})


def test_indexeddict_existing_after_before():
    a = IndexedDict(a=1, b=2, c=3, d=4, e=5)
    a.insert_before('e', 'c', 4)
    assert_equal(a, {'a': 1, 'b': 2, 'd': 4, 'c': 4, 'e': 5})


def test_indexeddict_existing_before_after():
    a = IndexedDict(a=1, b=2, c=3, d=4)
    a.insert_after('b', 'c', 3)
    assert_equal(a, {'a': 1, 'c': 3, 'b': 2, 'd': 4})


def test_indexeddict_existing_after_after():
    a = IndexedDict(a=1, b=2, c=3, d=4, e=5)
    a.insert_after('e', 'c', 4)
    assert_equal(a, {'a': 1, 'b': 2, 'd': 4, 'c': 4, 'e': 5})


def test_indexeddict_first():
    a = IndexedDict(a=1, b=2, c=3, d=4)
    a.insert_before('a', 'e', 5)
    assert_equal(a, {'e': 5, 'a': 1, 'b': 2, 'c': 3, 'd': 4})


def test_indexeddict_last():
    a = IndexedDict(a=1, b=2, c=3, d=4)
    a.insert_after('d', 'e', 5)
    assert_equal(a, {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5})


@pytest.mark.parametrize('val, res', [('a', 0), ('b', 1), ('c', 2), ('d', 3)])
def test_indexeddict_index(val, res):
    a = IndexedDict(a=1, b=2, c=3, d=4)
    assert_equal(a.index(val), res)


def test_indexeddict_invalid_key():
    a = IndexedDict(a=1, b=2, c=3, d=4)
    with pytest.raises(KeyError):
        a.index('e')
