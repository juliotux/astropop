# Licensed under a 3-clause BSD style license - see LICENSE.rst

import sys
import os
import pytest
import pytest_check as check
from astropop.py_utils import mkdir_p, string_fix, process_list, \
                              check_iterable, batch_key_replace, \
                              run_command, IndexedDict
import numpy as np
import numpy.testing as npt


def test_mkdir(tmpdir):
    p = tmpdir.join('level1/level2').strpath
    mkdir_p(p)
    check.is_true(os.path.isdir(p))
    # mkdir a existent dir should not raise error
    mkdir_p(p)


def test_mkdir_oserror(tmpdir):
    p = '/bin/bash'
    with pytest.raises(OSError):
        mkdir_p(p)


def test_run_command():
    com = ["python", "-c", "print(__import__('sys').version)"]
    stdout = []
    stderr = []
    res, out, err = run_command(com, stdout=stdout, stderr=stderr,
                                stdout_loglevel='WARN')
    for o in out:
        check.is_in(o, stdout)
    for e in err:
        check.is_in(e, stderr)
    check.equal('\n'.join(stdout), sys.version)


def test_run_command_string():
    com = "python -c \"print(__import__('sys').version)\""
    stdout = []
    stderr = []
    res, out, err = run_command(com, stdout=stdout, stderr=stderr,
                                stdout_loglevel='WARN')
    for o in out:
        check.is_in(o, stdout)
    for e in err:
        check.is_in(e, stderr)
    check.equal('\n'.join(stdout), sys.version)


def test_process_list():
    def dummy_func(i):
        return 1
    a = np.zeros(20)
    b = np.ones(20)
    c = process_list(dummy_func, a)
    npt.assert_array_equal(b, c)
    check.is_false(np.array_equal(a, c))


def test_process_list_with_args():
    def dummy_func(i, a, b):
        return (i+a)*b
    i_array = np.arange(20)
    a_val = 2
    b_val = 3
    res = process_list(dummy_func, i_array, a_val, b=b_val)
    npt.assert_array_equal((i_array+a_val)*b_val, res)


def test_check_iterabel_array():
    a = [1, 2, 3, 4, 5]
    check.is_true(check_iterable(a))


def test_check_iterabel_string():
    a = '12345'
    check.is_false(check_iterable(a))


def test_check_iterabel_nparray():
    a = np.zeros(20)
    check.is_true(check_iterable(a))


def test_check_iterabel_number():
    a = 10
    check.is_false(check_iterable(a))


def test_check_iterabel_range():
    a = range(10)
    check.is_true(check_iterable(a))


def test_check_iterabel_dict():
    a = dict(a=1, b=2, c=3, d=4)
    check.is_true(check_iterable(a))
    check.is_true(check_iterable(a.items()))
    check.is_true(check_iterable(a.keys()))
    check.is_true(check_iterable(a.values()))


def test_batch_key_replace():
    dic1 = {'a': '{b} value', 'b': '6{c}', 'c': 2}
    batch_key_replace(dic1)
    check.equal(dic1['a'], '62 value')


def test_batch_key_replace_list():
    dic1 = {'a': '{b} value', 'b': ['6{c}', '4{d}'], 'c': 1, 'd': 2}
    batch_key_replace(dic1)
    check.equal(dic1['a'], "['61', '42'] value")


@pytest.mark.parametrize("inp, enc, res", [("a!^1ö~[😀", None, "a!^1ö~[😀"),
                                           ("a!^1ö~[😀", "utf-8", "a!^1ö~[😀"),
                                           ("a!1[", 'latin-1', "a!1["),
                                           (b'bytes', None, 'bytes'),
                                           (42, None, "42")])
def test_string_fix(inp, enc, res):
    if enc is not None:
        inp = inp.encode(enc)
    check.equal(string_fix(inp, enc), res)


def test_indexeddict_create():
    d = dict(a=1, b=2, c=3)
    i = IndexedDict(a=1, b=2, c=3)
    check.is_instance(i, dict)
    check.equal(len(d), len(i))
    # Python 3.6 and above ensure items order
    check.equal(list(d.keys()), list(i.keys()))
    check.equal(list(d.values()), list(i.values()))
    check.equal(i, d)


def test_indexeddict_insert_at():
    a = IndexedDict(a=1, b=2, c=3, d=4)
    a.insert_at(2, 'e', 5)
    check.equal(a, {'a': 1, 'b': 2, 'e': 5, 'c': 3, 'd': 4})


def test_indexeddict_insert_at_first():
    a = IndexedDict(a=1, b=2, c=3, d=4)
    a.insert_at(0, 'e', 5)
    check.equal(a, {'e': 5, 'a': 1, 'b': 2, 'c': 3, 'd': 4})


def test_indexeddict_insert_at_last():
    a = IndexedDict(a=1, b=2, c=3, d=4)
    a.insert_at(4, 'e', 5)
    check.equal(a, {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5})


def test_indexeddict_insert_at_away():
    a = IndexedDict(a=1, b=2, c=3, d=4)
    a.insert_at(42, 'e', 5)
    check.equal(a, {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5})


def test_indexeddict_insert_at_negative():
    a = IndexedDict(a=1, b=2, c=3, d=4)
    a.insert_at(-2, 'e', 5)
    check.equal(a, {'a': 1, 'b': 2, 'c': 3, 'e': 5, 'd': 4})


def test_indexeddict_after():
    a = IndexedDict(a=1, b=2, c=3, d=4)
    a.insert_after('b', 'e', 5)
    check.equal(a, {'a': 1, 'b': 2, 'e': 5, 'c': 3, 'd': 4})


def test_indexeddict_before():
    a = IndexedDict(a=1, b=2, c=3, d=4)
    a.insert_before('b', 'e', 5)
    check.equal(a, {'a': 1, 'e': 5, 'b': 2, 'c': 3, 'd': 4})


def test_indexeddict_existing_before_before():
    a = IndexedDict(a=1, b=2, c=3, d=4)
    a.insert_before('b', 'c', 3)
    check.equal(a, {'a': 1, 'c': 3, 'b': 2, 'd': 4})


def test_indexeddict_existing_after_before():
    a = IndexedDict(a=1, b=2, c=3, d=4, e=5)
    a.insert_before('e', 'c', 4)
    check.equal(a, {'a': 1, 'b': 2, 'd': 4, 'c': 4, 'e': 5})


def test_indexeddict_existing_before_after():
    a = IndexedDict(a=1, b=2, c=3, d=4)
    a.insert_after('b', 'c', 3)
    check.equal(a, {'a': 1, 'c': 3, 'b': 2, 'd': 4})


def test_indexeddict_existing_after_after():
    a = IndexedDict(a=1, b=2, c=3, d=4, e=5)
    a.insert_after('e', 'c', 4)
    check.equal(a, {'a': 1, 'b': 2, 'd': 4, 'c': 4, 'e': 5})


def test_indexeddict_first():
    a = IndexedDict(a=1, b=2, c=3, d=4)
    a.insert_before('a', 'e', 5)
    check.equal(a, {'e': 5, 'a': 1, 'b': 2, 'c': 3, 'd': 4})


def test_indexeddict_last():
    a = IndexedDict(a=1, b=2, c=3, d=4)
    a.insert_after('d', 'e', 5)
    check.equal(a, {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5})


@pytest.mark.parametrize('val, res', [('a', 0), ('b', 1), ('c', 2), ('d', 3)])
def test_indexeddict_index(val, res):
    a = IndexedDict(a=1, b=2, c=3, d=4)
    check.equal(a.index(val), res)


def test_indexeddict_invalid_key():
    a = IndexedDict(a=1, b=2, c=3, d=4)
    with pytest.raises(KeyError):
        a.index('e')
