# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
import shlex
from astropop.py_utils import string_fix, process_list, \
                              check_iterable, batch_key_replace, \
                              run_command, IndexedDict, check_number, \
                              broadcast
import numpy as np

from astropop.testing import *
from astropop.logger import logger, log_to_list


@pytest.mark.parametrize('v, exp', [(1, True),
                                    (1.5, True),
                                    ('2', False),
                                    ('2.5', False),
                                    (1+3j, False),
                                    (5j, False),
                                    ('A', False),
                                    ('AB', False),
                                    ([1, 2, 3], False),
                                    (np.array([1, 2]), False),
                                    (np.int('3'), True),
                                    (np.float(3), True),
                                    (False, False),
                                    ((1, 2, 3), False)])
def test_check_number(v, exp):
    if exp:
        assert_true(check_number(v))
    else:
        assert_false(check_number(v))


@pytest.mark.parametrize("inp, enc, res", [("a!^1ö~[😀", None, "a!^1ö~[😀"),
                                           ("a!^1ö~[😀", "utf-8", "a!^1ö~[😀"),
                                           ("a!1[", 'latin-1', "a!1["),
                                           (b'bytes', None, 'bytes'),
                                           (42, None, "42")])
def test_string_fix(inp, enc, res):
    if enc is not None:
        inp = inp.encode(enc)
        assert_equal(string_fix(inp, enc), res)
    else:
        assert_equal(string_fix(inp), res)


class Test_RunCommand():
    com = (["bash", "-c", 'for i in {1..10}; do echo "$i"; sleep 0.1; done'],
           "bash -c 'for i in {1..10}; do echo \"$i\"; sleep 0.1; done'")
    com2 = ("bash -c 'echo \"this is an error\" 1>&2'",
            ["bash", "-c", 'echo "this is an error" 1>&2'])

    @pytest.mark.parametrize('com', com)
    def test_run_command(self, com):
        stdout = []
        stderr = []
        _, out, err = run_command(com, stdout=stdout, stderr=stderr,
                                  stdout_loglevel='WARN')
        assert_is(out, stdout)
        assert_is(err, stderr)
        assert_equal(stdout, [str(i) for i in range(1, 11)])
        assert_equal(stderr, [])

    @pytest.mark.parametrize('com', com2)
    def test_run_command_stderr(self, com):
        stdout = []
        stderr = []
        _, out, err = run_command(com, stdout=stdout, stderr=stderr,
                                  stdout_loglevel='WARN')
        assert_is(out, stdout)
        assert_is(err, stderr)
        assert_equal(stdout, [])
        assert_equal(stderr, ['this is an error'])

    @pytest.mark.parametrize('com', com)
    def test_logging(self, com):
        logger.setLevel('DEBUG')
        logl = []
        expect_log = []
        if not isinstance(com, list):
            com = shlex.split(com)
            # expect_log += ['Converting string using shlex']
        logcmd = com
        logcmd = " ".join(logcmd)
        expect_log += [f"Runing: {logcmd}"]
        expect_log += list(range(1, 11))
        expect_log += [f"Done with process: {logcmd}"]

        lh = log_to_list(logger, logl)
        stdout = []
        stderr = []
        _, out, err = run_command(com, stdout=stdout, stderr=stderr,
                                  stdout_loglevel='WARN')
        assert_is(out, stdout)
        assert_is(err, stderr)
        assert_equal(stdout, [str(i) for i in range(1, 11)])
        assert_equal(stderr, [])
        assert_equal(logl, expect_log)
        logger.removeHandler(lh)

    @pytest.mark.parametrize('com', com2)
    def test_logging_err(self, com):
        logger.setLevel('DEBUG')
        logl = []
        expect_log = []
        if not isinstance(com, list):
            com = shlex.split(com)
            # expect_log += ['Converting string using shlex']
        logcmd = com
        logcmd = " ".join(logcmd)
        expect_log += [f"Runing: {logcmd}"]
        expect_log += ['this is an error']
        expect_log += [f"Done with process: {logcmd}"]

        lh = log_to_list(logger, logl)
        stdout = []
        stderr = []
        _, out, err = run_command(com, stdout=stdout, stderr=stderr,
                                  stdout_loglevel='DEBUG',
                                  stderr_loglevel='ERROR')
        assert_is(out, stdout)
        assert_is(err, stderr)
        assert_equal(stdout, [])
        assert_equal(stderr, ['this is an error'])
        assert_equal(logl, expect_log)
        logger.removeHandler(lh)


class Test_ProcessList():
    def test_process_list(self):
        def dummy_func(i):
            i = 1
            return i
        a = np.zeros(20)
        b = np.ones(20)
        c = process_list(dummy_func, a)
        assert_equal(b, c)
        assert_false(np.array_equal(a, c))

    def test_process_list_with_args(self):
        def dummy_func(i, a, b):
            return (i+a)*b
        i_array = np.arange(20)
        a_val = 2
        b_val = 3
        res = process_list(dummy_func, i_array, a_val, b=b_val)
        assert_equal((i_array+a_val)*b_val, res)


class Test_CheckIterable():
    def test_check_iterabel_array(self):
        a = [1, 2, 3, 4, 5]
        assert_true(check_iterable(a))

    def test_check_iterabel_string(self):
        a = '12345'
        assert_false(check_iterable(a))

    def test_check_iterabel_nparray(self):
        a = np.zeros(20)
        assert_true(check_iterable(a))

    def test_check_iterabel_number(self):
        a = 10
        assert_false(check_iterable(a))

    def test_check_iterabel_range(self):
        a = range(10)
        assert_true(check_iterable(a))

    def test_check_iterabel_dict(self):
        a = dict(a=1, b=2, c=3, d=4)
        assert_true(check_iterable(a))
        assert_true(check_iterable(a.items()))
        assert_true(check_iterable(a.keys()))
        assert_true(check_iterable(a.values()))


class Test_BatchKeyReplace():
    def test_batch_key_replace(self):
        dic1 = {'a': '{b} value', 'b': '6{c}', 'c': 2}
        batch_key_replace(dic1)
        assert_equal(dic1['a'], '62 value')

    def test_batch_key_replace_list(self):
        dic1 = {'a': '{b} value', 'b': ['6{c}', '4{d}'], 'c': 1, 'd': 2}
        batch_key_replace(dic1)
        assert_equal(dic1['a'], "['61', '42'] value")


class Test_Broadcast():
    def test_broadcast(self):
        a = np.arange(10)
        b = np.arange(1, 11)
        c = 2

        bc = broadcast(a, b, c)
        iterator = iter(bc)

        indx = 0
        for i in range(10):
            assert_equal(next(iterator), [a[indx], b[indx], c])
            indx += 1

        with pytest.raises(StopIteration):
            next(iterator)

    def test_broadcast_empty(self):
        with pytest.raises(ValueError):
            broadcast()

    def test_broadcast_wrong_shape(self):
        a = np.arange(10)
        b = np.arange(5)

        with pytest.raises(ValueError):
            broadcast(a, b)

    def test_broadcast_only_scalars(self):
        a = 1
        b = 2
        c = 3

        bc = broadcast(a, b, c)

        for i in bc:
            assert_equal(i, [a, b, c])

    def test_broadcast_superpass_32_limit(self):
        arr = [np.arange(10)]*64
        bc = broadcast(*arr)
        assert_equal(len(bc), 10)
        it = iter(bc)

        for i in range(10):
            assert_equal(next(it), [i]*64)

    def test_broadcast_iters_only_scalars(self):
        bc = broadcast(1, 2, 3)
        assert_equal(bc.iters, [[1], [2], [3]])

    def test_broadcast_iters(self):
        bc = broadcast(np.arange(10), 3, 2)

        assert_equal(bc.iters, [np.arange(10), [3]*10, [2]*10])

class Test_IndexedDict():
    def test_indexeddict_create(self):
        d = dict(a=1, b=2, c=3)
        i = IndexedDict(a=1, b=2, c=3)
        assert_is_instance(i, dict)
        assert_equal(len(d), len(i))
        # Python 3.6 and above ensure items order
        assert_equal(list(d.keys()), list(i.keys()))
        assert_equal(list(d.values()), list(i.values()))
        assert_equal(i, d)

    def test_indexeddict_insert_at(self):
        a = IndexedDict(a=1, b=2, c=3, d=4)
        a.insert_at(2, 'e', 5)
        assert_equal(a, {'a': 1, 'b': 2, 'e': 5, 'c': 3, 'd': 4})

    def test_indexeddict_insert_at_first(self):
        a = IndexedDict(a=1, b=2, c=3, d=4)
        a.insert_at(0, 'e', 5)
        assert_equal(a, {'e': 5, 'a': 1, 'b': 2, 'c': 3, 'd': 4})

    def test_indexeddict_insert_at_last(self):
        a = IndexedDict(a=1, b=2, c=3, d=4)
        a.insert_at(4, 'e', 5)
        assert_equal(a, {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5})

    def test_indexeddict_insert_at_away(self):
        a = IndexedDict(a=1, b=2, c=3, d=4)
        a.insert_at(42, 'e', 5)
        assert_equal(a, {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5})

    def test_indexeddict_insert_at_negative(self):
        a = IndexedDict(a=1, b=2, c=3, d=4)
        a.insert_at(-2, 'e', 5)
        assert_equal(a, {'a': 1, 'b': 2, 'c': 3, 'e': 5, 'd': 4})

    def test_indexeddict_after(self):
        a = IndexedDict(a=1, b=2, c=3, d=4)
        a.insert_after('b', 'e', 5)
        assert_equal(a, {'a': 1, 'b': 2, 'e': 5, 'c': 3, 'd': 4})

    def test_indexeddict_before(self):
        a = IndexedDict(a=1, b=2, c=3, d=4)
        a.insert_before('b', 'e', 5)
        assert_equal(a, {'a': 1, 'e': 5, 'b': 2, 'c': 3, 'd': 4})

    def test_indexeddict_existing_before_before(self):
        a = IndexedDict(a=1, b=2, c=3, d=4)
        a.insert_before('b', 'c', 3)
        assert_equal(a, {'a': 1, 'c': 3, 'b': 2, 'd': 4})

    def test_indexeddict_existing_after_before(self):
        a = IndexedDict(a=1, b=2, c=3, d=4, e=5)
        a.insert_before('e', 'c', 4)
        assert_equal(a, {'a': 1, 'b': 2, 'd': 4, 'c': 4, 'e': 5})

    def test_indexeddict_existing_before_after(self):
        a = IndexedDict(a=1, b=2, c=3, d=4)
        a.insert_after('b', 'c', 3)
        assert_equal(a, {'a': 1, 'c': 3, 'b': 2, 'd': 4})

    def test_indexeddict_existing_after_after(self):
        a = IndexedDict(a=1, b=2, c=3, d=4, e=5)
        a.insert_after('e', 'c', 4)
        assert_equal(a, {'a': 1, 'b': 2, 'd': 4, 'c': 4, 'e': 5})

    def test_indexeddict_first(self):
        a = IndexedDict(a=1, b=2, c=3, d=4)
        a.insert_before('a', 'e', 5)
        assert_equal(a, {'e': 5, 'a': 1, 'b': 2, 'c': 3, 'd': 4})

    def test_indexeddict_last(self):
        a = IndexedDict(a=1, b=2, c=3, d=4)
        a.insert_after('d', 'e', 5)
        assert_equal(a, {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5})

    @pytest.mark.parametrize('val, res', [('a', 0), ('b', 1),
                                          ('c', 2), ('d', 3)])
    def test_indexeddict_index(self, val, res):
        a = IndexedDict(a=1, b=2, c=3, d=4)
        assert_equal(a.index(val), res)

    def test_indexeddict_invalid_key(self):
        a = IndexedDict(a=1, b=2, c=3, d=4)
        with pytest.raises(KeyError):
            a.index('e')
