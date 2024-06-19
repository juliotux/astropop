# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
import shlex
from astropop.py_utils import string_fix, batch_key_replace, \
                              run_command, check_number, \
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
                                    (np.int16('3'), True),
                                    (np.float64(3), True),
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

    # This test break all the others
    # def test_nested_async(self):
    #     import asyncio
    #     async def async_func():
    #         run_command(['ls', '/'])

    #     asyncio.run(async_func())

    def test_process_error(self):
        import subprocess
        with pytest.raises(subprocess.CalledProcessError):
            run_command('python -c "import sys; sys.exit(1000)"')

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

    def test_broadcast_with_None(self):
        bc = broadcast(np.arange(10), None, 2)

        assert_equal(bc.iters, [np.arange(10), [None]*10, [2]*10])

