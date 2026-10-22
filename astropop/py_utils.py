# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Small python addons to be used in astropop."""

from __future__ import annotations

import typing as _t
import logging as _logging


from os import linesep
import subprocess
import asyncio
import shlex
import sys
from numbers import Number

import numpy as np

from .logger import logger, resolve_level_string

_T = _t.TypeVar('_T')

__all__ = ['string_fix', 'batch_key_replace', 'check_number',
           'broadcast', 'run_command']


def string_fix(string: object, encode: str='utf-8') -> str:
    """
    Fix the byte<->string problem in python 3.

    This method converts anything is passed to it into a string. If it is
    `bytes` also handle the decoding.

    Parameters
    ----------
    string
        String to be checked or converted.
    encode
        Python compatible `bytes` like encoding.
        Default: 'utf-8'

    Returns
    -------
    string
        The Python 3 string.

    Examples
    --------
    >>> from astropop.py_utils import string_fix
    >>> print(string_fix(b'object name'))
    object name
    >>> print(string_fix(42))
    42
    """
    if isinstance(string, bytes):
        string = string.decode(encode)
    elif not isinstance(string, str):
        # everything is not bytes or string
        string = str(string)

    return string


def check_number(value: object) -> bool:
    """Check if a value passed is a scalar number.

    Parameters
    ----------
    value
        Value to be checked.

    Returns
    -------
    bool
        `True` if the value is a scalar number (not string, list or array) and
        `False` otherwise.

    Examples
    --------
    >>> from astropop.py_utils import check_number

    >>> # Boolean values and sequences do not count as scalar numbers here.
    >>> print(check_number(1.5), check_number(True), check_number([1.5]))
    True False False
    """
    if isinstance(value, Number):
        if not isinstance(value, (bool, complex)):
            return True
    return False


class _scalar_iterator(_t.Generic[_T]):
    """Iterator for scalar values."""
    def __init__(self, value: _T, length: int) -> None:
        """Retain a scalar and the number of times it should be exposed."""
        self.value = value
        self.length = length

    def __iter__(self) -> _t.Iterator[_T]:
        """Yield the same scalar the configured number of times."""
        for i in range(self.length):
            yield self.value

    def __getitem__(self, index: int) -> _T:
        """Return the scalar for a valid nonnegative index, otherwise raise IndexError."""
        if index >= self.length or index < 0:
            raise IndexError
        return self.value

    def __len__(self) -> int:
        """Return the configured repetition count."""
        return self.length


class broadcast:
    """Broadcast values to the list size. Alternative to `~numpy.broadcast`.

    Examples
    --------
    >>> from astropop.py_utils import broadcast

    >>> # Repeat scalar values while advancing through the coordinate list.
    >>> print(list(broadcast([1, 2, 3], 5, None)))
    [(1, 5, None), (2, 5, None), (3, 5, None)]
    """

    def __init__(self, *args: object) -> None:
        """Initialize the broadcast object."""
        if len(args) == 0:
            raise ValueError("Empty broadcast")

        lengths = np.array([self._get_length(a) for a in args])
        if np.all(lengths == -1):
            self.length = 1
        else:
            self.length = [i for i in lengths if i >= 0]
            if len(set(self.length)) > 1:
                raise ValueError("All array arguments must have the same "
                                 "length.")
            self.length = self.length[0]
        self.args = [a if lengths[i] >= 0 else _scalar_iterator(a, self.length)
                     for i, a in enumerate(args)]

    def __iter__(self) -> _t.Iterator[tuple[object, ...]]:
        """Return the iterator."""
        for i in zip(*self.iters):
            yield i

    @staticmethod
    def _get_length(value: object) -> int:
        """Get the length of iterable only values."""
        if value is None:
            return -1
        if np.isscalar(value):
            return -1
        return len(value)

    @property
    def iters(self) -> list[_t.Iterable[object]]:
        """Return the tuple containing the iterators."""
        return self.args

    def __len__(self) -> int:
        """Return the length of the broadcast."""
        return self.length


def batch_key_replace(
    dictionary: _t.MutableMapping[str, _t.Any],
    key: str | None=None,
) -> None:
    """Scan and replace {key} values in a dict by dict['key'] value.

    Notes
    -----
    All the replacement is did inplace. Nothing is returned.

    Parameters
    ----------
    dictionary
        Dictionary to replace the keys.
    key
        The key to be replaced.

    Examples
    --------
    >>> from astropop.py_utils import batch_key_replace
    >>> settings = {'root': '/data', 'file': '{root}/image.fits'}

    >>> # Resolve references in place, preserving unrelated dictionary entries.
    >>> batch_key_replace(settings)
    >>> print(settings['file'])
    /data/image.fits
    """
    if key is None:
        for i in dictionary.keys():
            batch_key_replace(dictionary, i)
        return

    if isinstance(dictionary[key], (str, bytes)):
        for i in dictionary.keys():
            if '{'+i+'}' in dictionary[key]:
                logger.debug("replacing key %s in key %s", i, key)
                batch_key_replace(dictionary, i)
                dictionary[key] = dictionary[key].format(**{i: dictionary[i]})
    elif not np.isscalar(dictionary[key]):
        for j in range(len(dictionary[key])):
            for i in dictionary.keys():
                v = dictionary[key][j]
                if '{'+i+'}' in str(v):
                    logger.debug("replacing key %s in key %s", i, key)
                    dictionary[key][j] = v.format(**{i: dictionary[i]})
    else:
        return


async def _read_stream(
    stream: asyncio.StreamReader,
    callback: _t.Callable[[bytes], object],
) -> None:
    """Read stream buffers."""
    # TODO: when deprcate py37
    # while (line := await stream.readline()):
    #     callback(line)
    while True:
        line = await stream.readline()
        if line:
            callback(line)
        else:
            break


async def _subprocess(
    args: _t.Sequence[str],
    stdout: list[str],
    stderr: list[str],
    stdout_loglevel: int,
    stderr_loglevel: int,
    logger: _logging.Logger,
    **kwargs: _t.Any,
) -> subprocess.CompletedProcess[str]:
    """Execute subprocesses."""
    def proccess_out(line: bytes, std_l: list[str], loglevel: int) -> None:
        """Decode a UTF-8 output line, remove its line ending, store it and log it."""
        line = line.decode("utf-8").rstrip('\r\n')
        std_l.append(line)
        logger.log(loglevel, line)

    if sys.platform == 'win32':
        # shlex.join uses POSIX quoting, which cmd.exe treats literally.
        # Pass argv directly so paths and shell metacharacters stay intact.
        create_process = asyncio.create_subprocess_exec
        command = args
    else:
        create_process = asyncio.create_subprocess_shell
        command = [shlex.join(args)]

    proc = await create_process(
        *command,
        limit=2**23,  # 8 MB
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **kwargs)

    tasks = []
    loop = asyncio.get_running_loop()
    if proc.stdout:
        t = loop.create_task(_read_stream(proc.stdout,
                                          lambda x:
                                          proccess_out(x, stdout,
                                                       stdout_loglevel)))
        tasks.append(t)
    if proc.stderr:
        t = loop.create_task(_read_stream(proc.stderr,
                                          lambda x:
                                          proccess_out(x, stderr,
                                                       stderr_loglevel)))
        tasks.append(t)
    await asyncio.wait(set(tasks))

    return subprocess.CompletedProcess(args=args,
                                       returncode=await proc.wait(),
                                       stdout=linesep.join(stdout) + linesep,
                                       stderr=linesep.join(stderr) + linesep)


def _run_async_task(task: _t.Coroutine[object, object, _T]) -> _T:
    """Run async task and avoid problems with Jupyter."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(task)

    # patch asyncio when running inside Jupyter or other running loop
    if loop and loop.is_running():
        try:
            import nest_asyncio
        except ImportError:
            raise ImportError('To run this command inside a running async '
                              'loop, like Jupyter Notebook, you need to '
                              'install `nest-asyncio` package. ')
        nest_asyncio.apply()

    task = asyncio.ensure_future(task)
    return loop.run_until_complete(task)


def run_command(
    args: str | _t.Sequence[str],
    stdout: list[str] | None=None,
    stderr: list[str] | None=None,
    stdout_loglevel: str | int='DEBUG',
    stderr_loglevel: str | int='ERROR',
    logger: _logging.Logger=logger,
    **kwargs: _t.Any,
) -> tuple[subprocess.CompletedProcess[str], list[str], list[str]]:
    """Run a command in command line with logging.

    Parameters
    ----------
    args
        Full command, with arguments, to be executed.
    stdout
        List to store the stdout of the command. If None, a new list will be
        created.
    stderr
        List to store the stderr of the command. If None, a new list will be
        created.
    stdout_loglevel
        Log level to print the stdout lines. Default is 'DEBUG'.
    stderr_loglevel
        Log level to print the stderr lines. Default is 'ERROR'.
    logger
        Custom logger to print the outputs.
    **kwargs
        Additional arguments passed to `~asyncio.create_subprocess_exec` on
        Windows or `~asyncio.create_subprocess_shell` on other platforms.

    Returns
    -------
        `~subprocess.CompletedProcess` results of the execution.

    Examples
    --------
    >>> import sys
    >>> from astropop.py_utils import run_command

    >>> # Use this interpreter and an argument list instead of platform shell syntax.
    >>> completed, stdout, stderr = run_command([sys.executable, '-c', 'print(42)'])
    >>> print(completed.returncode, stdout, stderr)
    0 ['42'] []
    """
    # Put the cmd in python list, required
    if isinstance(args, (str, bytes)):
        logger.debug('Converting string using shlex')
        args = shlex.split(args)

    stdout_loglevel = resolve_level_string(stdout_loglevel)
    stderr_loglevel = resolve_level_string(stderr_loglevel)
    if stdout is None:
        stdout = []
    if stderr is None:
        stderr = []

    logger.info('Runing: %s', " ".join(args))
    # Run the command
    loc_logger = logger
    proc = _subprocess(args, stdout, stderr, stdout_loglevel, stderr_loglevel,
                       logger=loc_logger, **kwargs)

    result = _run_async_task(proc)
    # restore original args to mimic subproces.run()
    result.args = args
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, args,
                                            output=result.stdout,
                                            stderr=result.stderr)
    logger.info("Done with process: %s", " ".join(args))

    return result, stdout, stderr
