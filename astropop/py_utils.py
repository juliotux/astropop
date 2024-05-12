# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Small python addons to be used in astropop."""

from os import linesep
import subprocess
import asyncio
import shlex
from numbers import Number

import numpy as np

from .logger import logger, resolve_level_string

__all__ = ['string_fix', 'batch_key_replace', 'check_number',
           'broadcast', 'run_command']


def string_fix(string, encode='utf-8'):
    """
    Fix the byte<->string problem in python 3.

    This method converts anything is passed to it into a string. If it is
    `bytes` also handle the decoding.

    Parameters
    ----------
    string: `str` or `bytes`
        String to be checked or converted.
    encode: string (optional)
        Python compatible `bytes` like encoding.
        Default: 'utf-8'

    Returns
    -------
    string: `str`
        The Python 3 string.
    """
    if isinstance(string, bytes):
        string = string.decode(encode)
    elif not isinstance(string, str):
        # everything is not bytes or string
        string = str(string)

    return string


def check_number(value):
    """Check if a value passed is a scalar number.

    Parameters
    ----------
    value: any
        Value to be checked.

    Returns
    -------
    bool:
        `True` if the value is a scalar number (not string, list or array) and
        `False` otherwise.
    """
    if isinstance(value, Number):
        if not isinstance(value, (bool, complex)):
            return True
    return False


class _scalar_iterator:
    """Iterator for scalar values."""
    def __init__(self, value, length):
        self.value = value
        self.length = length

    def __iter__(self):
        for i in range(self.length):
            yield self.value

    def __getitem__(self, index):
        if index >= self.length or index < 0:
            raise IndexError
        return self.value

    def __len__(self):
        return self.length


class broadcast:
    """Broadcast values to the list size. Alternative to `~numpy.broadcast`."""

    def __init__(self, *args):
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

    def __iter__(self):
        """Return the iterator."""
        for i in zip(*self.iters):
            yield i

    @staticmethod
    def _get_length(value):
        """Get the length of iterable only values."""
        if value is None:
            return -1
        if np.isscalar(value):
            return -1
        return len(value)

    @property
    def iters(self):
        """Return the tuple containing the iterators."""
        return self.args

    def __len__(self):
        """Return the length of the broadcast."""
        return self.length


def batch_key_replace(dictionary, key=None):
    """Scan and replace {key} values in a dict by dict['key'] value.

    Notes
    -----
    All the replacement is did inplace. Nothing is returned.

    Parameters
    ----------
    dictionary: dict_like
        Dictionary to replace the keys.
    key: string or None, optional
        The key to be replaced.
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


async def _read_stream(stream, callback):
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


async def _subprocess(args, stdout, stderr, stdout_loglevel, stderr_loglevel,
                      logger, **kwargs):
    """Execute subprocesses."""
    def proccess_out(line, std_l, loglevel):
        line = line.decode("utf-8").strip('\n')
        std_l.append(line)
        logger.log(loglevel, line)

    proc = await asyncio.create_subprocess_shell(
        shlex.join(args),
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


def _run_async_task(task):
    """Run async task and avoid problems with Jupyter."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

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


def run_command(args, stdout=None, stderr=None, stdout_loglevel='DEBUG',
                stderr_loglevel='ERROR', logger=logger, **kwargs):
    """Run a command in command line with logging.

    Parameters
    ----------
    args: list of strings or string
        Full command, with arguments, to be executed.
    stdout: list (optional)
        List to store the stdout of the command. If None, a new list will be
        created.
    stderr: list (optional)
        List to store the stderr of the command. If None, a new list will be
        created.
    stdout_loglevel: string (optional)
        Log level to print the stdout lines. Default is 'DEBUG'.
    stderr_loglevel: string (optional)
        Log level to print the stderr lines. Default is 'ERROR'.
    logger: `~logging.Logger` (optional)
        Custom logger to print the outputs.
    **kwargs: dict (optional)
        Additional arguments to be passed to `~asyncio.create_subprocess_shell`

    Returns
    -------
        `~subprocess.CompletedProcess` results of the execution.
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
