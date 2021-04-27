# Licensed under a 3-clause BSD style license - see LICENSE.rst

import subprocess  # nosec
import select
import shlex
import six
import errno
from os import path, makedirs
from numbers import Number

from .logger import logger, resolve_level_string

__all__ = ['mkdir_p', 'string_fix', 'process_list', 'check_iterable',
           'batch_key_replace', 'IndexedDict', 'check_number']


def mkdir_p(fname):
    """
    Simulate 'mkdir -p' bash function, with error handling.

    This function wrapps the `~os.makedirs` function to create rescursive
    directories. If the directory alread exists, this function ignores it.
    Any other error in the directory creation is raised.

    Parameters
    ----------
    fname: string or path_like
        Full path of the directory to be created.
    """
    try:
        makedirs(fname)
    except OSError as exc:
        if exc.errno == errno.EEXIST and path.isdir(fname):
            pass
        else:
            raise exc


def process_list(func, iterator, *args, **kwargs):
    """Run a function func for all i in a iterator list.

    The processing will occurr in serial form. The function will be
    processed multiple times using the values iterated from the `iterator`.

    Parameters
    ----------
    func: callable
        Function to be executed with the iterator argument.
    iterator: list
        Values to be passed to the func.
    *args, **kwargs:
        Other, non-varying, arguments to be passed to the function.

    Returns
    -------
    list:
        A list with the results.
    """
    return [func(i, *args, **kwargs) for i in iterator]


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


def check_iterable(value):
    """Check if a value is iterable (list), but not a string.

    The checking process is done by trying the `iter` method of Python and
    matching if the value is a string or not.

    Parametes
    ---------
    value: any
        Value to be checked if it is iterable.

    Returns
    -------
    bool:
        `True` if the value is iterable and not a string. `False` otherwise.
    """
    try:
        iter(value)
        if not isinstance(value, six.string_types):
            return True
        return False
    except TypeError:
        pass

    return False


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

    if isinstance(dictionary[key], (six.string_types)):
        for i in dictionary.keys():
            if '{'+i+'}' in dictionary[key]:
                logger.debug("replacing key %s in key %s", i, key)
                batch_key_replace(dictionary, i)
                dictionary[key] = dictionary[key].format(**{i: dictionary[i]})
    elif check_iterable(dictionary[key]):
        for j in range(len(dictionary[key])):
            for i in dictionary.keys():
                v = dictionary[key][j]
                if '{'+i+'}' in str(v):
                    logger.debug("replacing key %s in key %s", i, key)
                    dictionary[key][j] = v.format(**{i: dictionary[i]})
    else:
        return


def run_command(args, stdout=None, stderr=None, stdout_loglevel='DEBUG',
                stderr_loglevel='ERROR', logger=logger, **kwargs):
    """Run a command in command line with logging."""
    if isinstance(args, six.string_types):
        logger.debug('Converting string using shlex')
        args = shlex.split(args)

    stdout_loglevel = resolve_level_string(stdout_loglevel)
    stderr_loglevel = resolve_level_string(stderr_loglevel)

    logger.log(stdout_loglevel, 'Runing: %s', " ".join(args))

    process = subprocess.Popen(args, stdout=subprocess.PIPE,  # nosec
                               stderr=subprocess.PIPE, **kwargs)

    log_level = {process.stdout: stdout_loglevel,
                 process.stderr: stderr_loglevel}

    store = {process.stdout: stdout,
             process.stderr: stderr}

    def check_io():
        ready_to_read = select.select([process.stdout,
                                       process.stderr],
                                      [], [], 1000)[0]
        for io in ready_to_read:
            line = str(io.readline().decode()).strip('\n')
            if line != "":
                if store[io] is not None:  # only stores the desired io
                    store[io].append(line)
                logger.log(log_level[io], line[:-1])

    # keep checking stdout/stderr until the process exits
    while process.poll() is None:
        check_io()

    check_io()  # check again to catch anything after the process exits

    logger.log(stdout_loglevel, "Done with process: %s", " ".join(args))

    process.wait()
    return process, stdout, stderr


class IndexedDict(dict):
    """Extends Python3.7 dictionary to include indexing and inserting.

    Python3.7 keeps assignment ordering in default dict, like OrderedDict.
    """

    def index(self, key):
        """Return the index of a key in the list."""
        __keys = self.keys()

        if key not in __keys:
            raise KeyError(f"{key}")

        for i, k in enumerate(self.keys()):
            if k == key:
                return i

    def insert_before(self, key, new_key, val):
        """Insert new_key:value into dict before key."""
        index = self.index(key)
        self.insert_at(index, new_key, val)

    def insert_after(self, key, new_key, val):
        """Insert new_key:value into dict after key."""
        index = self.index(key)
        self.insert_at(index+1, new_key, val)

    def insert_at(self, index, key, value):
        """Insert a key:value to an specific index."""
        __keys = list(self.keys())
        __vals = list(self.values())

        # Romeve existing keys
        if key in __keys:
            ind = __keys.index(key)
            __keys.pop(ind)
            __vals.pop(ind)
            if index > ind:
                index = index-1

        if index < (len(__keys) - 1):
            __keys.insert(index, key)
            __vals.insert(index, value)
            self.clear()
            self.update({x: __vals[i] for i, x in enumerate(__keys)})
        else:
            self.update({key: value})
