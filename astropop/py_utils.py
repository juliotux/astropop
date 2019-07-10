# Licensed under a 3-clause BSD style license - see LICENSE.rst

import subprocess
import shlex
import six
from os import path, makedirs
import errno

from .logger import logger

__all__ = ['mkdir_p', 'string_fix', 'process_list', 'check_iterable',
           'batch_key_replace', 'IndexedDict']


def mkdir_p(fname):
    '''
    Function to simulate 'mkdir -p' bash function, with error handling.
    '''
    try:
        makedirs(fname)
    except OSError as exc:
        if exc.errno == errno.EEXIST and path.isdir(fname):
            pass
        else:
            raise exc


def string_fix(string):
    """Fix the byte<-> string problem in python 3"""
    if not isinstance(string, six.string_types):
        if six.PY3:
            try:
                string = str(string, 'utf-8')
            except Exception:
                try:
                    string = str(string, 'latin-1')
                except Exception:
                    string = string
    return string


def process_list(_func, iterator, *args, **kwargs):
    """Run a function func for all i in a iterator list."""
    return [_func(i, *args, **kwargs) for i in iterator]


def check_iterable(value):
    """Check if a value is iterable (list), but not a string."""
    try:
        iter(value)
        if not isinstance(value, six.string_types):
            return True
        else:
            return False
    except TypeError:
        pass

    return False


def batch_key_replace(dictionary, key=None):
    """Scan and replace {key} values in a dictionary by dictionary['key']
    value."""
    if key is None:
        for i in dictionary.keys():
            batch_key_replace(dictionary, i)
        return

    if isinstance(dictionary[key], (six.string_types)):
        for i in dictionary.keys():
            if '{'+i+'}' in dictionary[key]:
                logger.debug("replacing key {} in key {}".format(i, key))
                batch_key_replace(dictionary, i)
                dictionary[key] = dictionary[key].format(**{i: dictionary[i]})
    elif check_iterable(dictionary[key]):
        for j in range(len(dictionary[key])):
            for i in dictionary.keys():
                v = dictionary[key][j]
                if '{'+i+'}' in str(v):
                    logger.debug("replacing key {} in key"
                                 " {}".format(i, key))
                    dictionary[key][j] = v.format(**{i: dictionary[i]})
    else:
        return


def run_command(args, logger=logger):
    """Wrapper to run a command in command line with logging."""
    if isinstance(args, six.string_types):
        args = shlex.shlex(args)

    logger.debug('runing: ' + " ".join(args))

    process = subprocess.Popen(args, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)

    for line in process.stdout:
        line = line.decode('utf-8').strip('\n')
        if line != "":
            logger.debug(line)

    process.wait()

    return process.returncode


class IndexedDict(dict):
    """Extends Python3.7 dictionary to include indexing and inserting.

    Python3.7 keeps assignment ordering in default dict, like OrderedDict.
    """

    def index(self, key):
        """Return the index of a key in the list."""
        __keys = self.keys()

        if key not in __keys:
            raise KeyError("{}".format(k))

        for i, k in enumerate(self.keys()):
            if k == keys:
                return i

    def insert_before(self, key, new_key, val):
        """Insert new_key:value into dict before key"""
        index = self.index(key)
        self.insert_at(index, new_key, val)

    def insert_after(self, key, new_key, val):
        """Insert new_key:value into dict after key"""
        index = self.index(key)
        self.insert_at(index+1, new_key, val)

    def insert_at(self, index, key, value):
        """Insert a key:value to an specific index."""
        __keys = list(self.keys())
        __vals = list(self.values())

        if index < (len(__keys) - 1):
            __keys.insert(index, key)
            __vals.insert(index, value)
            self.clear()
            self.update({x: __vals[i] for i, x in enumerate(__keys)})
        else:
            self.update({new_key: val})
