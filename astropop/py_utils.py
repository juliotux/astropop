import six

from .logging import log as logger


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
    except Exception as e:
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
