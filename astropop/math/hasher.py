# Licensed under a 3-clause BSD style license - see LICENSE.rst

import hashlib


__all__ = ['hasher']


def hasher(string, size=8):
    """Simple function to generate a SHA1 hash of a string.

    Parameters:
        - string : string or bytes
            The string to be hashed.
        - size : int
            Size of the output hash string.

    Returns:
        - h : string
            Hash string trunked to size.
    """
    string = str(string)
    h = hashlib.sha256(string.encode()).hexdigest()
    return h[:size]
