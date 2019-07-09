# Licensed under a 3-clause BSD style license - see LICENSE.rst

import hashlib


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
    h = hashlib(str(string, "utf-8").encode()).hexdigest()
    return h[:size]
