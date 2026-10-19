# Licensed under a 3-clause BSD style license - see LICENSE.rst

import hashlib


__all__ = ['hasher']


def hasher(string, size=8):
    """Return a truncated SHA-256 hexadecimal digest of a stringified value.

    Parameters
    ----------
    string : object
        Value converted with str and encoded as UTF-8 before hashing.
    size : int, optional
        Number of leading hexadecimal characters to return. Default: 8.

    Returns
    -------
    digest : str
        Leading digest characters. Truncated digests can collide.
    """
    string = str(string)
    h = hashlib.sha256(string.encode()).hexdigest()
    return h[:size]
