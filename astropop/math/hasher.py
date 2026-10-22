# Licensed under a 3-clause BSD style license - see LICENSE.rst


"""Generate short identifiers from SHA-256 hashes."""
from __future__ import annotations

import hashlib


__all__ = ['hasher']


def hasher(string: object, size: int=8) -> str:
    """Return a truncated SHA-256 hexadecimal digest of a stringified value.

    Parameters
    ----------
    string
        Value converted with str and encoded as UTF-8 before hashing.
    size
        Number of leading hexadecimal characters to return. Default: 8.

    Returns
    -------
    digest
        Leading digest characters. Truncated digests can collide.

    Examples
    --------
    >>> from astropop.math.hasher import hasher
    >>> print(hasher('abc'))
    ba7816bf
    >>> print(hasher('abc', size=4))
    ba78
    """
    string = str(string)
    h = hashlib.sha256(string.encode()).hexdigest()
    return h[:size]
