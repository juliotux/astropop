# Licensed under a 3-clause BSD style license - see LICENSE.rst


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
    """
    string = str(string)
    h = hashlib.sha256(string.encode()).hexdigest()
    return h[:size]
