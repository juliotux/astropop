# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tools for dealing with headers and flags."""

from __future__ import annotations

import typing as _t
import numpy.typing as _npt
from .._typing import HeaderLike as _HeaderLike

import numpy as np
from astropy.io.fits import Header

from ..logger import logger
from ..framedata._compat import _normalize_and_strip_dict

__all__ = ['merge_header', 'merge_flag']


def merge_header(
    *headers: _HeaderLike,
    method: str='same',
    selected_keys: _t.Sequence[str] | None=None,
) -> Header:
    """Merge headers.

    Parameters
    ----------
    headers
        Headers to be merged.
    method
        ``'only_equal'`` keeps keys present with equal values in all
        headers. ``'first'`` copies the first header. ``'selected_keys'`` keeps
        keys named in selected_keys, taking each key's first occurrence.
        ``'no_merge'`` returns an empty header.
    selected_keys
        List of keywords to be merged. Used only if method is
        'selected_keys'.

    Returns
    -------
    header
        Merged header.

    Notes
    -----
    - All headers must have normalized keywords.

    Examples
    --------
    >>> from astropy.io.fits import Header
    >>> from astropop.image._tools import merge_header
    >>> first = Header({'FILTER': 'R', 'EXPTIME': 10})
    >>> second = Header({'FILTER': 'R', 'EXPTIME': 20})

    >>> # Preserve only values that agree in both headers.
    >>> merged = merge_header(first, second, method='only_equal')
    >>> print(dict(merged))
    {'FILTER': 'R'}
    """
    if method not in ['first', 'selected_keys', 'no_merge', 'only_equal']:
        raise ValueError(f'Unknown method {method}.')
    if method == 'selected_keys' and selected_keys is None:
        raise ValueError('selected_keys must be provided if method is '
                         'selected_keys.')

    meta = Header()
    if method == 'no_merge':
        return meta

    logger.debug('Merging headers with %s strategy.', method)

    if method == 'first':
        return headers[0].copy()

    first_hdr = headers[0]
    if method == 'first':
        return first_hdr.copy()

    for hdr in headers:
        hdr, _, _ = _normalize_and_strip_dict(hdr)
        if method == 'only_equal':
            # only keeps equal keys. If is the first header, add it
            # to the meta. If the key is different, remove it from
            # the meta.
            for key in hdr:
                if key not in meta and hdr == first_hdr:
                    meta.append((key, hdr[key], hdr.comments[key]))
                    continue
                if key not in meta:
                    continue
                if meta[key] != hdr[key]:
                    del meta[key]
            # remove all keys that are not in this header, since it will be not
            # equal
            for key in meta:
                if key not in hdr:
                    del meta[key]
        elif method == 'selected_keys':
            # only keeps the keys in the selected_keys list
            for key in hdr:
                if key in selected_keys and key not in meta:
                    meta.append((key, hdr[key], hdr.comments[key]))

    return meta


def merge_flag(
    *flags: _npt.NDArray,
    method: _t.Literal['or', 'and', 'no_merge']='or',
) -> _npt.NDArray:
    """Merge flags from images.

    Parameters
    ----------
    flags
        Flags to be merged.
    method
        ``'or'`` (default) combines flag bits by bitwise OR. ``'and'`` keeps
        only bits present in every input using bitwise AND. ``'no_merge'``
        returns zero flags with the first input's shape and dtype.

    Returns
    -------
    flag
        Merged flag.

    Examples
    --------
    >>> import numpy as np
    >>> from astropop.image._tools import merge_flag
    >>> first = np.array([1, 2], dtype='uint8')
    >>> second = np.array([2, 2], dtype='uint8')
    >>> print(merge_flag(first, second, method='or').tolist())
    [3, 2]
    >>> print(merge_flag(first, second, method='and').tolist())
    [0, 2]
    """
    if method not in ['or', 'and', 'no_merge']:
        raise ValueError(f'Unknown method {method}.')
    if method == 'no_merge':
        return np.zeros_like(flags[0])

    flag = flags[0]
    for f in flags[1:]:
        if method == 'or':
            flag = np.bitwise_or(flag, f)
        elif method == 'and':
            flag = np.bitwise_and(flag, f)

    return flag
