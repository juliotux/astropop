# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tools for dealing with headers and flags."""

import numpy as np
from astropy.io.fits import Header

from ..logger import logger
from ..framedata._compat import _normalize_and_strip_dict

__all__ = ['merge_header', 'merge_flag']


def merge_header(*headers, method='same', selected_keys=None):
    """Merge headers.

    Parameters
    ----------
    headers: list of `astropy.io.fits.Header`
        Headers to be merged.
    method: {'only_equal', 'first', 'selected_keys', 'no_merge'}
        Method to merge the headers. 'only_equal' will merge only the keywords
        with the same value in all headers. 'first' will use the first
        header as the result. 'selected_keys' will merge only the keywords
        in the list `header_merge_keys`, prioritizing the first appearence.
        'no_merge' will return an empty header.
    selected_keys: list of str
        List of keywords to be merged. Used only if method is
        'selected_keys'.

    Returns
    -------
    header: `dict`
        Merged header.

    Notes
    -----
    - All headers must have normalized keywords.
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


def merge_flag(*flags, method='or'):
    """Merge flags from images.

    Parameters
    ----------
    flags: list of `numpy.ndarray`
        Flags to be merged.
    method: {'or', 'and', 'no_merge'}
        Method to merge the flags. 'or' will merge using the logical or
        operation. 'and' will merge using the logical and operation.
        'sum' will merge using the sum operation.

    Returns
    -------
    flag: `numpy.ndarray`
        Merged flag.
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
