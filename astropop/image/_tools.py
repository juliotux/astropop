# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tools for dealing with headers and flags."""

import numpy as np
from astropy.io.fits import Header

from ..logger import logger
from ..framedata._compat import _normalize_and_strip_dict
from ..fits_utils import string_to_header_key

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
        in the list `header_merge_keys`. 'no_merge' will return an empty
        header.
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

    meta = {}
    if method == 'no_merge':
        return meta

    logger.debug('Merging headers with %s strategy.', method)

    if method == 'first':
        return headers[0].copy()

    summary = None
    for hdr in headers:
        hdr, _, _ = _normalize_and_strip_dict(hdr)
        hdr = Header(hdr)
        if summary is None:
            summary = {k: [v] for k, v in hdr.items()}
        for key in hdr.keys():
            if key not in summary.keys():
                # avoid only_equal problems
                summary[key] = [None]
            if hdr[key] not in summary[key]:
                summary[key].append(hdr[key])

    if method == 'selected_keys':
        keys = selected_keys
    else:
        keys = summary.keys()

    for k in keys:
        # do not use np.unique to avoid problems with None
        k = string_to_header_key(k)
        uniq = list(set(summary[k]))
        if len(uniq) == 1:
            meta[k] = uniq[0]
        elif method == 'selected_keys':
            logger.debug('Keyword %s is different across headers. '
                         'Unsing first one.', k)
            meta[k] = summary[k][0]
        else:
            logger.debug('Keyword %s is different across headers. '
                         'Skipping.', k)

    return dict(meta)


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
