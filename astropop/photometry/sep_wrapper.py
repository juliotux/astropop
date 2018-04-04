# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import sep
import sys


def _fix_byte_order(data):
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
    if not data.dtype.isnative:
        data = data.byteswap().newbyteorder()
    if data.dtype.type in [np.uint, np.uintc, np.uint8, np.uint16, np.uint32]:
        data = data.astype(np.intc)
    return data


def find_sources(data, threshold=None, bkg=0.0, rms=None, snr=None, *args,
                 **kwargs):
    data = _fix_byte_order(data)

    if threshold is None:
        if snr is None or rms is None:
            raise ValueError('You must give a threshold or bkg, snr and rms.')
        else:
            return sep.extract(data-bkg, snr, err=rms)

    return sep.extract(data-bkg, threshold)


def calculate_background(data, **kwargs):
    data = _fix_byte_order(data)
    bkg = sep.Background(data, **kwargs)

    return bkg.globalback, bkg.globalrms


def aperture_photometry(data, x, y, r, r_in, r_out, err=None, **kwargs):
    data = _fix_byte_order(data)
    flux, fluxerr, flag = sep.sum_circle(data, x, y, r,
                                         bkgann=(r_in, r_out), err=err,
                                         **kwargs)

    return np.array(list(zip(x, y, flux, fluxerr, flag)),
                    dtype=np.dtype([('x', 'f8'), ('y', 'f8'),
                                    ('flux', flux.dtype),
                                    ('flux_error', fluxerr.dtype),
                                    ('flags', flag.dtype)]))


def psf_photometry(*args, **kwargs):
    raise NotImplementedError("SEP do not have a psf photometry algorithm.")
