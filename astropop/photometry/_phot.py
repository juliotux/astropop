import numpy as np
from astropy.table import vstack, Table, hstack

from ..fits_utils import check_hdu
from ..py_utils import check_iterable
from .aperture import aperture_photometry
from .detection import (background, starfind, sexfind, calc_fwhm,
                        sources_mask)
from ..logger import logger


def process_photometry(image, photometry_type, detect_fwhm=None,
                       detect_snr=None, box_size=30,
                       r='auto', r_ann='auto', psf_model='gaussian',
                       psf_niters=3, x=None, y=None, mask=None,
                       gain=None, readnoise=None):
    """Process standart photometry in one image, without calibrations."""
    image = check_hdu(image)
    data = image.data
    bkg, rms = background(data, box_size=64,
                          filter_size=3, mask=mask,
                          global_bkg=False)
    # if readnoise is not None:
    #     rms = readnoise
    # else:
    #     rms = np.median(rms)
    rms = np.median(rms)

    if x is None or y is None:
        # First, we find the sources with sep, compute the FWHM and after
        # refine the point sources with daofind
        # Prefer using readnoise
        sources = sexfind(data, detect_snr, bkg, rms, mask=mask,
                           fwhm=detect_fwhm, segmentation_map=False)
        # make a better identification
        sources = starfind(data, detect_snr, bkg, rms, mask=mask,
                           fwhm=detect_fwhm, box_size=box_size,
                           sharp_limit=(0.1, 2.0), round_limit=(-2.0, 2.0))

        x = sources['x']
        y = sources['y']

    result = Table()
    ind = Table()
    ind['star_index'] = np.arange(len(x))
    if photometry_type in ('aperture', 'both'):
        # process multiple apertures
        if not check_iterable(r):
            r = [r]
        for ri in r:
            ap = aperture_photometry(data, x, y, r=ri, r_ann=r_ann,
                                     gain=gain, readnoise=rms, mask=mask)
            ap = hstack([ind, ap])
            result = vstack([result, ap])

    if photometry_type in ('psf', 'both'):
        raise NotImplementedError()
        # result['aperture'] = ['psf']*len(x)
        # result = vstack([results, ap])

    return result
