from astropy.stats import gaussian_sigma_to_fwhm

from ..fits_utils import check_hdu
from .aperture import aperture_photometry
from .detection import (background, starfind, sexfind, calc_fwhm,
                        sources_mask)


def process_photometry(image, photometry_type, detect_fwhm=None,
                       detect_snr=None, box_size=30,
                       r='auto', r_ann='auto', psf_model='gaussian',
                       psf_niters=3, x=None, y=None, mask=None,
                       bkg_mask_sources=True, gain=None, readnoise=None):
    """Process standart photometry in one image, without calibrations."""
    image = check_hdu(image)
    data = image.data

    if x is None or y is None:
        # First, we find the sources with sep, compute the FWHM and after
        # refine the point sources with daofind
        bkg, rms = background(data, box_size=64,
                              filter_size=3, mask=mask,
                              global_bkg=False)
        sources = sexfind(data, detect_snr, bkg, rms, mask=mask,
                           fwhm=detect_fwhm)
        if bkg_mask_sources:
            bkg_mask = sources_mask(data.shape, sources['x'], sources['y'],
                                    sources['a'], sources['b'],
                                    sources['theta'])
        else:
            bkg_mask = None
        # make a better identification
        sources = starfind(data, detect_snr, bkg, rms, mask=mask,
                           fwhm=detect_fwhm, box_size=box_size,
                           sharp_limit=(0.1, 2.0), round_limit=(-2.0, 2.0))

        x = sources['x']
        y = sources['y']
    else:
        fwhm = calc_fwhm(data, x, y, box_size=box_size, model='gaussian')
        if bkg_mask_sources:
            bkg_mask = sources_mask(data.shape, x, y,
                                    [0.75*fwhm]*len(x), [0.75*fwhm]*len(x),
                                    [0.0]*len(x))

    if photometry_type == 'aperture':
        result = aperture_photometry(data, x, y, r=r, r_ann=r_ann, gain=gain,
                                     readnoise=readnoise, mask=mask,
                                     bkg_mask=bkg_mask)
    elif photometry_type == 'psf':
        raise NotImplementedError()
        # result['aperture'] = ['psf']*len(x)

    return result
