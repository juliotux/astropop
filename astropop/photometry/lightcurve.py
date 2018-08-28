import os
import numpy as np
from astropy.io import fits
from astropy.table import Table, hstack, vstack
from astropy.time import Time

from ..image_processing.register import register_translation
from ..image_processing.imarith import check_hdu
from ._phot import process_photometry
from .aperture import aperture_photometry
from ..logger import logger


# TODO: Need refactor


def temporal_photometry(image_list, x=None, y=None, ext=0,
                        photometry_type='aperture',
                        r=5, r_in=50, r_out=60, detect_snr=5,
                        detect_fwhm=3, psf_model='gaussian', psf_niters=1,
                        time_key='DATE-OBS', time_format='isot',
                        align_images=True, nstars_thresh=None):
    """Perform photometry on a set of images, optimized for large datasets.

    Arguments:
    ----------
        image_list : list_like
            The list containing the name of the images to be used.
        x :  array_like
            Array conatining the x coordinates of the sources. If None,
            the code will find the sources in the first image.
        y :  array_like
            Array conatining the y coordinates of the sources. If None,
            the code will find the sources in the first image.
        photometry_type : string
            Type of the photometry to be performed. Can be `aperture` or `psf`
        r : float
            The aperture radius to be used if aperture photometry will be
            performed.
        r_in : float
            The inner radius of the sky subtraction annulus to be used
            if aperture photometry will be performed.
        r_out : float
            The outer radius of the sky subtraction annulus to be used
            if aperture photometry will be performed.
        detect_snr : float
            The minimum signal to noise ratio to detect sources.
        detect_fwhm : float
            The fwhm to detect sources.
        psf_model : `gaussian` or `moffat`
            Model of the psf to fit to the sources.
        psf_niters : int
            Number of iterations for iterative psf subtraction.
        time_key : string
            Keyword in the fits header to use as time indication.
        time_format : string
            Format of the time in the fits header. Can be any valid Format
            of astropy.time.Time.
        align_images : bool
            Calculate the shifts between the images and consider it in the
            photometry.
    """
    n_imgs = len(image_list)

    if x is None or y is None:
        # use the first image to calculate the positions of the stars
        sources = aperture_photometry(check_hdu(image_list[0], ext=ext).data,
                                      detect_fwhm=detect_fwhm,
                                      r=5, detect_snr=detect_snr)
    else:
        sources = np.array(list(zip(x, y)),
                           dtype=np.dtype([('x', 'f8'), ('y', 'f8')]))
    n_src = len(sources)

    # if the images are not aligned, get the shifts between tem
    shifts = np.zeros(n_imgs, dtype=np.dtype([('x', 'f8'), ('y', 'f8')]))
    if align_images:
        logger.info('Processing image shifts for {} images.'.format(n_imgs))
        im0 = check_hdu(image_list[0], ext=ext)
        for i in range(1, n_imgs):
            im = check_hdu(image_list[i])
            s = register_translation(im0.data, im.data)[0]
            shifts[i]['x'] = s[1]
            shifts[i]['y'] = s[0]

    # use these coordinates to perform the photometry in all data series
    phot_table = None
    for i in range(n_imgs):
        if nstars_thresh is not None:
            ns = aperture_photometry(check_hdu(image_list[i], ext=ext).data,
                                     detect_fwhm=detect_fwhm,
                                     r=5, detect_snr=detect_snr)
            if ns < nstars_thresh:
                logger.warn('Image {} have less stars then the threshold: '
                            '{} stars'.format(image_list[i], ns))
                continue
        imname = os.path.basename(image_list[i])
        logger.info("Processing photometry of {} image.".format(imname))
        jd = Time(fits.getval(image_list[i], time_key, ext=ext),
                  format=time_format)
        jd = jd.jd
        p = process_photometry(check_hdu(image_list[i], ext=ext),
                               photometry_type=photometry_type,
                               r=r, r_in=r_in, r_out=r_out,
                               psf_model=psf_model, psf_niters=psf_niters,
                               x=sources['x'] - shifts[i]['x'],
                               y=sources['y'] - shifts[i]['y'])

        # stack the photometry results with file/time/star infos and
        # instrumental magnitudes
        t = hstack([Table([[imname]*n_src, [i]*n_src, [jd]*n_src,
                           np.arange(n_src)],
                          names=('file_name', 'file_index', 'jd',
                                 'star_index')),
                    p,
                    Table([-2.5*np.log10(p['flux'])-25,
                           1.086*(p['flux_error']/p['flux'])],
                          names=('inst_mag', 'inst_mag_error'))])
        if phot_table is None:
            phot_table = t
        else:
            phot_table = vstack([phot_table, t])

    return phot_table


def process_lightcurve(image_list, x=None, y=None, photometry_type='aperture',
                       r=5, r_in=50, r_out=60, detect_snr=5,
                       detect_fwhm=3, psf_model='gaussian', psf_niters=1,
                       time_key='DATE-OBS', time_format='isot',
                       align_images=True, check_dist=True):
    '''Generates a light curve using a list of images.
    See function above for parameters description.

    jd_header_keyword: keyword of the JD in header
    x, y: stars coordinates. The first has to be the scinece star. The other
          will be the references.
    '''
    if x is None or y is None:
        raise ValueError('No star coordinates passed. You must give them!')
    else:
        if len(x) < 2 or len(y) < 2:
            raise ValueError('x and y must have the coordinates for at last '
                             '2 stars.')

    tmp_phot = temporal_photometry(image_list, x=x, y=y,
                                   photometry_type=photometry_type,
                                   r=r, r_in=r_in, r_out=r_out,
                                   detect_snr=detect_snr,
                                   detect_fwhm=detect_fwhm,
                                   psf_niters=psf_niters,
                                   psf_model=psf_model,
                                   time_key=time_key, time_format=time_format,
                                   align_images=align_images)

    g = tmp_phot.group_by('star_index')
    t = Table()
    t.add_column(g.groups[0]['jd'])
    for i in range(1, len(x), 1):
        t['star{}-star0'.format(i)] = g.groups[i]['inst_mag'] - \
                                      g.groups[0]['inst_mag']
        err = g.groups[0]['inst_mag_error'] + g.groups[i]['inst_mag_error']
        t['star{}-star0_error'.format(i)] = err

    return t