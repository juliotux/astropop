# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Skyview helper for default plots"""

import numpy as np
from astropy.wcs.utils import proj_plane_pixel_scales
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
from reproject import reproject_interp
from astropy import units as u


def get_dss_image(shape, wcs, survey='DSS'):
    '''Use astroquery SkyView to get a DSS image projected to wcs and shape.'''
    shape = np.array(shape)
    platescale = proj_plane_pixel_scales(wcs)
    ra, dec = wcs.wcs_pix2world(*(shape/2), 0)
    sk = SkyCoord(ra, dec, unit=('degree', 'degree'), frame='icrs')
    im = SkyView.get_images(sk, survey='DSS', coordinates='ICRS',
                            width=shape[0]*platescale[0]*u.degree,
                            height=shape[1]*platescale[1]*u.degree)[0]
    im = reproject_interp(im[0], output_projection=wcs, shape_out=shape)[0]
    return im
