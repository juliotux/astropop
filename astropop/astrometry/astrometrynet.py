# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Astrometry.net wrapper.

This module wraps the astrometry.net solve-field routine, with automatic
improvements.
"""


import os
import shutil
from subprocess import CalledProcessError
import copy
from tempfile import NamedTemporaryFile, mkdtemp

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import Angle, SkyCoord
from astropy.units import UnitsError

from ..framedata.compat import extract_header_wcs
from ..logger import logger
from ..py_utils import run_command


__all__ = ['AstrometrySolver', 'solve_astrometry_xy', 'solve_astrometry_image',
           'create_xyls', 'AstrometryNetUnsolvedField']


_fit_wcs = shutil.which('fit-wcs')
_solve_field = shutil.which('solve-field')


class AstrometryNetUnsolvedField(CalledProcessError):
    """Raised if Astrometry.net could not solve the field."""

    def __init__(self, path):
        self.path = path

    def __str__(self):
        return f"{self.path}: could not solve field"


def _parse_angle(angle, unit=None):
    """Transform angle in float."""
    # plate scale
    # radius
    # bare ra and dec
    if isinstance(angle, str):
        try:
            angle = Angle(angle)
        except UnitsError:
            if unit is not None:
                angle = Angle(angle, unit=unit)
    if isinstance(angle, Angle):
        angle = angle.degree
    if not isinstance(angle, float):
        raise ValueError(f'{angle} (type {type(angle)})not recognized as a '
                         'valid angle.')


def _parse_coordinates(coord=None, ra=None, dec=None):
    """Parse filed center coordinates."""
    if coord is not None and (ra is not None or dec is not None):
        raise ValueError('Field center defined by `coord` conflicts with `ra`'
                         ' and `dec` fields')

    # coord can by a tuple of ra, dec
    if isinstance(coord, (list, tuple)):
        ra, dec = coord
    elif isinstance(coord, SkyCoord):
        ra = coord.ra.degree
        dec = coord.dec.degree

    ra = _parse_angle(ra, 'hourangle')
    dec = _parse_angle(dec, 'degree')

    return {'ra': ra, 'dec': dec}


def _parse_pltscl(pltscl, tolerance=0.2):
    """Parse plate scale."""
    low = pltscl*(1-tolerance)
    hi = pltscl*(1+tolerance)
    return {'scale-low': low, 'scale-high': hi, 'scale-units': 'arcsecperpix'}


class AstrometrySolver():
    """Use astrometry.net to solve the astrometry of images or list of stars.
    For convenience, all the auxiliary files will be deleted, except you
    specify to keep it with 'keep_files'.
    """

    def __init__(self, astrometry_command=_solve_field,
                 defaults=None, keep_files=False):
        # declare the defaults here to be safer
        self._defaults = {'no-plot': None, 'overwrite': None}
        if defaults is None:
            defaults = {}
        self._defaults.update(defaults)

        self._command = astrometry_command
        self._keep = keep_files
        self.logger = logger

    def solve_field(self, filename, output_file=None, wcs=False,
                    image_params=None, solve_params=None, scamp_basename=None,
                    **kwargs):
        # TODO: put some of these arguments in a config file
        """Try to solve an image using the astrometry.net.

        The image params can be:
            'ra' : approximate center of the image in RA in decimal degrees
            'dec' : approximate center of the image in DEC in decimal degrees
            'pltscl' : plate scale of the image in arcsec/px
            'radius' : maximum radius to search around the center

        The solve_params are additional parameters that will be passed to
        solve-filed command. They can be:
            'no-fits2fits', 'overwrite', 'depth', 'no-plot', etc.

        'hdu' defines the number of the HDU to be solved, generally 0 (first)
        for images.

        Returns:
        --------
        header : `astropy.io.fits.Header` or `astropy.wcs.WCS`
            A header containing the solved wcs of the field and another
            information from astrometry.net. If return_wcs=True, a WCS
            object will be returned.
        """
        if solve_params is None:
            solve_params = {}

        options = copy.copy(self._defaults)
        options.update(solve_params)

        solved_header = self._run_solver(filename, params=options, **kwargs)

        if not wcs:
            return solved_header
        else:
            return WCS(solved_header, relax=True)

    def _run_solver(self, filename, params, output_dir=None, **kwargs):
        """Run the astrometry.net localy using the given params.

        STDOUT and STDERR can be stored in variables for better check after.
        """
        basename = os.path.basename(filename)
        root, _ = os.path.splitext(basename)
        if output_dir is None:
            output_dir = mkdtemp(prefix=root + '_', suffix='_astrometry.net')
            tmp_dir = True
        else:
            tmp_dir = False
        solved_file = os.path.join(output_dir, root + '.solved')

        args = [self._command, filename, '--dir', output_dir]

        for i, v in params.items():
            ndashes = 1 if len(i) == 1 else 2
            args.append(f"{ndashes * '-'}{i}")
            if v is not None:
                args.append(str(v))

        try:
            process, _, _ = run_command(args, **kwargs)
            if process.returncode != 0:
                raise CalledProcessError(process.returncode, self._command)

            # .solved file must exist and contain a binary one
            with open(solved_file, 'rb') as fd:
                if ord(fd.read()) != 1:
                    raise AstrometryNetUnsolvedField(filename)
            solved_wcs_file = os.path.join(output_dir, root + '.wcs')
            self.logger.info('Loading solved header from %s', solved_wcs_file)
            solved_header = fits.getheader(solved_wcs_file, 0)

            # remove the tree if the file is temporary and not set to keep
            if not self._keep and tmp_dir:
                shutil.rmtree(output_dir)

            return solved_header

        except CalledProcessError as e:
            if not self._keep and tmp_dir:
                shutil.rmtree(output_dir)
            raise e

        # If .solved file doesn't exist or contain one
        except (IOError, AstrometryNetUnsolvedField):
            if not self._keep and tmp_dir:
                shutil.rmtree(output_dir)
            raise AstrometryNetUnsolvedField(filename)


def create_xyls(fname, x, y, flux, imagew, imageh, header=None, dtype='f8'):
    '''
    Create and save the xyls file to run in astrometry.net

    Parameters:
    -----------
    fname : str
        The path to save the .xyls file.
    x : array_like
        X coordinates of the sources.
    y : array_like
        Y coordinates of the sources.
    flux : array_like
        Estimated flux of the sources.
    imagew : float or int
        Width of the original image. `IMAGEW` header field of .xyls
    imageh : float or int
        Height of the original image. `IMAGEH` header field of .xyls
    dtype : `numpy.dtype`, optional
        Data type of the fields. Default: 'f8'
    '''
    head = {'IMAGEW': imagew,
            'IMAGEH': imageh}
    sort = np.argsort(flux)
    xyls = np.array(list(zip(x, y, flux)),
                    np.dtype([('x', dtype), ('y', dtype), ('flux', dtype)]))
    f = fits.HDUList([fits.PrimaryHDU(header=header),
                      fits.BinTableHDU(xyls[sort[::-1]],
                                       header=fits.Header(head))])
    logger.debug('Saving .xyls to %s', fname)
    f.writeto(fname)


def solve_astrometry_xy(x, y, flux, image_header, image_width, image_height,
                        return_wcs=False, image_params=None,
                        **kwargs):
    """Solve astrometry from a (x,y) sources list.

    Notes
    -----
    - image_params are:
      * pltscl: plate scale (arcsec/px)
      * ra: right ascension (decimal degrees)
      * dec: declination (decimal degrees)
      * radius: maximum search radius
    """
    if image_params is None:
        image_params = {}
    image_header, _ = extract_header_wcs(image_header)
    f = NamedTemporaryFile(suffix='.xyls')
    create_xyls(f.name, x, y, flux, image_width, image_height,
                header=image_header)
    solver = AstrometrySolver()
    return solver.solve_field(f.name, wcs=return_wcs,
                              image_params=image_params, **kwargs)


def solve_astrometry_image(filename, return_wcs=False, image_params=None,
                           **kwargs):
    """
    image_params are:
        pltscl: plate scale (arcsec/px)
        ra: right ascension (decimal degrees)
        dec: declination (decimal degrees)
        radius: maximum search radius
    """
    if image_params is None:
        image_params = {}
    solver = AstrometrySolver()
    return solver.solve_field(filename, wcs=return_wcs,
                              image_params=image_params, **kwargs)


def solve_astrometry_hdu(hdu, return_wcs=False, image_params=None,
                         **kwargs):
    """
    image_params are:
        pltscl: plate scale (arcsec/px)
        ra: right ascension (decimal degrees)
        dec: declination (decimal degrees)
        radius: maximum search radius
    """
    if image_params is None:
        image_params = {}
    hdu.header, _ = extract_header_wcs(hdu.header)
    f = NamedTemporaryFile(suffix='.fits')
    hdu = fits.PrimaryHDU(hdu.data, header=hdu.header)
    hdu.writeto(f.name)
    solver = AstrometrySolver()
    return solver.solve_field(f.name, wcs=return_wcs,
                              image_params=image_params, **kwargs)


def fit_wcs(x, y, ra, dec, image_width, image_height, sip=False,
            command=_fit_wcs, **kwargs):
    """Run astrometry.net fit-wcs command

    sip is sip order, int
    """
    solved_wcs_file = NamedTemporaryFile(prefix='fitwcs', suffix='.wcs')
    tmp_table = NamedTemporaryFile(prefix='fitwcs_xyrd', suffix='.fits')

    xyrd = np.array(list(zip(x, y, ra, dec)),
                    np.dtype([('FIELD_X', 'f8'), ('FIELD_Y', 'f8'),
                              ('INDEX_RA', 'f8'), ('INDEX_DEC', 'f8')]))
    f = fits.HDUList([fits.PrimaryHDU(),
                      fits.BinTableHDU(xyrd)])
    logger.debug('Saving xyrd to %s', tmp_table.name)
    f.writeto(tmp_table.name)

    args = [command]
    args += ['-c', tmp_table.name]

    if sip:
        args += ['-s', str(sip), '-W', str(image_width), '-H',
                 str(image_height)]
    args += ['-o', solved_wcs_file.name]

    try:
        process, _, _ = run_command(args, **kwargs)
        if process.returncode != 0:
            raise CalledProcessError(process.returncode, args)
        logger.info('Loading solved header from %s', solved_wcs_file.name)
        solved_header = fits.getheader(solved_wcs_file.name, 0)
    except Exception as e:
        raise AstrometryNetUnsolvedField("Could not fit wcs to this lists."
                                         f" Error: {e}")

    return WCS(solved_header, relax=True)
