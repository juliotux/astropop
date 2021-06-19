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

from .coords_utils import guess_coordinates
from ..logger import logger
from ..py_utils import check_iterable, run_command


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


wcs_header_keys = ['CRPIX1', 'CRPIX2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
                   'CRVAL1', 'CRVAL2', 'CTYPE1', 'CTYPE2', 'CUNIT1',
                   'CUNIT2']


def clean_previous_wcs(header):
    """Clean any previous WCS keyowrds in header."""
    h = fits.Header()
    for k in header.keys():
        if k not in wcs_header_keys:
            indx = header.index(k)
            card = header.cards[indx]
            h.append(card)
    return h


class AstrometrySolver():
    """Use astrometry.net to solve the astrometry of images or list of stars.
    For convenience, all the auxiliary files will be deleted, except you
    specify to keep it with 'keep_files'.
    """
    _defaults = None

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

    def _guess_coordinates(self, header, ra_key='RA', dec_key='DEC'):
        """Guess the field center based in header keys."""
        ra = header.get(ra_key)
        dec = header.get(dec_key)
        return guess_coordinates(ra, dec)

    def _guess_field_params(self, header, image_params):
        """Guess the approximate field parameters from the header.

        The estimated parameters are:
            coordinates : 'ra' and 'dec'
            plate scale : 'scale-units', 'scale-high', 'scale-low'
        """
        options = {}
        keys = image_params.keys()
        if 'ra' in keys and 'dec' in keys:
            try:
                ra = float(image_params.get('ra'))
                dec = float(image_params.get('dec'))
                self.logger.info("Usign given field coordinates: %s %s",
                                 ra, dec)
                options['ra'] = ra
                options['dec'] = dec
            except ValueError:
                self.logger.warning('Could not convert field coordinates to'
                                    ' decimal degrees. Ignoring it: %s %s',
                                    ra, dec)
        elif 'ra_key' in keys and 'dec_key' in keys:
            self.logger.info("Figuring out field center coordinates")
            try:
                coords = self._guess_coordinates(header,
                                                 image_params.get('ra_key'),
                                                 image_params.get('dec_key'))
                options['ra'] = coords.ra.degree
                options['dec'] = coords.dec.degree
            except KeyError:
                self.logger.warning("Cannot understand coordinates in"
                                    " FITS header")
        else:
            self.logger.warning("Astrometry.net will try to solve without the"
                                " field center")

        if 'pltscl' in keys:
            try:
                pltscl = image_params.get('pltscl')
                if check_iterable(pltscl):
                    pltscl = [float(i) for i in pltscl]
                else:
                    pltscl = float(image_params.get('pltscl'))
                    pltscl = [0.8*pltscl, 1.2*pltscl]
                pltscl = np.array(sorted(pltscl))
                self.logger.info("Usign given plate scale: %s", pltscl)
            except ValueError:
                self.logger.warning('Plate scale value not recognized.'
                                    ' Ignoring it. %s', pltscl)
        elif 'pltscl_key' in keys:
            self.logger.info("Figuring out the plate scale from FITS header")
            try:
                pltscl = header[image_params.get('pltscl_key')]
                pltscl = float(pltscl)
            except KeyError:
                self.logger.warning("Cannot understand plate scale in FITS"
                                    " header")
        try:
            options['scale-high'] = pltscl[1]
            options['scale-low'] = pltscl[0]
            options['scale-units'] = 'arcsecperpix'
        except NameError:
            self.logger.warning('Astrometry.net will be run without plate'
                                ' scale.')

        if 'ra' in options.keys():
            if 'radius' in keys:
                options['radius'] = float(image_params.get('radius', 1.0))
            else:
                options['radius'] = 1.0

        return options

    def solve_field(self, filename, output_file=None, wcs=False,
                    image_params=None, solve_params=None, scamp_basename=None,
                    **kwargs):
        # TODO: put some of these arguments in a config file
        """Try to solve an image using the astrometry.net.

        The image params can be:
            'ra_key' : header key for RA
            'dec_key' : header key for DEC
            'pltscl_key' : header key for plate scale
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
        if image_params is None:
            image_params = {}
        if solve_params is None:
            solve_params = {}

        options = copy.copy(self._defaults)
        options.update(solve_params)

        if scamp_basename is not None:
            options['scamp-ref'] = scamp_basename + ".ref"
            options['scamp-config'] = scamp_basename + ".scamp"
            options['scamp'] = scamp_basename + ".cat"

        if output_file is not None:
            options['new-fits'] = output_file

        try:
            field_params = self._guess_field_params(fits.getheader(filename),
                                                    image_params=image_params)
        except (OSError, IOError):
            self.logger.warning('Could not guess field center and plate scale.'
                                ' Running in slow mode.')
            field_params = {}
        options.update(field_params)

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
      * pltscl_key: header key for plate scale
      * ra_key: header key for right ascension
      * dec_key: header key for declination
      * radius: maximum search radius
    """
    if image_params is None:
        image_params = {}
    image_header = clean_previous_wcs(image_header)
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
        pltscl_key: header key for plate scale
        ra_key: header key for right ascension
        dec_key: header key for declination
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
        pltscl_key: header key for plate scale
        ra_key: header key for right ascension
        dec_key: header key for declination
        radius: maximum search radius
    """
    if image_params is None:
        image_params = {}
    hdu.header = clean_previous_wcs(hdu.header)
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
