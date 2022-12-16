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
from astropy.table import Table
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

_center_help = 'only search in indexes within `radius` of the field center ' \
               'given by `ra` and `dec`'
solve_field_params = {
    'center': '<SkyCoord or [ra, dec]>' + _center_help,
    'ra': '<Angle, float degrees or hh:mm:ss>' + _center_help,
    'dec': '<Angle, float degrees or +-hh:mm:ss>' + _center_help,
    'radius': 'Angle or float degrees> ' + _center_help,
    'scale': '<arcsec/pix> guess pixel scale in arcsec/pix. Alternative to '
             'scale-low, scale-high and scale-unit',
    'scale-tolerance': '<float> fraction tolerance for scale for lower and '
                       'upper limits.',
    'scale-low': '<float scale>lower bound of image scale estimate',
    'scale-high': '<float scale> upper bound of image scale estimate',
    'scale-units': '<units> in what units are the lower and upper bounds?'
                   '\nchoices:\n'
                   '"degwidth", "degw", "dw"   : width of the image, '
                   'in degrees (default)\n"arcminwidth", "amw", "aw" : width'
                   ' of the image, in arcminutes\n"arcsecperpix", "app": '
                   'arcseconds per pixel\n"focalmm": 35-mm (width-based)'
                   ' equivalent focal length',
    'depth': '<int or range> number of field objects to look at, or range '
             'of numbers; 1 is the brightest star, so "depth=10" or '
             '"depth=\'1-10\'" mean look at the top ten brightest stars.',
    'objs': '<int> cut the source list to have this many items (after '
            'sorting, if applicable).',
    'cpulimit': '<int seconds> give up solving after the specified number of'
                ' seconds of CPU time',
    'resort': 'sort the star brightnesses by background-subtracted '
              'flux; the default is to sort using acompromise between '
              'background-subtracted and non-background-subtracted flux',
    'fits-image': "assume the input files are FITS images",
    'timestamp': "add timestamps to log messages",
    'parity': '<pos/neg> only check for matches with positive/negative parity'
              ' (default: try both)',
    'code-tolerance': '<distance> matching distance for quads (default: 0.01)',
    'pixel-error': '<pixels> for verification, size of pixel positional error'
                   '(default: 1)',
    'quad-size-min': '<fraction> minimum size of quads to try, as a fraction'
                     'of the smaller image dimension, (default: 0.1)',
    'quad-size-max': '<fraction> maximum size of quads to try, as a fraction'
                     'of the image hypotenuse, default 1.0',
    'extension': '<int> FITS extension to read image from.',
    'invert': 'invert the image (for black-on-white images)',
    'downsample': '<int> downsample the image by factor <int> before running'
                  'source extraction',
    'no-background-subtraction': "don't try to estimate a smoothly-varying sky"
                                 ' background during source extraction.',
    'sigma': '<float> set the noise level in the image',
    'nsigma': 'number of sigma for a source detection; default 8',
    'no-remove-lines': "don't remove horizontal and vertical overdensities of"
                       " sources.",
    'uniformize': '<int> select sources uniformly using roughly this many '
                  'boxes (0=disable; default 10)',
    'crpix-center': 'set the WCS reference point to the image center',
    'crpix-x': 'set the WCS reference point to the given position',
    'crpix-y': 'set the WCS reference point to the given position',
    'no-tweak': "don't fine-tune WCS by computing a SIP polynomial",
    'tweak-order': '<int> polynomial order of SIP WCS corrections',
    'predistort': '<filename>: apply the inverse distortion in this SIP WCS'
                  ' header before solving',
    'xscale': '<factor> for rectangular pixels: factor to apply to measured X'
              ' positions to make pixels square',
    'fields': '<number or range> the FITS extension(s) to solve, inclusive',
    'no-plots': 'don\'t create any plots of the results',
    'overwrite': 'overwrite output files if they already exist'
}


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
    return float(angle)


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


class AstrometricSolution():
    """Store astrometric solution.

    Parameters
    ----------
    wcs: `~astropy.wcs.WCS`
        World Coordinate System (wcs) object containing the solved solution.
    header: `~astropy.io.fits.Header` (optional)
        Astrometric solved fits header.
    correspondences: `~astropy.table.Table` (optional)
        A table containing the matched correspondences between the (x, y)
        positions and the matched (ra, dec) catalog coordinates.
    """

    _wcs = None
    _header = None
    _corr = None

    def __init__(self, wcs, header=None, correspondences=None):
        self._wcs = WCS(wcs)
        if header is not None:
            self._header = fits.Header(header)
        if correspondences is not None:
            self._corr = Table(correspondences)

    @property
    def wcs(self):
        return copy.deepcopy(self._wcs)

    @property
    def header(self):
        return copy.deepcopy(self._header)

    @property
    def correspondences(self):
        return copy.deepcopy(self._corr)


class AstrometrySolver():
    """Use astrometry.net to solve the astrometry of images or list of stars.
    For convenience, all the auxiliary files will be deleted, except you
    specify to keep it with ``keep_files``.

    Parameters
    ----------
    solve_field: string (optional)
        ``solve-field`` command from astrometry.net package. If not set, it
        will be determined by `~shutil.which` function.
    defaults: `dict` (optional)
        Default arguments to be passed to ``solve-field`` program. If not set,
        arguments ``no-plot`` and ``overwrite`` will be used. Use only double
        dashed arguments, ignoring the dashed in the `dict` keys.
    keep_files: bool (optional)
        Keep the temporary files after finish.
    """

    def __init__(self, solve_field=_solve_field,
                 defaults=None, keep_files=False):
        # declare the defaults here to be safer
        self._defaults = {'no-plots': None, 'overwrite': None}
        if defaults is None:
            defaults = {}
        self._defaults.update(defaults)

        self._command = solve_field
        self._keep = keep_files
        self.logger = logger

    def solve_field(self, filename, options=None, **kwargs):
        """Try to solve an image using the astrometry.net.

        Parameters
        ----------
        filename: str
            Name of the file to be solved. Can be a fits image or a xyls
            sources file.
        options: `dict` (optional)
            Dictionary of ``solve-field`` options. See
            `~astropop.astrometry.astrometrynet.print_options_help` for all
            available options. The most useful are:
            - center or ra and dec: field center
            - radius: maximum search radius
            - scale: pixel scale in arcsec/pixel
            - tweak-order: SIP order to fit
        **kwargs:
            Additional keyword arguments to be passed to
            `~astropop.py_utils.run_command`.

        Returns:
        --------
        `~astropop.astrometry.AstrometricSolution`
            Astrometric solution class containing the solved header,
            the `~astropy.wcs.WCS` solved class and the correspondence table
            between the sources and the catalog.
        """
        if options is None:
            options = {}

        n_opt = copy.copy(self._defaults)
        n_opt.update(options)

        solved_header, coorespond = self._run_solver(filename,
                                                     options=n_opt,
                                                     **kwargs)

        return AstrometricSolution(WCS(solved_header, relax=True),
                                   header=solved_header,
                                   correspondences=coorespond)

    def _get_output_dir(self, root, output_dir):
        """Check output directory and create the temporary directory."""
        if output_dir is None:
            output_dir = mkdtemp(prefix=root + '_', suffix='_astrometry.net')
            tmp_dir = True
        else:
            tmp_dir = False
        return output_dir, tmp_dir

    def _parse_options(self, options):
        """Parse and check all known options."""
        for key in options.keys():
            if key not in solve_field_params:
                raise KeyError(f'option {key} not supported.')

        args = []
        # parse center
        if 'center' in options or ('ra' in options and 'dec' in options):
            center = _parse_coordinates(options.pop('center', None),
                                        options.pop('ra', None),
                                        options.pop('dec', None))
            args += ['--ra', str(center['ra']),
                     '--dec', str(center['dec'])]
        else:
            logger.info('Astrometry solving with blind field center.')

        # parse plate scale

        return args

    def _run_solver(self, filename, options, output_dir=None, **kwargs):
        """Run the astrometry.net localy using the given params.

        STDOUT and STDERR can be stored in variables for better check after
        using kwargs.
        """
        basename = os.path.basename(filename)
        root, _ = os.path.splitext(basename)
        output_dir, tmp_dir = self._get_output_dir(root, output_dir)
        solved_file = os.path.join(output_dir, root + '.solved')
        correspond = os.path.join(output_dir, root + '.corr')

        args = [self._command, filename, '--dir', output_dir]
        args += self._parse_options(options)
        args += ['--corr', correspond]

        print(args)

        try:
            process, _, _ = run_command(args, **kwargs)
            if process.returncode != 0:
                raise CalledProcessError(process.returncode, self._command)

            # .solved file must exist and contain a binary one
            with open(solved_file, 'rb') as fd:
                if ord(fd.read()) != 1:
                    raise AstrometryNetUnsolvedField(filename)
            solved_wcs_file = os.path.join(output_dir, root + '.wcs')
            self.logger.debug('Loading solved header from %s', solved_wcs_file)
            solved_header = fits.getheader(solved_wcs_file, 0)
            print(open(correspond, 'rb').read())

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


def solve_astrometry_xy(x, y, flux, width, height,
                        image_header=None, options=None,
                        command=_solve_field,
                        **kwargs):
    """Solve astrometry from a (x,y) sources list using astrometry.net.

    Parameters
    ----------
    x: array-like
        x positions of the sources in the image.
    y: array-like
        y positions of the sources in the image.
    flux: array_like
        Estimated fluxes of the sources. A higher value indicates a brighter
        star.
    width: int
        Image width.
    height: int
        Image height.
    image_header: `~astropy.fits.Header` (optional)
        Original image header.
    options: dict
        Dictionary of ``solve-field`` options. See
        `~astropop.astrometry.astrometrynet.print_options_help` for all
        available options. The most useful are:
        - center or ra and dec: field center
        - radius: maximum search radius
        - scale: pixel scale in arcsec/pixel
        - tweak-order: SIP order to fit
    command: str
        Full path of astrometry.net ``solve-field`` command.

    Returns
    -------
    `~astropop.astrometry.AstrometricSolution`
        Astrometric solution ot the field.
    """
    if options is None:
        options = {}
    image_header, _ = extract_header_wcs(image_header)
    f = NamedTemporaryFile(suffix='.xyls')
    create_xyls(f.name, x, y, flux, width, height,
                header=image_header)
    solver = AstrometrySolver(astrometry_command=command)
    return solver.solve_field(f.name, options=options, **kwargs)


def solve_astrometry_image(filename, options=None, command=_solve_field,
                           **kwargs):
    """Solve astrometry from an image using astrometry.net.

    Parameters
    ----------
    filename: str
        Path to the image to solve.
    options: dict
        Dictionary of ``solve-field`` options. See
        `~astropop.astrometry.astrometrynet.print_options_help` for all
        available options. The most useful are:
        - center or ra and dec: field center
        - radius: maximum search radius
        - scale: pixel scale in arcsec/pixel
        - tweak-order: SIP order to fit
    command: str
        Full path of astrometry.net ``solve-field`` command.

    Returns
    -------
    `~astropop.astrometry.AstrometricSolution`
        Astrometric solution ot the field.
    """
    if options is None:
        options = {}
    solver = AstrometrySolver(astrometry_command=command)
    return solver.solve_field(filename, options=options, **kwargs)


def solve_astrometry_hdu(hdu, options=None, command=_solve_field, **kwargs):
    """Solve astrometry from a `~astropy.fits.ImageHDU` using astrometry.net.

    Parameters
    ----------
    hdu: `astropy.fits.ImageHDU`
        FITS hdu containing the image to solve.
    options: dict
        Dictionary of ``solve-field`` options. See
        `~astropop.astrometry.astrometrynet.print_options_help` for all
        available options. The most useful are:
        - center or ra and dec: field center
        - radius: maximum search radius
        - scale: pixel scale in arcsec/pixel
        - tweak-order: SIP order to fit
    command: str
        Full path of astrometry.net ``solve-field`` command.

    Returns
    -------
    `~astropop.astrometry.AstrometricSolution`
        Astrometric solution ot the field.
    """
    if options is None:
        options = {}
    hdu.header, _ = extract_header_wcs(hdu.header)
    f = NamedTemporaryFile(suffix='.fits')
    hdu = fits.PrimaryHDU(hdu.data, header=hdu.header)
    hdu.writeto(f.name)
    solver = AstrometrySolver(astrometry_command=command)
    return solver.solve_field(f.name, options=options, **kwargs)


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
