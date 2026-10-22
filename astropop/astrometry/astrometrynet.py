# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Astrometry.net wrapper.

This module wraps the astrometry.net solve-field routine, with automatic
improvements.
"""

from __future__ import annotations

import typing as _t
from inspect import cleandoc as _cleandoc
import numpy.typing as _npt
import subprocess as _subprocess
from ..framedata import FrameData as _FrameData
from .._typing import (PathLike as _PathLike, UnitLike as _UnitLike,
                       HeaderLike as _HeaderLike, ImageHDU as _ImageHDU)

import os
import shutil
from subprocess import CalledProcessError
import copy
from packaging import version
from tempfile import NamedTemporaryFile, mkdtemp
import warnings

import numpy as np

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import Angle, SkyCoord
from astropy.units import UnitsError

from ..framedata._compat import extract_header_wcs
from ..logger import logger
from ..py_utils import run_command, check_number


__all__ = ['AstrometrySolver', 'solve_astrometry_xy', 'solve_astrometry_image',
           'solve_astrometry_framedata', 'create_xyls',
           'AstrometryNetUnsolvedField', 'SolveFieldCommand']


_solve_field = shutil.which('solve-field')


class SolveFieldCommand:
    """Locate and invoke an Astrometry.net solve-field executable.

    Parameters
    ----------
    command
        Existing executable path, defaulting to solve-field found on PATH.
        An existing wrapper can be copied to reuse its cached version.

    Notes
    -----
    Construction checks path existence. The version property runs
    ``solve-field --version`` lazily; run executes the supplied arguments.
    """
    _version = None

    def __init__(self, command: _PathLike | None=_solve_field) -> None:
        """Validate the executable path or reuse an existing command wrapper."""
        if isinstance(command, SolveFieldCommand):
            self._version = command._version
            command = command._command
        elif command is None or not os.path.exists(command):
            raise FileNotFoundError('solve-field command not found.')
        self._command = command

    @property
    def version(self) -> version.Version:
        """Return the version of the solve-field command."""
        if self._version is None:
            _, sout, _ = run_command([self._command, '--version'])
            self._version = version.parse(sout[0])
        return self._version

    def run(
        self,
        args: _t.Sequence[str],
        **kwargs: _t.Any,
    ) -> tuple[_subprocess.CompletedProcess[str], list[str], list[str]]:
        """Run the solve-field command."""
        return run_command([self._command, *args], **kwargs)


_center_help = 'only search in indexes within `radius` of the field center ' \
               'given by `ra` and `dec`'
solve_field_params = {
    'center': (
        '<SkyCoord or [ra, dec]>',
        _center_help
    ),
    'ra': (
        '<Angle, float degrees or hh:mm:ss>',
        _center_help
    ),
    'dec': (
        '<Angle, float degrees or +-dd:mm:ss>',
        _center_help
    ),
    'radius': (
        'Angle or float degrees> ',
        _center_help
    ),
    'plate-scale': (
        '<arcsec/pix>',
        'guess pixel scale in arcsec/pix. Alternative to scale-low, scale-high'
        ' and scale-unit'
    ),
    'scale-tolerance': (
        '<float>',
        'fraction tolerance for scale for lower and upper limits.'
    ),
    'scale-low': (
        '<float scale>',
        'lower bound of image scale estimate'
    ),
    'scale-high': (
        '<float scale>',
        'upper bound of image scale estimate'
    ),
    'scale-units': (
        '<units>',
        'in what units are the lower and upper bounds? Coices are:'
        '``degwidth``, ``degw``, ``dw`` for width of the image, '
        'in degrees (default); ``arcminwidth``, ``amw``, ``aw`` for width'
        ' of the image, in arcminutes; ``arcsecperpix``, ``app`` for '
        'arcseconds per pixel and ``focalmm`` 35-mm (width-based)'
        ' equivalent focal length'
    ),
    'depth': (
        '<int or range>',
        'number of field objects to look at, or range of numbers; 1 is the '
        'brightest star, so ``depth=10`` or ``depth=\'1-10\'`` mean look at '
        'the top ten brightest stars.'
    ),
    'objs': (
        '<int>',
        'cut the source list to have this many items (after  sorting, '
        'if applicable).'
    ),
    'cpulimit': (
        '<int seconds>',
        'give up solving after the specified number of  seconds of CPU time'
    ),
    'resort': (
        None,
        'sort the star brightnesses by background-subtracted  flux; the '
        'default is to sort using acompromise between background-subtracted '
        'and non-background-subtracted flux'
    ),
    'fits-image': (
        None,
        "assume the input files are FITS images"
    ),
    'timestamp': (
        None,
        "add timestamps to log messages"
    ),
    'parity': (
        '<pos/neg>',
        'only check for matches with positive/negative parity (default try '
        'both)'
    ),
    'code-tolerance': (
        '<distance>',
        'matching distance for quads (default=0.01)'
    ),
    'pixel-error': (
        '<pixels>',
        'for verification, size of pixel positional error (default=1)'
    ),
    'quad-size-min': (
        '<fraction>',
        'minimum size of quads to try, as a fraction of the smaller image '
        'dimension, (default=0.1)'
    ),
    'quad-size-max': (
        '<fraction>',
        'maximum size of quads to try, as a fraction of the image hypotenuse, '
        'default=1.0'
    ),
    'extension': (
        '<int>',
        'FITS extension to read image from.'
    ),
    'invert': (
        None,
        'invert the image (for black-on-white images)'
    ),
    'downsample': (
        '<int>',
        'downsample the image by factor <int> before running source extraction'
    ),
    'no-background-subtraction': (
        None,
        "don't try to estimate a smoothly-varying sky background during source"
        ' extraction.'
    ),
    'sigma': (
        '<float>',
        'set the noise level in the image'
    ),
    'nsigma': (
        '<int>',
        'number of sigma for a source detection; default 8'
    ),
    'no-remove-lines': (
        None,
        "don't remove horizontal and vertical overdensities of  sources."
    ),
    'uniformize': (
        '<int>',
        'select sources uniformly using roughly this many boxes (0=disable; '
        'default=10)'
    ),
    'crpix-center': (
        None,
        'set the WCS reference point to the image center'
    ),
    'crpix-x': (
        '<float>',
        'set the WCS reference point to the given position'
    ),
    'crpix-y': (
        '<float>',
        'set the WCS reference point to the given position'
    ),
    'no-tweak': (
        None,
        "don't fine-tune WCS by computing a SIP polynomial"
    ),
    'tweak-order': (
        '<int>',
        'polynomial order of SIP WCS corrections'
    ),
    'predistort': (
        '<filename>',
        'apply the inverse distortion in this SIP WCS header before solving'
    ),
    'xscale': (
        '<factor>',
        'for rectangular pixels: factor to apply to measured X positions to'
        ' make pixels square'
    ),
    'fields': (
        '<number or range>',
        'the FITS extension(s) to solve, inclusive'
    ),
    'no-plots': (
        None,
        "don't create any plots of the results"
    ),
    'overwrite': (
        None,
        'overwrite output files if they already exist'
    ),
    'width': (
        '<pixels>',
        'specify the field width'
    ),
    'height': (
        '<pixels>',
        'specify the field height'
    ),
    'minwidth': (
        '<degree>',
        'To astrometry.cfg file. If no scale estimate is given, use these '
        'limits on field width in deg.'
    ),
    'maxwidth': (
        '<degree>',
        'To astrometry.cfg file. If no scale estimate is given, use these '
        'limits on field width in deg.'
    ),
    'index-dir': (
        '<string or list(string)>',
        'Add a location of astrometry.net index files.'
        ' ``astrometry-engine`` will search index files in all listed folders.'
        ' Additive, do not override defaults.'
    )
}


def get_options_help() -> str:
    """Describe the supported solve-field options.

    Returns
    -------
    help_text
        A reStructuredText Notes section listing option names, accepted values
        and their meanings. This only formats local definitions; it does not
        invoke solve-field.
    """
    s = "Notes\n-----\n\nThe supported options are:\n"
    for k, v in solve_field_params.items():
        s += f"    {k}: {v[0]}\n        {v[1]}\n"
    return s


get_options_help.__doc__ = _cleandoc(get_options_help.__doc__) + '\n\n' + get_options_help()


class AstrometryNetUnsolvedField(CalledProcessError):
    """Raised if Astrometry.net could not solve the field."""

    def __init__(self, path: _PathLike) -> None:
        """Record the input path whose astrometric solution failed."""
        self.path = path

    def __str__(self) -> str:
        """Describe the unsolved field using its input path."""
        return f"{self.path}: could not solve field"


def _parse_angle(angle: str | float | Angle, unit: _UnitLike=None) -> float:
    """Transform angle in float."""
    # plate scale
    # radius
    # bare ra and dec
    if isinstance(angle, str):
        try:
            angle = Angle(angle)
        except UnitsError as e:
            if unit is not None:
                angle = Angle(angle, unit=unit)
            else:
                raise e
    if check_number(angle) and unit is not None:
        angle = Angle(angle, unit=unit)
    if isinstance(angle, Angle):
        angle = angle.degree
    if not isinstance(angle, float):
        raise ValueError(f'{angle} (type {type(angle)})not recognized as a '
                         'valid angle.')
    return float(angle)


def _parse_coordinates(options: dict[str, _t.Any]) -> list[str]:
    """Parse filed center coordinates."""
    coord = options.pop('center', None)
    ra = options.pop('ra', None)
    dec = options.pop('dec', None)

    if coord is not None and (ra is not None or dec is not None):
        raise ValueError('Field center defined by `coord` conflicts with `ra`'
                         ' and `dec` fields')
    if coord is None and (ra is None or dec is None):
        logger.info('Astrometry solving with blind field center.')
        return []

    # coord can by a tuple of ra, dec
    if isinstance(coord, (list, tuple)):
        ra, dec = coord
    elif isinstance(coord, SkyCoord):
        ra = coord.ra.degree
        dec = coord.dec.degree
    if isinstance(ra, str):
        ra = _parse_angle(ra, 'hourangle')
    if isinstance(dec, str):
        dec = _parse_angle(dec, 'degree')

    return ['--ra', str(ra), '--dec', str(dec)]


def _parse_pltscl(options: dict[str, _t.Any]) -> list[str]:
    """Parse plate scale."""
    low = options.pop('scale-low', None)
    hi = options.pop('scale-high', None)
    units = options.pop('scale-units', None)
    pltscl = options.pop('plate-scale', None)
    tolerance = options.pop('scale-tolerance', 0.2)

    if pltscl is not None and (low is not None or hi is not None):
        raise ValueError("'plate-scale' is in conflict with 'scale-low' and"
                         " 'scale-hi' options. Use one or another.")
    if low is not None and hi is not None and units is None:
        raise ValueError("When using 'scale-low' and 'scale-hi' options you "
                         "must specify 'scale-units' too.")

    if pltscl is not None:
        low = pltscl*(1-tolerance)
        hi = pltscl*(1+tolerance)
        units = units or 'arcsecperpix'

    if low is None or hi is None:
        return []

    return ['--scale-low', str(low), '--scale-high', str(hi),
            '--scale-units', str(units)]


def _parse_crpix(options: dict[str, _t.Any]) -> list[str | float]:
    """Consume reference-pixel options and build their command arguments.

    The presence of ``'crpix-center'`` selects the image center. Otherwise both
    ``'crpix-x'`` and ``'crpix-y'`` are required to emit an explicit position.
    Combining the center option with both coordinates raises ValueError.
    """
    center = False
    if 'crpix-center' in options:
        center = True
        options.pop('crpix-center')

    crpix_x = options.pop('crpix-x', None)
    crpix_y = options.pop('crpix-y', None)

    if center and (crpix_x is not None and crpix_y is not None):
        raise ValueError('crpix-center is in conflicts with crpix-x and '
                         'crpix-y. Use one or another.')

    if center:
        return ['--crpix-center']
    elif crpix_x is not None and crpix_y is not None:
        return ['--crpix-x', float(crpix_x), '--crpix-y', float(crpix_y)]
    return []


class AstrometricSolution():
    """Store a WCS, its FITS header and optional matched-source table.

    Parameters
    ----------
    header
        FITS header from which the WCS is constructed.
    correspondences
        A table containing the matched correspondences between the (x, y)
        positions and the matched (ra, dec) catalog coordinates.

    Examples
    --------
    >>> from astropop.astrometry import AstrometricSolution, wcs_from_coords
    >>> known = wcs_from_coords(16, 16, 120., -30., 0.34, north='top')

    >>> # Wrap a known WCS locally; this does not call Astrometry.net.
    >>> solution = AstrometricSolution(known.to_header())
    >>> print(solution.wcs.pixel_n_dim, solution.correspondences)
    2 None
    """

    _wcs = None
    _header = None
    _corr = None

    def __init__(
        self,
        header: _HeaderLike,
        correspondences: Table | _npt.NDArray | None=None,
    ) -> None:
        """Build a WCS and copy the header and optional correspondence table."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._wcs = WCS(header, relax=True)
            self._header = fits.Header(header)
            if correspondences is not None:
                self._corr = Table(correspondences)

    @property
    def wcs(self) -> WCS:
        """World Coordinate System `~astropy.wcs.WCS` solution."""
        return copy.deepcopy(self._wcs)

    @property
    def header(self) -> fits.Header:
        """`~astropy.io.fits.Header` containing the astrometric solution."""
        return copy.deepcopy(self._header)

    @property
    def correspondences(self) -> Table | None:
        """Copy of the matched-source table, or None if none was supplied.

        No matching or residual calculation is performed by this property.
        An Astrometry.net correspondence table includes the following columns:

        - field_x and field_y
            FITS one-based image coordinates of each detected star.
        - index_x and index_y
            Image-plane coordinates of matched index stars.
        - field_ra and field_dec
            Solved right ascension and declination of the detected stars.
        - index_ra and index_dec
            Reference right ascension and declination of the matched index stars.

        These matches contributed to solving the image. Assess accuracy using
        independent reference stars as well as residuals of these matches.
        """
        return copy.deepcopy(self._corr)


class AstrometrySolver():
    """Solve image or source-list astrometry with a local Astrometry.net command.

    Parameters
    ----------
    solve_field
        Existing solve-field executable path. The default is found on PATH.
        Astrometry.net version 0.95 or newer is required.
    defaults
        Default solve-field options, using names without leading dashes.
        These update the initial no-plots and overwrite flag options. Per-call
        options supplied to solve_field override the defaults. See
        `~astropop.astrometry.astrometrynet.get_options_help` for supported keys.
    keep_files
        Keep auxiliary solver products instead of removing them after solving.
        Default is False.

    Notes
    -----
    Index installation and Astrometry.net's own configuration are external
    setup steps. This wrapper has no config or config_file constructor option
    and does not synthesize astrometry.cfg files from arbitrary parameters.
    """

    def __init__(
        self,
        solve_field: _PathLike | SolveFieldCommand | None=_solve_field,
        defaults: dict[str, _t.Any] | None=None,
        keep_files: bool=False,
    ) -> None:
        """Configure solver defaults, validate its version and set file retention."""
        # declare the defaults here to be safer
        self._defaults = {'no-plots': None, 'overwrite': None}
        if defaults is None:
            defaults = {}
        self._defaults.update(defaults)

        self._command = SolveFieldCommand(solve_field)
        if self._command.version < version.parse('0.95'):
            raise ValueError('Astrometry.net version must be at least 0.95.')
        self._keep = keep_files

    def solve_field(
        self,
        filename: _PathLike,
        options: dict[str, _t.Any] | None=None,
        output_dir: _PathLike | None=None,
        **kwargs: _t.Any,
    ) -> AstrometricSolution:
        """Solve an image or source-list file using local Astrometry.net.

        Parameters
        ----------
        filename
            Name of the file to be solved. Can be a fits image or a xyls
            sources file.
        options
            Dictionary of ``solve-field`` options. See
            `~astropop.astrometry.astrometrynet.get_options_help` for all
            available options. The most useful are:
            - center or ra and dec: field center
            - radius: maximum search radius
            - scale-low and scale-high: bounds in the specified scale-units
            - tweak-order: SIP order to fit
        output_dir
            Directory to output all solved files.
        **kwargs
            Additional keyword arguments to be passed to
            `~astropop.py_utils.run_command`.

        Returns
        -------
        solution
            Astrometric solution class containing the solved header,
            the `~astropy.wcs.WCS` solved class and the correspondence table
            between the sources and the catalog.

        Raises
        ------
        AstrometryNetUnsolvedField
            The solver did not produce a solved-field marker.
        subprocess.CalledProcessError
            The external command failed with a nonzero exit code.
        KeyError
            An option is not supported by this wrapper.
        ValueError
            Mutually exclusive options were supplied.
        """
        if options is None:
            options = {}

        n_opt = copy.copy(self._defaults)
        n_opt.update(options)

        solved_header, coorespond = self._run_solver(filename,
                                                     options=n_opt,
                                                     output_dir=output_dir,
                                                     **kwargs)

        return AstrometricSolution(header=solved_header,
                                   correspondences=coorespond)

    def print_options_help(self) -> None:
        """Print the available options for ``solve-field``."""
        print(get_options_help())

    def _get_output_dir(
        self,
        root: str,
        output_dir: _PathLike | None,
    ) -> tuple[_PathLike, bool]:
        """Check output directory and create the temporary directory."""
        if output_dir is None:
            output_dir = mkdtemp(prefix=root + '_', suffix='_astrometry.net')
            tmp_dir = True
            os.makedirs(output_dir, exist_ok=True)
        else:
            tmp_dir = False
        return output_dir, tmp_dir

    def _parse_options(self, options: dict[str, _t.Any]) -> list[str | float]:
        """Parse and check all known options."""
        for key in options.keys():
            if key not in solve_field_params:
                raise KeyError(f'option {key} not supported.')

        args = []
        args += _parse_coordinates(options)  # parse center
        args += _parse_pltscl(options)  # parse plate scale
        if 'radius' in options:  # parse radius
            args += ['--radius',
                     str(_parse_angle(options.pop('radius', None)))]
        args += _parse_crpix(options)  # parse crpix

        for key, value in options.items():
            if value is None:
                args.append(f'--{key}')
            else:
                if isinstance(value, (list, tuple)):
                    value = ",".join(value)
                args += [f'--{key}', str(value)]

        return args

    def _get_args(
        self,
        root: str,
        fname: _PathLike,
        options: dict[str, _t.Any],
        output_dir: _PathLike,
        correspond: _PathLike,
    ) -> list[_PathLike | float]:
        """Build solve-field arguments for the input and output products.

        Parsing consumes recognized entries from options. The root argument is
        retained for compatibility but is not used to assemble the command.
        """
        args = [fname, '--dir', output_dir]
        args += self._parse_options(options)
        args += ['--corr', correspond]
        return args

    def _run_solver(
        self,
        filename: _PathLike,
        options: dict[str, _t.Any],
        output_dir: _PathLike | None=None,
        **kwargs: _t.Any,
    ) -> tuple[fits.Header, Table]:
        """Run the astrometry.net localy using the given params.

        STDOUT and STDERR can be stored in variables for better check after
        using kwargs.
        """
        basename = os.path.basename(filename)
        root, _ = os.path.splitext(basename)
        output_dir, tmp_dir = self._get_output_dir(root, output_dir)
        # Ensure the output directory exist
        os.makedirs(output_dir, exist_ok=True)
        correspond = os.path.join(output_dir, root + '.corr')
        solved_file = os.path.join(output_dir, root + '.solved')
        args = self._get_args(root, filename, options, output_dir=output_dir,
                              correspond=correspond)

        try:
            process, _, _ = self._command.run(args, **kwargs)
            # .solved file must exist and contain a binary one
            with open(solved_file, 'rb') as fd:
                if ord(fd.read()) != 1:
                    raise AstrometryNetUnsolvedField(filename)
            solved_wcs_file = os.path.join(output_dir, root + '.wcs')
            logger.debug('Loading solved header from %s', solved_wcs_file)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                solved_header = fits.getheader(solved_wcs_file, 0)
                corr = Table.read(correspond)

            # remove the tree if the file is temporary and not set to keep
            if not self._keep and tmp_dir:
                shutil.rmtree(output_dir)

            return solved_header, corr

        except CalledProcessError as e:
            if not self._keep and tmp_dir:
                shutil.rmtree(output_dir)
            raise e

        # If .solved file doesn't exist or contain one
        except (IOError, AstrometryNetUnsolvedField):
            if not self._keep and tmp_dir:
                shutil.rmtree(output_dir)
            raise AstrometryNetUnsolvedField(filename)


def create_xyls(
    fname: _PathLike,
    x: _npt.ArrayLike,
    y: _npt.ArrayLike,
    flux: _npt.ArrayLike,
    imagew: int,
    imageh: int,
    header: _HeaderLike | None=None,
    dtype: _npt.DTypeLike='f8',
) -> None:
    """Create and save the xyls file to run in astrometry.net

    Parameters
    ----------
    fname
        The path to save the .xyls file.
    x
        Source x positions in one-based FITS pixels, written unchanged.
    y
        Source y positions in one-based FITS pixels, written unchanged.
    flux
        Estimated flux of the sources.
    imagew
        Width of the original image. ``IMAGEW`` header field of .xyls
    imageh
        Height of the original image. ``IMAGEH`` header field of .xyls
    header
        Optional header passed to the primary FITS HDU. Image dimensions are
        stored separately on the binary-table HDU.
    dtype
        Data type of the fields. Default: 'f8'

    Examples
    --------
    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> from astropy.io import fits
    >>> from astropop.astrometry import create_xyls
    >>> with TemporaryDirectory() as directory:
    ...     filename = Path(directory) / 'sources.xyls'
    ...     # Positions are already one-based FITS pixels; rows sort by flux.
    ...     create_xyls(filename, [11., 21.], [16., 26.], [100., 200.], 32, 32)
    ...     with fits.open(filename) as hdus:
    ...         print(hdus[1].data['x'].tolist())
    [21.0, 11.0]
    """
    head = {'IMAGEW': imagew,
            'IMAGEH': imageh}
    sort = np.argsort(flux)
    xyls = np.array(list(zip(x, y, flux)),
                    np.dtype([('x', dtype), ('y', dtype), ('flux', dtype)]))
    f = fits.HDUList([fits.PrimaryHDU(header=header),
                      fits.BinTableHDU(xyls[sort[::-1]],
                                       header=fits.Header(head))])
    logger.debug('Saving .xyls to %s', fname)
    f.writeto(fname, output_verify="ignore")


def solve_astrometry_xy(
    x: _npt.ArrayLike,
    y: _npt.ArrayLike,
    flux: _npt.ArrayLike,
    width: int,
    height: int,
    image_header: _HeaderLike | None=None,
    options: dict[str, _t.Any] | None=None,
    command: _PathLike | SolveFieldCommand | None=_solve_field,
    **kwargs: _t.Any,
) -> AstrometricSolution:
    """Solve astrometry from a (x,y) sources list using astrometry.net.

    Parameters
    ----------
    x
        Source x positions in FITS one-based pixels. Add one to zero-based
        detection coordinates.
    y
        Source y positions in FITS one-based pixels. Add one to zero-based
        detection coordinates.
    flux
        Estimated fluxes of the sources. A higher value indicates a brighter
        star.
    width
        Image width in pixels (number of columns).
    height
        Image height in pixels (number of rows).
    image_header
        Original image header.
    options
        Dictionary of ``solve-field`` options. See
        `~astropop.astrometry.astrometrynet.get_options_help` for all
        available options. The most useful are:
        - center or ra and dec: field center
        - radius: maximum search radius
        - scale: pixel scale in arcsec/pixel
        - tweak-order: SIP order to fit
        - index-dir: path to index files
    command
        Full path of astrometry.net ``solve-field`` command.

    Returns
    -------
    result
        Astrometric solution of the field.
    """
    options = options or {}
    if image_header is not None:
        image_header, _ = extract_header_wcs(image_header)
    else:
        image_header = fits.Header()

    f = NamedTemporaryFile(suffix='.xyls')
    create_xyls(f.name, x, y, flux, width, height,
                header=image_header)
    solver = AstrometrySolver(solve_field=command)
    options.update({'width': width, 'height': height})
    return solver.solve_field(f.name, options=options, **kwargs)


def solve_astrometry_framedata(
    frame: _FrameData,
    options: dict[str, _t.Any] | None=None,
    command: _PathLike | SolveFieldCommand | None=_solve_field,
    **kwargs: _t.Any,
) -> AstrometricSolution:
    """Solve astrometry from a `~astropop.framedata.FrameData`.

    Parameters
    ----------
    frame
        Image to be solved.
    options
        Dictionary of ``solve-field`` options. See
        `~astropop.astrometry.astrometrynet.get_options_help` for all
        available options. The most useful are:
        - center or ra and dec: field center
        - radius: maximum search radius
        - scale: pixel scale in arcsec/pixel
        - tweak-order: SIP order to fit
        - index-dir: path to index files
    command
        Full path of astrometry.net ``solve-field`` command.

    Returns
    -------
    result
        Astrometric solution ot the field.
    """
    f = NamedTemporaryFile(suffix='.fits')
    frame.write(f.name, no_fits_standard_units=True)
    options = options or {}
    solver = AstrometrySolver(solve_field=command)
    return solver.solve_field(f.name, options=options, **kwargs)


def solve_astrometry_image(
    filename: _PathLike,
    options: dict[str, _t.Any] | None=None,
    command: _PathLike | SolveFieldCommand | None=_solve_field,
    **kwargs: _t.Any,
) -> AstrometricSolution:
    """Solve astrometry from an image using astrometry.net.

    Parameters
    ----------
    filename
        Path to the image to solve.
    options
        Dictionary of ``solve-field`` options. See
        `~astropop.astrometry.astrometrynet.get_options_help` for all
        available options. The most useful are:
        - center or ra and dec: field center
        - radius: maximum search radius
        - scale: pixel scale in arcsec/pixel
        - tweak-order: SIP order to fit
        - index-dir: path to index files
    command
        Full path of astrometry.net ``solve-field`` command.

    Returns
    -------
    result
        Astrometric solution ot the field.
    """
    options = options or {}
    solver = AstrometrySolver(solve_field=command)
    return solver.solve_field(filename, options=options, **kwargs)


def solve_astrometry_hdu(
    hdu: _ImageHDU,
    options: dict[str, _t.Any] | None=None,
    command: _PathLike | SolveFieldCommand | None=_solve_field,
    **kwargs: _t.Any,
) -> AstrometricSolution:
    """Solve astrometry from a `~astropy.io.fits.ImageHDU` using astrometry.net.

    Parameters
    ----------
    hdu
        FITS hdu containing the image to solve.
    options
        Dictionary of ``solve-field`` options. See
        `~astropop.astrometry.astrometrynet.get_options_help` for all
        available options. The most useful are:
        - center or ra and dec: field center
        - radius: maximum search radius
        - scale: pixel scale in arcsec/pixel
        - tweak-order: SIP order to fit
        - index-dir: path to index files
    command
        Full path of astrometry.net ``solve-field`` command.

    Returns
    -------
    result
        Astrometric solution ot the field.
    """
    options = options or {}
    hdu.header, _ = extract_header_wcs(hdu.header)
    f = NamedTemporaryFile(suffix='.fits')
    hdu = fits.PrimaryHDU(hdu.data, header=hdu.header)
    hdu.writeto(f.name)
    solver = AstrometrySolver(solve_field=command)
    return solver.solve_field(f.name, options=options, **kwargs)
