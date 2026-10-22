.. include:: ../../references.txt

Astrometry Calibration
======================

Astrometric calibration assigns sky coordinates to image pixels. Astropop wraps
the local Astrometry.net ``solve-field`` executable and returns a WCS, its FITS
header and a table of matched stars. It does not install the solver or reference
indexes, and solving does not resample pixels or modify the input frame.

Use :doc:`../../tutorials/astrometry_catalogs/astrometry_net` for an observed-image workflow and
:doc:`../../tutorials/astrometry_catalogs/astrometric_calibration` for an executable offline example
with independent reference-star residuals. This page describes setup and the
wrapper's input, output and failure contracts.

.. plot::
    :include-source: false
    :alt: Five labeled stars on a detector grid and the same stars mapped into east and north sky offsets by a rotated TAN WCS. East increases left in the sky panel.
    :caption: A known TAN WCS maps this asterism at 0.34 arcsec per pixel. Grid lines represent the same detector rows and columns in both panels. This illustrates coordinate transformation, not an Astrometry.net solve.

    import matplotlib.pyplot as plt
    import numpy as np

    from astropy.wcs import WCS

    camera = WCS(naxis=2)
    camera.wcs.crpix = [64.5, 64.5]
    camera.wcs.crval = [150., -30.]
    camera.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    angle = np.deg2rad(15)

    # Combine the rotation and plate scale in the pixel-to-sky matrix.
    camera.wcs.cd = 0.34 / 3600 * np.array([[-np.cos(angle), np.sin(angle)],
                                           [np.sin(angle), np.cos(angle)]])
    x = np.array([20., 39., 65., 92., 110.])
    y = np.array([28., 96., 60., 35., 103.])
    center = camera.pixel_to_world(63.5, 63.5)

    # Use spherical offsets rather than subtracting right ascensions directly.
    east, north = center.spherical_offsets_to(camera.pixel_to_world(x, y))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
    grid = np.linspace(0, 127, 64)
    for fixed in np.linspace(0, 127, 5):
        for xx, yy in [(grid, np.full_like(grid, fixed)), (np.full_like(grid, fixed), grid)]:
            axes[0].plot(xx, yy, color='0.85', linewidth=1)
            dx, dy = center.spherical_offsets_to(camera.pixel_to_world(xx, yy))
            axes[1].plot(dx.arcsec, dy.arcsec, color='0.85', linewidth=1)

    for ax, xx, yy in [(axes[0], x, y), (axes[1], east.arcsec, north.arcsec)]:
        ax.plot(xx, yy, 'C0*', markersize=11)
        for name, sx, sy in zip('ABCDE', xx, yy):
            ax.annotate(name, (sx, sy), xytext=(5, 5), textcoords='offset points')
        ax.set_aspect('equal')

    axes[0].set(xlim=(-5, 132), ylim=(-5, 132), xlabel='x / column (pixel)',
                ylabel='y / row (pixel)', title='Detector coordinates; origin = 0')
    axes[1].set(xlim=(28, -28), ylim=(-28, 28), xlabel='East offset (arcsec)',
                ylabel='North offset (arcsec)', title='Sky offsets; east increases left')

About astrometry.net
--------------------

Astrometry.net identifies stellar patterns against precomputed reference
indexes. Approximate pointing and angular scale can constrain the search;
incorrect constraints can exclude the correct solution. See the
`upstream solver guide <https://astrometry.net/doc/readme.html>`_ for solving
and index selection. Astropop's wrapper runs the local executable; it does not
submit images to the hosted service.

Installing astrometry.net
^^^^^^^^^^^^^^^^^^^^^^^^^

Follow the `upstream installation instructions
<https://astrometry.net/doc/build.html>`_ for your platform. Those instructions
include running the Linux build in WSL on Windows. Astropop requires
``solve-field`` version 0.95 or newer. Check the executable in the environment
that will run Python:

.. code-block:: bash

    solve-field --version
    solve-field --help

The default executable is discovered on PATH when the module is imported.
Restart the Python session after changing PATH, or pass an explicit executable
path through ``AstrometrySolver(solve_field=...)``. The helper functions instead
name this argument ``command``. Installing the Python package alone does not
satisfy this external dependency.

Index files
^^^^^^^^^^^

Choose indexes covering the field's sky region and angular scale using the
`index selection instructions
<https://astrometry.net/doc/readme.html#getting-index-files>`_. Configure their
location in Astrometry.net, or supply ``'index-dir'`` in the wrapper's options.
For example, ``{'index-dir': '/data/astrometry-indexes'}`` adds a local directory
to the search. Its contents must be actual index files, not the science images.
Configuration-file directives such as ``add_path`` are not wrapper option keys.

Astropop wrapper usage
----------------------

Choose the input interface that matches your data. All solver interfaces return
an |AstrometricSolution|. Assign its WCS to a frame explicitly before writing
that frame if the saved product should contain the solution.

Solution Format
^^^^^^^^^^^^^^^

The solution exposes three properties, each returned as a copy:

* ``wcs``: an Astropy |WCS| for pixel/sky transformations.
* ``header``: the FITS header describing that WCS.
* ``correspondences``: the solver's matched-source table, or ``None`` when a
  container was constructed without a table.

A solution can also hold an already known WCS. This example constructs a TAN
mapping, without invoking a solver or claiming to fit observations::

    >>> from astropy.wcs import WCS
    >>> from astropop.astrometry import AstrometricSolution

    >>> known = WCS(naxis=2)
    >>> known.wcs.crpix = [32.5, 24.5]
    >>> known.wcs.crval = [132.5, 11.83]

    >>> # Convert the plate scale from arcseconds to degrees per pixel.
    >>> known.wcs.cdelt = [-0.5 / 3600, 0.5 / 3600]
    >>> known.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    >>> solution = AstrometricSolution(known.to_header())

    >>> # Subtract one from FITS CRPIX to address the same center with this API.
    >>> center = solution.wcs.pixel_to_world(31.5, 23.5)

    >>> print(f'Center: {center.ra.deg:.3f}, {center.dec.deg:.3f} deg')
    Center: 132.500, 11.830 deg
    >>> print('Correspondences:', solution.correspondences)
    Correspondences: None

FITS CRPIX values are one-based; ``pixel_to_world`` accepts zero-based pixels.
See :doc:`../../getting_started/conventions` before mixing image detections, source-list
files and WCS calls. A plausible WCS grid is not an accuracy test. Inspect
residuals across the detector using independent reference stars; solver
correspondences participated in the fit and are not an independent validation
set. The offline tutorial shows both the residual vectors and summary values.

Class-like interface
^^^^^^^^^^^^^^^^^^^^

Use |AstrometrySolver| for repeated calls with shared defaults. Per-call
``options`` override ``defaults``; names omit leading dashes, and flag-only
options use ``None``. The following is an external-command recipe requiring
your own calibrated FITS image and installed indexes:

.. code-block:: python

    from astropop.astrometry import AstrometrySolver

    solver = AstrometrySolver(defaults={'cpulimit': 60})
    solution = solver.solve_field(
        'reduced/science.fits',
        options={'ra': 132.5, 'dec': 11.83, 'radius': 1.0,
                 'scale-low': 0.4, 'scale-high': 0.6,
                 'scale-units': 'arcsecperpix', 'crpix-center': None})

    print('Matched stars:', len(solution.correspondences))

Numbers supplied for ``'ra'``, ``'dec'`` and ``'radius'`` are in degrees.
Unitless sexagesimal RA strings are interpreted as hour angles; declination
strings use degrees. Use explicit units to avoid ambiguous numeric strings:
``'ra': '08h50m00s'`` and ``'dec': '+11d49m48s'`` describe the same center as
above. A ``'center'`` SkyCoord is another alternative, but cannot be combined
with ``'ra'`` or ``'dec'``. Both RA and Dec must be supplied to constrain the
center; specifying only one leaves the center search blind.

The default temporary solver directory is removed after success and handled
solver failures. ``keep_files=True`` retains it for diagnosis. An explicit
``output_dir`` is retained regardless of that setting. Use a dedicated output
directory: defaults include ``'overwrite': None`` for solver products.
There is no ``config`` property or ``config_file`` constructor parameter.

.. _astrometry-failure-handling:

Handle failures and retain useful logs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Capture output without embedding a long solver transcript in the application.
The specialized unsolved-field exception subclasses ``CalledProcessError``, so
catch it first:

.. code-block:: python

    from subprocess import CalledProcessError
    from astropop.astrometry.astrometrynet import AstrometryNetUnsolvedField

    stdout, stderr = [], []
    try:
        solver = AstrometrySolver(keep_files=True)
        solution = solver.solve_field('reduced/science.fits',
                                      options={'cpulimit': 60},
                                      stdout=stdout, stderr=stderr)
    except FileNotFoundError as error:
        print('Check the executable and input path:', error)
    except AstrometryNetUnsolvedField as error:
        print('No solution:', error)
        print('\n'.join(stdout[-10:] + stderr[-10:]))
    except CalledProcessError as error:
        print('Solver process failed with exit code:', error.returncode)
        print('\n'.join(stderr[-10:]))
    else:
        print('Solved axes:', solution.wcs.world_axis_physical_types)

An unsupported option raises ``KeyError``; conflicting options or an unsupported
solver version raise ``ValueError``. Correct these configuration errors before
retrying. For an unsolved field, inspect index coverage, source quality and
position/scale limits. Preserve the input and diagnostic products instead of
silently treating an unsolved image as calibrated. This recipe is not executed
by the offline documentation tests.

Type-specific helper functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The convenience functions prepare temporary input files and run the same solver:

* `~astropop.astrometry.solve_astrometry_image`: an image filename.
* `~astropop.astrometry.solve_astrometry_framedata`: a |FrameData| object.
* `~astropop.astrometry.solve_astrometry_hdu`: a FITS image HDU.
* `~astropop.astrometry.solve_astrometry_xy`: source ``x``, ``y`` and ``flux``
  arrays plus image ``width`` and ``height``.

Each accepts ``options``, an optional executable ``command``, and keyword
arguments forwarded to |run_command|, including output lists and logging levels.
For source lists, supply FITS one-based coordinates: add one to Astropop's
zero-based detection coordinates exactly once. Image shape is
``(height, width)``. See :doc:`../../tutorials/astrometry_catalogs/astrometry_net` for a complete example
of each input path and attaching the resulting WCS.

For the same calibrated exposure, choose one of the following input paths.
These are external-solver recipes, requiring the executable, indexes and the
specified local files; they are not offline doctests. Reuse one set of options
so changing the input container does not change the intended sky search:

.. code-block:: python

    from astropop.astrometry import (solve_astrometry_image,
                                    solve_astrometry_framedata,
                                    solve_astrometry_hdu, solve_astrometry_xy)
    from astropop.framedata import read_framedata
    from astropy.io import fits
    from astropy.table import Table

    filename = 'reduced/science.fits'
    options = {'ra': 132.5, 'dec': 11.83, 'radius': 1., 'cpulimit': 60}

    # Use a filename when the calibrated image is already on disk.
    from_file = solve_astrometry_image(filename, options=options)

    print('File solution axes:', from_file.wcs.world_axis_physical_types)

    # Use FrameData when the calibrated frame is already in memory.

    frame = read_framedata(filename)
    from_frame = solve_astrometry_framedata(frame, options=options)

    print('Frame solution axes:', from_frame.wcs.world_axis_physical_types)

    # Use an HDU when working directly with Astropy FITS containers.

    with fits.open(filename) as hdus:
        from_hdu = solve_astrometry_hdu(hdus[0], options=options)

    print('HDU solution axes:', from_hdu.wcs.world_axis_physical_types)

    # Use a measured source list to bypass image-source extraction by the solver.

    sources = Table.read('reduced/photometry.ecsv')  # zero-based x,y and measured flux
    from_sources = solve_astrometry_xy(
        sources['x'] + 1, sources['y'] + 1, sources['flux'],
        width=frame.shape[1], height=frame.shape[0], options=options)

    print('Source-list matches:', len(from_sources.correspondences))

For a multi-extension image, load the intended science HDU rather than an
empty primary HDU. The helper functions return the solution separately; they
do not attach it to ``frame``. Assign ``frame.wcs = from_frame.wcs`` explicitly
when that is the desired product. ``solve_astrometry_xy`` assumes the supplied
positions and dimensions describe the same image geometry.

Write a solver source list
^^^^^^^^^^^^^^^^^^^^^^^^^^

``create_xyls`` writes an Astrometry.net FITS source-list file without running
the solver. Supply matching-length arrays of one-based pixel positions and
fluxes from the same image, plus its width and height. The binary table has
``x``, ``y`` and ``flux`` fields sorted by decreasing flux; its header records
``IMAGEW`` and ``IMAGEH``. Use it to inspect or pass a measured list to an
external solver workflow::

    >>> from tempfile import TemporaryDirectory
    >>> from pathlib import Path
    >>> from astropy.io import fits
    >>> from astropop.astrometry import create_xyls

    >>> with TemporaryDirectory() as directory:
    ...     path = Path(directory) / 'sources.xyls'
    ...     create_xyls(path, x=[3., 8., 5.], y=[6., 4., 9.],
    ...                 flux=[100., 300., 200.], imagew=32, imageh=24)
    ...     with fits.open(path) as hdus:
    ...         print('Columns:', hdus[1].columns.names)
    ...         print('Sorted flux:', hdus[1].data['flux'].tolist())
    ...         print('Image width and height:', hdus[1].header['IMAGEW'], hdus[1].header['IMAGEH'])
    Columns: ['x', 'y', 'flux']
    Sorted flux: [300.0, 200.0, 100.0]
    Image width and height: 32 24

The default table dtype is ``'f8'``. An optional ``header`` supplies the primary
HDU header; this is distinct from the image-size cards on the table HDU.
The writer does not overwrite existing files. ``solve_astrometry_xy`` creates
its own temporary source list, so an application using that helper need not
call ``create_xyls`` separately.

Prepare coordinates and an approximate WCS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``guess_coordinates`` accepts RA/Dec scalars or equal-length arrays. Numeric
values, including numeric strings, are interpreted in degrees. If numeric
conversion fails, the helper tries RA as hour angle and Dec as degrees.
``skycoord=True`` returns a SkyCoord; ``skycoord=False`` returns numerical
RA/Dec in degrees. Keep input formats consistent and prefer an explicitly
unit-bearing SkyCoord when the format is already known::

    >>> from astropop.astrometry import guess_coordinates, wcs_from_coords

    >>> decimal = guess_coordinates('132.5', '11.83')
    >>> sexagesimal = guess_coordinates('08:50:00', '+11:49:48')

    >>> print(f'Numeric strings: {decimal.ra.deg:.3f}, {decimal.dec.deg:.3f} deg')
    Numeric strings: 132.500, 11.830 deg
    >>> print(f'Sexagesimal strings: {sexagesimal.ra.deg:.3f}, {sexagesimal.dec.deg:.3f} deg')
    Sexagesimal strings: 132.500, 11.830 deg

    >>> ra, dec = guess_coordinates([132.5, 133.], [11.83, 12.], skycoord=False)

    >>> print('Degree arrays:', ra.tolist(), dec.tolist())
    Degree arrays: [132.5, 133.0] [11.83, 12.0]

``wcs_from_coords`` constructs a TAN WCS from a reference pixel, pointing,
plate scale in arcseconds per pixel and north direction. Use it when an
approximate geometric mapping is sufficient, for example to initialize a
calibration. It does not solve an observed field or measure optical distortion.
The reference pixel supplied to this helper is one-based FITS ``CRPIX``, unlike
Astropy's high-level pixel coordinates::

    >>> approximate = wcs_from_coords(32.5, 24.5, 132.5, 11.83, 0.5, north='top')
    >>> center = approximate.pixel_to_world(31.5, 23.5)

    >>> print(f'Approximate reference position: {center.ra.deg:.3f}, {center.dec.deg:.3f} deg')
    Approximate reference position: 132.500, 11.830 deg

``north='top'`` corresponds to 90 degrees counterclockwise from positive x;
``'right'``, ``'left'`` and ``'bottom'`` correspond to 0, 180 and 270 degrees.
A numeric ``north`` gives that angle directly. ``flip='ra'`` flips the RA axis,
``'dec'`` flips Dec, ``'all'`` flips both, and ``None`` applies no extra flip.
See :doc:`../../getting_started/conventions` for a nonsquare pixel-grid example and the
relationship between display orientation and coordinate origins.

Supported ``solve-field`` options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Only supported wrapper keys are accepted. Inspect their descriptions locally
without invoking a solver:

.. currentmodule:: None

.. py:function:: astropop.astrometry.astrometrynet.get_options_help()

    Return a formatted string listing supported solve-field option names and
    descriptions. This inspects local metadata without invoking solve-field.

.. code-block:: python

    from astropop.astrometry.astrometrynet import get_options_help

    print(get_options_help())

The same help is available through
`~astropop.astrometry.AstrometrySolver.print_options_help`. Configuration-file
entries and arbitrary ``solve-field`` options are not automatically accepted.




Reference/API
-------------

.. automodapi:: astropop.astrometry
    :no-inheritance-diagram:

External command interface
--------------------------

The convenience solver functions use ``SolveFieldCommand`` to locate and invoke
Astrometry.net. Supply a custom command path when ``solve-field`` is not on
``PATH``. Constructing this wrapper checks the external executable; it is not
required for using image arithmetic or a manually constructed WCS.

.. autoclass:: astropop.astrometry.astrometrynet.SolveFieldCommand
    :members:


.. currentmodule:: None
