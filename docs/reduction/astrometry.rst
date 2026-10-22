.. include:: ../references.txt

Astrometry Calibration
======================

Astrometric calibration assigns sky coordinates to image pixels. Astropop wraps
the local Astrometry.net ``solve-field`` executable and returns a WCS, its FITS
header and a table of matched stars. It does not install the solver or reference
indexes, and solving does not resample pixels or modify the input frame.

Use :doc:`../tutorials/astrometry_net` for an observed-image workflow and
:doc:`../tutorials/astrometric_calibration` for an executable offline example
with independent reference-star residuals. This page describes setup and the
wrapper's input, output and failure contracts.

.. plot:: _ext/task_figures.py plot_wcs
    :include-source: False
    :alt: Five labeled stars on a detector grid and the same stars mapped into east and north sky offsets by a rotated TAN WCS. East increases left in the sky panel.
    :caption: A known TAN WCS maps this asterism at 0.34 arcsec per pixel. Grid lines represent the same detector rows and columns in both panels. This illustrates coordinate transformation, not an Astrometry.net solve.

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
    >>> known.wcs.cdelt = [-0.5 / 3600, 0.5 / 3600]
    >>> known.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    >>> solution = AstrometricSolution(known.to_header())
    >>> center = solution.wcs.pixel_to_world(31.5, 23.5)
    >>> print(f'Center: {center.ra.deg:.3f}, {center.dec.deg:.3f} deg')
    Center: 132.500, 11.830 deg
    >>> print('Correspondences:', solution.correspondences)
    Correspondences: None

FITS CRPIX values are one-based; ``pixel_to_world`` accepts zero-based pixels.
See :doc:`../dev/pixel_convention` before mixing image detections, source-list
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
``(height, width)``. See :doc:`../tutorials/astrometry_net` for a complete example
of each input path and attaching the resulting WCS.

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

Astrometry API
--------------

See :doc:`../reference/astrometry` for the API reference.

.. raw:: html

    <span id="id1"></span>
    <span id="astropop-astrometry-package"></span>
    <span id="astropop.astrometry.astrometrynet.SolveFieldCommand"></span>
    <span id="astropop.astrometry.astrometrynet.SolveFieldCommand.run"></span>
    <span id="astropop.astrometry.astrometrynet.SolveFieldCommand.version"></span>
    <span id="classes"></span>
    <span id="external-command-interface"></span>
    <span id="functions"></span>
    <span id="module-astropop.astrometry"></span>
