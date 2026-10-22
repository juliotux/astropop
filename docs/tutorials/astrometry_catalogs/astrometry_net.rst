Solve an observed field with Astrometry.net
===========================================

Start with a calibrated image containing isolated, unsaturated stars. Solve the
field, inspect the returned WCS and save the calibrated image. It uses a local Astrometry.net
installation and index files, rather than an online solver or a fit to already
identified counterparts.

The external calls below are recipes for your own observations. They are not
executed by documentation builds. For a complete executable offline example
with supplied observations and local counterparts, use
:doc:`astrometric_calibration`.

Prepare the image and solver
----------------------------

Install ``solve-field`` version 0.95 or newer, and indexes covering your field
and angular scale, as described in :doc:`../../modules/astrometry/index`. A working
Astropop installation alone does not install these resources. The example
assumes a field near RA 132.5 degrees, Dec 11.83 degrees and a scale around
0.5 arcsec per pixel. Replace all three estimates and the paths for your camera.

.. code-block:: python

    from pathlib import Path
    from astropop.framedata import read_framedata
    from astropop.astrometry import AstrometrySolver

    image_path = Path('reduced/science.fits')
    index_directory = Path('astrometry-indexes').resolve()
    frame = read_framedata(image_path)
    height, width = frame.shape

    print(f'Image: {width} columns by {height} rows; unit: {frame.unit}')

    options = {
        'ra': 132.5,
        'dec': 11.83,
        'radius': 1.,
        'scale-low': 0.4,
        'scale-high': 0.6,
        'scale-units': 'arcsecperpix',
        'index-dir': str(index_directory),
        'crpix-center': None,
    }
    solver = AstrometrySolver()

``'ra'`` and ``'dec'`` specify the approximate center in degrees; ``'radius'``
limits its search in degrees. ``'scale-low'`` and ``'scale-high'`` bound the
pixel scale in the specified ``'scale-units'``. ``'index-dir'`` adds a directory
to the index search. ``'crpix-center'`` places the reference pixel at the image
center; its presence selects the option, so its value is ``None``.
Use option names with hyphens, not configuration-file directives such as
``add_path``. An incorrect narrow position or scale constraint can exclude the
actual solution.

Solve and inspect the result
----------------------------

.. code-block:: python

    solution = solver.solve_field(image_path, options=options,
                                  stdout_loglevel='INFO')
    center = solution.wcs.pixel_to_world((width - 1) / 2, (height - 1) / 2)

    print(f'Field center: {center.ra.deg:.6f}, {center.dec.deg:.6f} deg')
    print(f'Solver correspondences: {len(solution.correspondences)}')
    print(solution.header)

The printed center and correspondence count depend on the input image and
indexes; there is no universal expected count. Inspect correspondence residuals
across the image and compare with independent reference stars. A returned WCS
does not by itself establish your required positional accuracy. Begin without
high-order distortion fitting; increase model complexity only when the field
coverage and residuals justify it.

Inspect the coordinate grid on the actual pixels:

.. code-block:: python

    import matplotlib.pyplot as plt
    from astropy.visualization import simple_norm

    fig = plt.figure(figsize=(7, 6), layout='constrained')
    ax = fig.add_subplot(projection=solution.wcs)
    ax.imshow(frame.data, origin='lower', cmap='gray',
              norm=simple_norm(frame.data, 'asinh', percent=99.5))
    ax.coords.grid(color='white', alpha=0.5)
    ax.set(xlabel='Right ascension', ylabel='Declination')
    plt.show()

A grid overlay helps detect gross orientation or pointing errors. Use the
independent residual diagnostic in :doc:`astrometric_calibration` to assess
accuracy; a visually plausible grid is insufficient.

Attach the WCS and save
-----------------------

.. code-block:: python

    # Save the fitted coordinates alongside the original calibrated pixels.
    frame.wcs = solution.wcs
    output = image_path.with_name(image_path.stem + '_wcs.fits')
    frame.write(output, overwrite=True)

    print(f'Saved image and WCS: {output}')

Assign the solution to ``frame.wcs`` before writing: solving returns an
``AstrometricSolution`` and does not attach it to your original frame. The
example overwrites its named derived output when rerun, leaving the input file
as supplied. Reopen the output to inspect its stored coordinate system.

Choose a different input interface
----------------------------------

To solve an already loaded frame instead of a filename, replace the solver call
with this alternative:

.. code-block:: python

    from astropop.astrometry import solve_astrometry_framedata

    solution = solve_astrometry_framedata(frame, options=options,
                                          stdout_loglevel='INFO')

If you already detected stars, pass their positions and relative brightnesses
instead of asking the solver to detect them again:

.. code-block:: python

    from astropop.astrometry import solve_astrometry_xy
    from astropop.photometry import background, starfind

    sky, rms = background(frame.data, mask=frame.mask)
    sources = starfind(frame.data, background=sky, noise=rms,
                       threshold=5, mask=frame.mask)

    print(f'Detected source-list candidates: {len(sources)}')

    # This solver expects FITS pixel centers, so add one to the detected coordinates.
    solution = solve_astrometry_xy(
        sources['x'] + 1, sources['y'] + 1, sources['flux'],
        width=width, height=height, image_header=frame.header,
        options=options, stdout_loglevel='INFO')

Astropop source detections are zero-based ``(x, y)`` coordinates. The source-list
wrapper writes the supplied coordinates directly into the solver's FITS table;
the ``+1`` converts these detections to FITS one-based coordinates. Image shape
is ``(height, width)``, including for nonsquare images. Do not swap these values
or add another offset when the source list is already one-based. See
:doc:`../../getting_started/conventions` for WCS origins.

All three interfaces return the same solution container. Choose one input
path for a given reduction, then attach, validate and save its WCS as above.
If solving fails, inspect the solver log, available indexes, source quality
and position/scale constraints before changing the scientific model.
Use the :ref:`solver failure-handling recipe <astrometry-failure-handling>`
to distinguish an unsolved field from a failed process and retain diagnostics.
