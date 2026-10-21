Pixel and coordinate conventions
================================

Arrays and source positions
---------------------------

NumPy images have shape ``(n_rows, n_columns)`` and are indexed ``data[y, x]``.
Rows correspond to y; columns correspond to x. Astropop source detections,
aperture positions, and Astropy high-level WCS pixel coordinates use ``(x, y)``
ordering. These interfaces start both axes at zero at the center of the first
pixel; the FITS-oriented helpers described below are exceptions. Its boundaries are at
``-0.5`` and ``+0.5``; an integer array index selects that pixel's value, not a
geometric corner::

    >>> import numpy as np
    >>> image = np.zeros((3, 5))
    >>> x, y = 4, 2
    >>> image[y, x] = 1
    >>> print('image.shape:', image.shape)
    image.shape: (3, 5)
    >>> print('image[2, 4]:', image[2, 4])
    image[2, 4]: 1.0

``np.indices`` returns ``(y_grid, x_grid)``. Astropop's ``iraf_indices`` helper
instead returns ``(x_grid, y_grid)``, despite using the same zero-based centers.
Python slices remain stop-exclusive, so ``data[1:3, 2:5]`` selects two rows and
three columns.

Display orientation
-------------------

Matplotlib's ``imshow`` defaults to an upper origin, displaying row zero at
the top. Use ``origin='lower'`` to display increasing y upward. This changes
only the display; it does not flip or reorder image data. Astropy's FITS
reader does not automatically flip an image vertically. Do not insert an extra
array flip merely because a file is FITS.

WCS origins
-----------

Astropy's high-level ``pixel_to_world`` and ``world_to_pixel`` interfaces use
zero-based ``(x, y)`` pixel centers. The lower-level ``all_pix2world`` and
``all_world2pix`` methods require an explicit origin: ``0`` means zero-based
coordinates, while ``1`` means FITS one-based coordinates. FITS header
``CRPIX`` values use the one-based convention::

    >>> from astropy.wcs import WCS
    >>> wcs = WCS(naxis=2)
    >>> wcs.wcs.crpix = [1, 1]
    >>> wcs.wcs.crval = [0, 0]
    >>> wcs.wcs.cdelt = [1, 1]
    >>> zero_based = wcs.all_pix2world([[0, 0]], 0)
    >>> fits_based = wcs.all_pix2world([[1, 1]], 1)
    >>> print('zero_based:', np.round(zero_based, 6))
    zero_based: [[0. 0.]]
    >>> print('fits_based:', np.round(fits_based, 6))
    fits_based: [[0. 0.]]

Photutils aperture positions also use zero-based ``(x, y)`` centers. Convert
one-based external source lists explicitly; do not infer their origin from
a column's name alone.

Astropop helpers taking FITS coordinates
---------------------------------------

:func:`~astropop.astrometry.manual_wcs.wcs_from_coords` copies its ``x, y``
arguments directly into one-based FITS ``CRPIX``. For the nonsquare image above,
the zero-based pixel ``(4, 2)`` must therefore be supplied as ``(5, 3)``::

    >>> from astropop.astrometry.manual_wcs import wcs_from_coords
    >>> approximate = wcs_from_coords(5, 3, 120., -30., 0.34, north='top')
    >>> print('Reference pixel (FITS):', approximate.wcs.crpix.astype(int))
    Reference pixel (FITS): [5 3]
    >>> sky = approximate.pixel_to_world(4, 2)
    >>> print(f'Reference sky position: {sky.ra.deg:.3f}, {sky.dec.deg:.3f} deg')
    Reference sky position: 120.000, -30.000 deg

The named ``north`` directions correspond to angles measured counterclockwise
from positive x: ``'right'`` is 0 degrees, ``'top'`` is 90, ``'left'`` is 180,
and ``'bottom'`` is 270. These describe an image displayed with ``origin='lower'``
before any requested axis flip. This helper supplies an approximate geometric
WCS, not a fitted distortion calibration.

:func:`~astropop.astrometry.astrometrynet.solve_astrometry_xy` also expects
one-based ``x, y`` in the source list passed to astrometry.net. Add one to
Astropop detection positions when using that wrapper; pass the image width as
``image.shape[1]`` and height as ``image.shape[0]``. See
:doc:`../tutorials/astrometry_net` for the external-solver recipe.

Registration windows and cropping
---------------------------------

Registration windows use ``(row_slice, column_slice)`` ordering. Estimated
transforms must be expressed in full-image coordinates before applying them
to a full image or catalog; the registration implementation handles the window
origin. See :doc:`../reduction/registering` for the transform and shift sign
conventions. A crop changes the coordinate origin, so crop coordinates and WCS
together rather than applying a full-image WCS to unadjusted cutout positions.
