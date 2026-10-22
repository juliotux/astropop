.. include:: ../../references.txt

FITS file utilities
===================

Use :mod:`astropy.io.fits` to open and write FITS files, and
:func:`~astropop.framedata.read_framedata` when image data, units, masks, and
uncertainties should travel together. The small :mod:`astropop.fits_utils`
module provides shared FITS conventions used by those interfaces.

Identifying image HDUs
----------------------

``imhdus`` is a tuple of the Astropy image-HDU classes accepted by Astropop,
including primary, image-extension, compressed-image, and streaming HDUs. Use
it with ``isinstance``; it does not inspect whether the HDU contains data::

    >>> from astropy.io import fits
    >>> from astropop.fits_utils import imhdus

    >>> isinstance(fits.PrimaryHDU(), imhdus)
    True
    >>> isinstance(fits.BinTableHDU(), imhdus)
    False

Header-key normalization
------------------------

``string_to_header_key`` uppercases standard FITS keywords of at most eight
characters. For longer or nonstandard keywords, it preserves their spelling
and removes a leading ``HIERARCH`` prefix and its following space if supplied. Astropy's Header
interface handles that prefix when the key is written::

    >>> from astropop.fits_utils import string_to_header_key

    >>> string_to_header_key('exptime')
    'EXPTIME'
    >>> string_to_header_key('HIERARCH astropop example')
    'astropop example'

This helper normalizes a key string; it does not modify a Header or validate
its associated value.




Reference/API
-------------

.. automodapi:: astropop.fits_utils
    :no-inheritance-diagram:

.. autofunction:: astropop.fits_utils.string_to_header_key

.. currentmodule:: None

.. py:data:: astropop.fits_utils.imhdus

    Tuple of Astropy image-HDU classes suitable for ``isinstance`` checks,
    as illustrated in :doc:`fits_utils`. An HDU can belong to one of these classes and still
    contain no image data.


.. currentmodule:: None
