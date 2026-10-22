First steps with FrameData
==========================

Continue with :doc:`../tutorials/ccd_reduction/index` for a complete offline
FITS-to-photometry example after this first calculation.

Check which installation the interpreter imports, especially when using a
notebook kernel (the version depends on your installation)::

    >>> import astropop

    >>> print('Astropop version:', astropop.__version__)  # doctest: +ELLIPSIS
    Astropop version: ...

FrameData associates image values with units, optional standard uncertainties,
flags, and metadata. Use explicit reduction functions rather than Python
arithmetic operators on FrameData::

    >>> import numpy as np
    >>> from astropop.framedata import FrameData
    >>> from astropop.image.imarith import imarith

    >>> image = FrameData(np.full((4, 4), 100.), uncertainty=2., unit='adu')

    >>> # Multiplying by an exact scalar scales the standard uncertainty too.
    >>> result = imarith(image, 2, '*')

    >>> print('result.data:', np.round(result.data, 6))
    result.data: [[200. 200. 200. 200.]
     [200. 200. 200. 200.]
     [200. 200. 200. 200.]
     [200. 200. 200. 200.]]
    >>> print('result.uncertainty:', np.round(result.uncertainty, 6))
    result.uncertainty: [[4. 4. 4. 4.]
     [4. 4. 4. 4.]
     [4. 4. 4. 4.]
     [4. 4. 4. 4.]]

Use ``read_framedata('image.fits')`` for existing files. Set optional-array
configuration **before loading or creating frames** when memory savings are
required; see :ref:`optional-frame-arrays`. Continue with :doc:`../tutorials/index`
for complete workflows or :doc:`../modules/index` for module explanations and simple API examples.
