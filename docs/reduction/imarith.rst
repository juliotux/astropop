.. include:: ../references.txt


Image Arithmetics and Combining
===============================

Image Arithmetics (|imarith|)
-----------------------------

.. TODO:: Write imarith documentation


Image Combining (|imcombine|)
-----------------------------

Combining images is performed in astropop using the |imcombine| function. It is designed to combine |FrameData| using several methods and rejection algorithms.

The current combine methods are ``median``, ``mean`` and ``sum``.

The ``dtype`` argument selects the real numeric type used consistently for
pixel working arrays, combination arithmetic, uncertainty calculations and
output arrays. It accepts supported NumPy integer and floating-point types,
dtype objects and dtype strings, and defaults to ``numpy.float64``.

Choose ``dtype`` to control the memory required per pixel throughout the
combination. The same dtype applies to both nominal values and uncertainties::

    >>> import numpy as np
    >>> from astropop.framedata import FrameData
    >>> from astropop.image import imcombine
    >>> images = [FrameData(np.full((2, 2), value), unit='adu')
    ...           for value in (1, 3)]
    >>> combined = imcombine(images, method='mean', dtype='float32')
    >>> combined.dtype.name
    'float32'
    >>> combined.uncertainty.dtype.name
    'float32'

``memory_limit`` sizes the rejection and combination chunks. Input copies,
output arrays and additional temporary arrays are outside this budget, so it
does not cap total process memory. For large collections,
``use_disk_cache=True`` explicitly places internal image copies in
memory-mapped temporary files. It does not release caller-owned input frames.

The current rejection methods are:
- ``minmax_clip``
- ``sigma_clip``

Math Details
^^^^^^^^^^^^

.. TODO:: describe the error computing here

Sum Combine
~~~~~~~~~~~

Sigma Clipping
~~~~~~~~~~~~~~

To perform sigma clipping, it is recomended to use ``median`` as central tendency and ``mad_std`` as deviation estimation. These two estimators handle outliers better then the mean and standard deviation of the distribution, so produce much better results in sigma clipping. A detailed analysis can be found on the `CCD Data Reduction Guide <https://mwcraig.github.io/ccd-as-book/01-06-Image-combination.html#The-solution:-average-combine,-but-clip-the-extreme-values>`_.

API
---

.. automodapi:: astropop.image.imarith
    :no-inheritance-diagram:

.. automodapi:: astropop.image.imcombine
    :no-inheritance-diagram:
