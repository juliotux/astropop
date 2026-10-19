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

The ``dtype`` argument accepts integer and floating-point types. It controls
pixel working arrays, reduction accumulators, uncertainty calculations and
output arrays. It defaults to ``numpy.float64``. Using ``numpy.float32`` halves
the storage of these floating-point arrays, at the cost of precision::

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

With ``dtype='int16'``, inputs are converted before combining, and both nominal
and uncertainty calculations use integer arithmetic. For example, the inputs
1.9 and 4.9 become 1 and 4, giving an integer mean of 2::

    >>> images = [FrameData([[value]], uncertainty=error)
    ...           for value, error in [(1.9, 3.9), (4.9, 4.9)]]
    >>> combined = imcombine(images, method='mean', dtype='int16')
    >>> combined.data.tolist(), combined.uncertainty.tolist()
    ([[2]], [[1]])
    >>> combined.dtype.name, combined.uncertainty.dtype.name
    ('int16', 'int16')

Integer division truncates towards zero; square roots truncate downwards.
Overflow wraps in the chosen dtype, including sums, squared errors and the
addition used to average the middle pair of an even-sample median. For example,
two int16 pixels of 30000 sum to -5536. Fractional clipping thresholds, the
Gaussian MAD scale factor, and sum normalization factors are also truncated.
Thus a normalization factor of 3/2 becomes 1, and integer uncertainty estimates
can differ substantially from floating-point estimates. A negative variance
after overflow raises ``ValueError``. Choose a wider or floating-point dtype
when these effects are undesirable.

Integer inputs and uncertainties must be finite. Rejection uses a separate
Boolean mask because integers cannot store NaN. Fully rejected output pixels
and uncertainties contain zero; those pixels are masked unless flag storage
is disabled. Counts and selection indices use NumPy's index dtype, and the
image count must fit the selected integer dtype for use in divisions. Custom
sigma-clipping callbacks receive masked integer data and are responsible for
respecting the requested dtype within their own calculations.

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
