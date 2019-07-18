Notes About Pixel Convention
============================

Numpy/Python
------------

See https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html


Matplotlib
----------


Astropy's WCS/Photutils/Cotout2D/etc
------------------------------------

.. todo:: Check this informations carefully 

``Astropy.wcs.WCS`` use the standard FITS-WCS pixel convention, as described
better in Astropy WCS Docs [#]_. It uses a 0-based index counter, where the
coordinate ``(0, 0)`` is the *center* of the first *bottom-left* pixel. This means
that the fisrt pixel goes from ``-0.5`` to ``+0.5``, and not ``0.0`` to ``1.0``
like Python/Numpy do.

However, FITS store the bottom-left pixel first in the file and, when Astropy
(and other FITS loaders) read the image, it automatically flips the image vertically.
This means that, for plotting FITS images, Matplotlib need to be set with ``origin='lower'``.

For a coordinate propose, however, no flip need to be done. ``(0, 0)`` will correspond
correctly to the center of first array element.

Photutils uses this convention too. So, a source identified in ``(x.xx, y.yy)``
coordinate by photutils, will have its coordinates determined corectly by the WCS.

Cotout2D class also seems to match this convention.


SEP/SExtractor
--------------

SEP uses the same SExtractor convention, that matches the standard pixel convention.
The convention is ``(x, y) = (0, 0)`` to the *center* of the first *bottom-left* pixel [#]_.


Astropop
--------


References
----------

.. [#] https://docs.astropy.org/en/stable/wcs/wcsapi.html#pixel-conventions-and-definitions

.. [#] https://sep.readthedocs.io/en/v1.0.x/api/sep.sum_circle.html#sep.sum_circle