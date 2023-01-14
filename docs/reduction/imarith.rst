.. include:: ../references.txt


Image Arithmetics and Combining
===============================

Image Arithmetics (|imarith|)
-----------------------------

.. TODO:: Write imarith documentation


Image Combining (|imcombine|)
-----------------------------

Combining images is performed in astropop using the |imcombine| function. It is designed to combine |FrameData| using several methods and rejection algorithms.

The current combine methods are:
- ``median``
- ``average``
- ``sum``

The current rejection methods are:
- ``minmax_clip``
- ``sigma_clip``

Math Details
````````````

.. TODO:: describe the error computing here

Sum Combine
'''''''''''

Sigma Clipping
''''''''''''''

To perform sigma clipping, it is recomended to use ``median`` as central tendency and ``mad_std`` as deviation estimation. These two estimators handle outliers better then the mean and standard deviation of the distribution, so produce much better results in sigma clipping. A detailed analysis can be found on the `CCD Data Reduction Guide <https://mwcraig.github.io/ccd-as-book/01-06-Image-combination.html#The-solution:-average-combine,-but-clip-the-extreme-values>`_.

API
---

.. automodapi:: astropop.image.imarith
    :no-inheritance-diagram:

.. automodapi:: astropop.image.imcombine
    :no-inheritance-diagram:
