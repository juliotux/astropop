API Reference
=============

The complete ASTROPOP API, divided by modules and tasks, is presented bellow.

General Tools
-------------

.. autosummary::
   :toctree: _api

   astropop.file_collection
   astropop.fits_utils
   astropop.logger
   astropop.py_utils

Astrometry Processing
---------------------
.. autosummary::
   :toctree: _api

   astropop.astrometry
   astropop.astrometry.astrometrynet
   astropop.astrometry.coods_utils
   astropop.astrometry.manual_wcs

Catalogs Handling
-----------------

.. autosummary::
   :toctree: _api

   astropop.catalogs
   astropop.catalogs.local
   astropop.catalogs.online
   astropop.catalogs.utils

FrameData Container
-------------------

.. autosummary::
   :toctree: _api

   astropop.framedata
   astropop.framedata.framedata
   astropop.framedata.memmap
   astropop.framedata.compat
   astropop.framedata.utils

Image Processing
----------------

.. autosummary::
   :toctree: _api

   astropop.image
   astropop.image.imarith
   astropop.image.processing
   astropop.image.register
   astropop.image.imcombine

Math Utils
----------

.. autosummary::
   :toctree: _api

   astropop.math
   astropop.math.array
   astropop.math.gaussian
   astropop.math.hasher
   astropop.math.moffat
   astropop.math.physical

Photometry Processing
---------------------

.. autosummary::
   :toctree: _api

   astropop.photometry
   astropop.photometry.aperture
   astropop.photometry.detection
   astropop.photometry.solve_photometry

Pipelines
---------

.. autosummary::
   :toctree: _api

   astropop.pipelines

Polarimetry Processing
----------------------

.. autosummary::
   :toctree: _api

   astropop.polarimetry
   astropop.polarimetry.dualbeam


Testing Helpers
---------------

.. autosummary::
   :toctree: _api

   astropop.testing
