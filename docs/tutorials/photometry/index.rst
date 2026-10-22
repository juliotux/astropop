Photometric measurements
========================

Start with a calibrated image and its noise/mask information. Stacked-image
photometry produces a measurement table; magnitude calibration also requires
a solved WCS and local standards from :doc:`../astrometry_catalogs/index`.
The light curve uses a separate individual-exposure sequence with the same
calibration masters, not the stacked image. Each workflow saves measurements
and quality information. All three run offline with :doc:`../../getting_started/examples`.

.. toctree::
    :maxdepth: 1

    photometry
    photometric_calibration
    light_curve
