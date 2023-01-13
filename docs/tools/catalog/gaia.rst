.. include:: ../../references.txt

.. |GaiaDR3SourcesCatalog| replace:: `~astropop.catalogs.gaia.GaiaDR3SourcesCatalog`
.. |gaiadr3| replace:: `~astropop.catalogs.gaia.gaiadr3`

Gaia Catalog
============

The European space mission Gaia aims to measure the positions, distances, motions, and brightnesses of over 1 billion stars within the Milky Way galaxy, as well as provide data for a significant number of extragalactic and Solar system objects. This information, along with data on multiplicity, variability, and other astrophysical parameters, is stored in the Gaia Archive.

|astroquery| provides a query interface to Gaia, using `~astroquery.gaia` module. But, as we do with another sources, |GaiaDR3SourcesCatalog| can directly query the Gaia Archive and return a |SourcesCatalog| object. This can be done using the |gaiadr3| or |GaiaDR3SourcesCatalog|.

.. ipython:: python

    from astropop.catalogs import gaia
    gaia_sources = gaia.gaiadr3('Sirius', radius='20 arcsec')
    gaia_sources.table()

.. ipython:: python

    from astropop.catalogs.gaia import GaiaDR3SourcesCatalog
    gaia_sources = GaiaDR3SourcesCatalog('Sirius', radius='20 arcsec')
    gaia_sources.table()

Currently, only the Data Release 3 (DR3) is supported. The |GaiaDR3SourcesCatalog| class is a subclass of |SourcesCatalog|, so all the methods of |SourcesCatalog| are available.

All the 3 Gaia filters are available for photometry: ``G``, ``BP`` and ``RP``. As Gaia catalog do not provide a direct magnitude error, we compute it from ``phot_*_mean_flux_over_error`` columns, using the approximation:

.. math::

    \sigma_{mag} \approx \frac{1.1}{SNR_{flux}}


.. automodapi:: astropop.catalogs.gaia
    :no-inheritance-diagram:
    :inherited-members:
