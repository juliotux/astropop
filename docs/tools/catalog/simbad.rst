.. include:: ../../references.txt

.. |simbad| replace:: `~astropop.catalogs.simbad.simbad`
.. |simbad_query_id| replace:: `~astropop.catalogs.simbad.simbad_query_id`
.. |SimbadSourcesCatalog| replace:: `~astropop.catalogs.simbad.SimbadSourcesCatalog`

Simbad Database
===============

The Simbad database is a very commonly used database of astronomical objects. It is maintained by the CDS (Centre de Données astronomiques de Strasbourg) and is available at http://simbad.u-strasbg.fr/simbad/.

It provides objects identification across multiple catalogs, photometry for several filters and states of the art astrometry. This data comes from several sources, compiled by the CDS.

|SourcesCatalog| from Simbad
-----------------------------

To generate a |SourcesCatalog| from any Simbad query, use the |SimbadSourcesCatalog| class or simply by |simbad| interfaces. During the instance creation, instead of pass the data itself, you must pass a query ``center``, a search ``radius`` and (optionally) the list of photometric ``band`` you want to retrieve. The query will be performed and the data will be downloaded and stored in the instance and can be accessed just like any other |SourcesCatalog|.

.. ipython:: python

    from astropop.catalogs import simbad
    simbad_sources = simbad.simbad(center='Sirius', radius='1 arcmin',
                                   band=['g'])
    simbad_sources.table()

.. ipython:: python

    simbad_sources = simbad.SimbadSourcesCatalog(center='Sirius',
                                                 radius='1 arcmin',
                                                 band=['B', 'V'])
    simbad_sources.table()


The default behavior of the catalog is that whe no photometric band is passed, no photometric information will be retrieved. For a list of all available filters, see `~astropop.catalogs.simbad.SimbadSourcesCatalog.available_filters` property.

Also, additionally to the |SourcesCatalog| methods, |SimbadSourcesCatalog| have `~astropop.catalogs.simbad.SimbadSourcesCatalog.coordinates_bibcode` and `~astropop.catalogs.simbad.SimbadSourcesCatalog.magnitudes_bibcode` properties that returns the bibcode of the coordinates and photometry data, respectively.

.. ipython:: python

    simbad_sources.coordinates_bibcode()
    simbad_sources.magnitudes_bibcode('V')

The full original query table is accessed using `~astropop.catalogs.simbad.SimbadSourcesCatalog.query_table` property.

Identifying Sources in Simbad
-----------------------------

The `~astropop.catalogs.simbad` module provides an additional method for identify a source or a list of sources based on their RA and Dec coordinates. This is done using |simbad_query_id| method.

.. ipython:: python

    from astropop.catalogs.simbad import simbad_query_id
    simbad_query_id(101.287155, -16.716115833333333, limit_angle='1 arcsec')
    simbad_query_id(ra=[101.287155, 101.28876708333333],
                    dec=[-16.716115833333333, -16.716867777777775],
                    limit_angle='1 arcsec')

See that a ``limit_angle`` must be provided. This is the maximum distance between the coordinates and the object coordinates to be considered a match. If no match is found, an empty string name is returned.

The default behavior is to use the ``MAIN_ID`` column as name. But you can choose a priority order to get the name of the star. Example: if you want to get only HD, HYP and TYC names, in this priority order, you can do with ``name_order`` parameter:

.. ipython:: python

    simbad_query_id(101.287155, -16.716115833333333, limit_angle='1 arcsec',
                    name_order=['HD', 'HYP', 'TYC'])

.. warning::

    This function is vectorized using `~numpy.vectorize`. So, each object will get an individual server query. So, even if this method is able to query more than one object, it is very slow and not recomended for high number of sources.

.. automodapi:: astropop.catalogs.simbad
    :no-inheritance-diagram:
    :inherited-members:
