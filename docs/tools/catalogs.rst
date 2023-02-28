.. include:: ../references.txt

Catalogs Data
=============

|astropop| offers an easy-to-use interface for working with sources catalogs, primarily for handling astrometric RA and Dec coordinates and photometric magnitudes, as well as cross-matching between catalogs. It can either work with offline data or access online catalogs through the |astroquery| package. However, it is not as comprehensive as |astroquery| and is intended to provide a straightforward and consistent way to interact with (RA, Dec) catalogs of sources, with limited capabilities by design.

The main class for this work is |SourcesCatalog|, which serves as a wrapper for |SkyCoord|, adding information such as source IDs and magnitudes. However, it is not derived from |SkyCoord|, so not all |SkyCoord| methods can be used on a |SourcesCatalog| object.

.. _manual_catalogs:

Manual (Offline) Catalog
------------------------

The |SourcesCatalog| class is created by providing almost the same arguments as |SkyCoord|, along with a list of source names and a dictionary of magnitudes. The code extracts the ``ids``, ``mag``, and ``query_table`` arguments and passes the rest to |SkyCoord|. However, it's recommended to use ``ra`` and ``dec`` compatible arguments as described in the |SourcesCatalog| API to avoid issues with data access and cross-matching.

If you want to create a basic astrometric catalog without photometric data, simply pass a list of object ``ids`` and the appropriate astrometric information, as shown in the example provided.

.. ipython:: python

    from astropop.catalogs import SourcesCatalog
    cat = SourcesCatalog(ids=['a', 'b', 'c'], ra=[1, 2, 3],
                         dec=[4, 5, 6], unit='deg')
    cat.table()

For the additional magnitudes, a dictionary must be passed, with one list of magnitudes for each filter. The dictionary keys are the filter names, and the values are the corresponding lists of magnitudes. The example below shows how to create a catalog with three filters, ``'g'``, ``'r'``, and ``'i'``. These lists can be plain values for magnitudes without errors, or |QFloat| objects for magnitudes with errors. If errors are present, they will be extracted and stored in the ``*_error`` columns. If no errors are present, the ``*_error`` columns will be filled with zeros.

.. ipython:: python

    cat = SourcesCatalog(ids=['a', 'b', 'c'], ra=[1, 2, 3],
                         dec=[4, 5, 6], unit='deg',
                         mag={'g': [1, 2, 3],
                              'r': [4, 5, 6],
                              'i': [7, 8, 9]})
    cat.table()

.. ipython:: python

    from astropop.math import QFloat
    cat = SourcesCatalog(ids=['a', 'b', 'c'], ra=[1, 2, 3],
                         dec=[4, 5, 6], unit='deg',
                         mag={'g': QFloat([1, 2, 3], [0.1, 0.2, 0.3]),
                              'r': QFloat([4, 5, 6], [0.4, 0.5, 0.6]),
                              'i': QFloat([7, 8, 9], [0.7, 0.8, 0.9])})
    cat.table()

An optional table with additional informations can be passed as ``query_table`` argument. This table will be accessed only by ``query_table`` property and will also be filtered for cross-matching. However, it cannot be used for automatic extraction the default |SourcesCatalog| data.

.. ipython:: python

    from astropy.table import Table
    cat = SourcesCatalog(ids=['a', 'b', 'c'], ra=[1, 2, 3],
                         dec=[4, 5, 6], unit='deg',
                         query_table=Table({'id': ['a', 'b', 'c'],
                                            'z': [1, 2, 3]}))
    cat.table()
    cat.query_table


Properties Accessing
--------------------

The |SourcesCatalog| class provides several functions to get the data stored in the catalog. No setter function is present, to protect the catalog internally. So |SourcesCatalog| are inteded to be read-only objects.

Coordinates
^^^^^^^^^^^

Coordinates can be accessed using the `~astropop.catalogs.SourcesCatalog.skycoord`, `~astropop.catalogs.SourcesCatalog.ra_dec_list` and `~astropop.catalogs.SourcesCatalog.get_coordinates` functions. The difference is that in the first a |SkyCoord| object is returned, while in the second a list of tuples is returned. The last function is useful to get coordinates in |SkyCoord| format with space motion applied.

.. ipython:: python

    cat = SourcesCatalog(ids=['a', 'b', 'c'], ra=[1, 2, 3],
                         dec=[4, 5, 6], unit='deg')
    cat.skycoord()
    cat.ra_dec_list()
    cat.get_coordinates()

You can also access separated arrays for RA and Dec coordinates using the `~astropop.catalogs.SourcesCatalog.ra` and `~astropop.catalogs.SourcesCatalog.dec` properties.

.. ipython:: python

    cat.ra()
    cat.dec()

Soures IDs
^^^^^^^^^^

Sources IDs can be accessed using the `~astropop.catalogs.SourcesCatalog.sources_id` property.

.. ipython:: python

    cat = SourcesCatalog(ids=['a', 'b', 'c'], ra=[1, 2, 3],
                         dec=[4, 5, 6], unit='deg')
    cat.sources_id()

Magnitudes
^^^^^^^^^^

Magnitudes can be either accessed using `~astropop.catalogs.SourcesCatalog.magnitude` or `~astropop.catalogs.SourcesCatalog.mag_list` functions. For both of them, the ``band`` argument must be passed to access the magnitudes in a given filter. The difference is that the first function returns a |QFloat| object, while the second returns a list of tuples containing (magnitude, error). If the ``band`` argument is not passed, an error is raised. Also, the filter must be available in the catalog.

.. ipython:: python
    :okwarning:

    cat = SourcesCatalog(ids=['a', 'b', 'c'], ra=[1, 2, 3],
                         dec=[4, 5, 6], unit='deg',
                         mag={'g': [1, 2, 3],
                              'r': [4, 5, 6],
                              'i': [7, 8, 9]})
    cat.magnitude('g')
    cat.mag_list('g')

Summary Table
^^^^^^^^^^^^^

A summary table of the catalog can by obtained via `~astropop.catalogs.SourcesCatalog.table` function. This table contains the sources IDs, coordinates, proper motion (if available) and magnitudes (if available). No additional information is included. See the examples present in :ref:`manual_catalogs` section.

Object Matching
---------------

To identify a list of sources in a |SourcesCatalog|, you can use the `~astropop.catalogs.SourcesCatalog.match_objects` method. In this method, lists of ``ra`` and ``dec`` coordinates are passed and the catalog will find the closest sources within a ``limit_angle`` radius around them. A new |SourcesCatalog| instance is returned, containing only the matched sources in the exact order of the ``ra`` and ``dec`` lists passed to the method. To handle high proper motion, ``obstime`` argument can also be passed, but only is applied when proper motion is available in the catalog.

.. ipython:: python

    cat = SourcesCatalog(ids=['a', 'b', 'c'], ra=[1, 2, 3],
                         dec=[4, 5, 6], unit='deg')
    matched = cat.match_objects([1, 3], [4, 6], limit_angle='1 arcmin')
    matched.table()

Magnitudes and additional ``query_table`` informations are also filtered and returned in the exact same order of the ``ra`` and ``dec`` lists passed to the method.

.. ipython:: python

    cat = SourcesCatalog(ids=['a', 'b', 'c'], ra=[1, 2, 3],
                         dec=[4, 5, 6], unit='deg',
                         mag={'g': [1, 2, 3],
                              'r': [4, 5, 6],
                              'i': [7, 8, 9]},
                         query_table=Table({'id': ['a', 'b', 'c'],
                                            'z': [1, 2, 3]}))
    matched = cat.match_objects([1, 3], [4, 6], limit_angle='1 arcmin')
    matched.table()
    matched.query_table

Online Catalogs
---------------

The |astroquery| package allows for querying online catalogs, but the |astropop| package offers a more user-friendly interface through the `~astropop.catalogs` module. As the queries from |astroquery| is not exactly homogeneous and some manual work must be done some times, we offer specialized classes for pre-defined catalogs. See the following descriptions for more information.

.. toctree::
   :maxdepth: 1

   catalog/simbad.rst
   catalog/vizier.rst
   catalog/gaia.rst

.. automodapi:: astropop.catalogs
    :no-inheritance-diagram:
