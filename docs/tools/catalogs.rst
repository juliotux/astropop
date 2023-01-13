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

    >>> from astropop.catalogs import SourcesCatalog
    >>> cat = SourcesCatalog(ids=['a', 'b', 'c'], ra=[1, 2, 3], dec=[4, 5, 6], unit='deg')
    >>> print(cat.table())
     id  ra dec
        deg deg
    --- --- ---
      a 1.0 4.0
      b 2.0 5.0
      c 3.0 6.0

For the additional magnitudes, a dictionary must be passed, with one list of magnitudes for each filter. The dictionary keys are the filter names, and the values are the corresponding lists of magnitudes. The example below shows how to create a catalog with three filters, ``'g'``, ``'r'``, and ``'i'``. These lists can be plain values for magnitudes without errors, or |QFloat| objects for magnitudes with errors. If errors are present, they will be extracted and stored in the ``*_error`` columns. If no errors are present, the ``*_error`` columns will be filled with zeros.

    >>> cat = SourcesCatalog(ids=['a', 'b', 'c'], ra=[1, 2, 3], dec=[4, 5, 6], unit='deg',
    ...                      mag={'g': [1, 2, 3], 'r': [4, 5, 6], 'i': [7, 8, 9]})
    >>> print(cat.table())
     id  ra dec  g  g_error  r  r_error  i  i_error
        deg deg
    --- --- --- --- ------- --- ------- --- -------
      a 1.0 4.0 1.0     0.0 4.0     0.0 7.0     0.0
      b 2.0 5.0 2.0     0.0 5.0     0.0 8.0     0.0
      c 3.0 6.0 3.0     0.0 6.0     0.0 9.0     0.0

    >>> from astropop.math import QFloat
    >>> cat = SourcesCatalog(ids=['a', 'b', 'c'], ra=[1, 2, 3], dec=[4, 5, 6], unit='deg',
    ...                      mag={'g': QFloat([1, 2, 3], [0.1, 0.2, 0.3]),
    ...                           'r': QFloat([4, 5, 6], [0.4, 0.5, 0.6]),
    ...                           'i': QFloat([7, 8, 9], [0.7, 0.8, 0.9])})
    >>> print(cat.table())
     id  ra dec  g  g_error  r  r_error  i  i_error
        deg deg
    --- --- --- --- ------- --- ------- --- -------
      a 1.0 4.0 1.0     0.1 4.0     0.4 7.0     0.7
      b 2.0 5.0 2.0     0.2 5.0     0.5 8.0     0.8
      c 3.0 6.0 3.0     0.3 6.0     0.6 9.0     0.9

An optional table with additional informations can be passed as ``query_table`` argument. This table will be accessed only by ``query_table`` property and will also be filtered for cross-matching. However, it cannot be used for automatic extraction the default |SourcesCatalog| data.

    >>> from astropy.table import Table
    >>> cat = SourcesCatalog(ids=['a', 'b', 'c'], ra=[1, 2, 3], dec=[4, 5, 6], unit='deg',
    ...                      query_table=Table({'id': ['a', 'b', 'c'], 'z': [1, 2, 3]}))
    >>> print(cat.table())
     id  ra dec
        deg deg
    --- --- ---
      a 1.0 4.0
      b 2.0 5.0
      c 3.0 6.0
    >>> print(cat.query_table)
     id  z
    --- --
      a  1
      b  2
      c  3


Properties Accessing
--------------------

The |SourcesCatalog| class provides several functions to get the data stored in the catalog. No setter function is present, to protect the catalog internally. So |SourcesCatalog| are inteded to be read-only objects.

Coordinates
^^^^^^^^^^^

Coordinates can be accessed using the `~astropop.catalogs.SourcesCatalog.skycoord`, `~astropop.catalogs.SourcesCatalog.ra_dec_list` and `~astropop.catalogs.SourcesCatalog.get_coordinates` functions. The difference is that in the first a |SkyCoord| object is returned, while in the second a list of tuples is returned. The last function is useful to get coordinates in |SkyCoord| format with space motion applied.

    >>> cat = SourcesCatalog(ids=['a', 'b', 'c'], ra=[1, 2, 3], dec=[4, 5, 6], unit='deg')
    >>> print(cat.skycoord())
    <SkyCoord (ICRS): (ra, dec) in deg
        [(1., 4.), (2., 5.), (3., 6.)]>
    >>> print(cat.ra_dec_list())
    [(1.0, 4.0), (2.0, 5.0), (3.0, 6.0)]
    >>> print(cat.get_coordinates())
    <SkyCoord (ICRS): (ra, dec) in deg
        [(1., 4.), (2., 5.), (3., 6.)]>

Soures IDs
^^^^^^^^^^

Sources IDs can be accessed using the `~astropop.catalogs.SourcesCatalog.sources_id` property.

    >>> cat = SourcesCatalog(ids=['a', 'b', 'c'], ra=[1, 2, 3], dec=[4, 5, 6], unit='deg')
    >>> print(cat.sources_id())
    ['a' 'b' 'c']

Magnitudes
^^^^^^^^^^

Magnitudes can be either accessed using `~astropop.catalogs.SourcesCatalog.magnitude` or `~astropop.catalogs.SourcesCatalog.mag_list` functions. For both of them, the ``band`` argument must be passed to access the magnitudes in a given filter. The difference is that the first function returns a |QFloat| object, while the second returns a list of tuples containing (magnitude, error). If the ``band`` argument is not passed, an error is raised. Also, the filter must be available in the catalog.

    >>> cat = SourcesCatalog(ids=['a', 'b', 'c'], ra=[1, 2, 3], dec=[4, 5, 6], unit='deg',
    ...                      mag={'g': [1, 2, 3], 'r': [4, 5, 6], 'i': [7, 8, 9]})
    >>> print(cat.magnitude('g'))
    1.0 +/- 0.0
    >>> print(cat.mag_list('g'))
    [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]

Summary Table
^^^^^^^^^^^^^

A summary table of the catalog can by obtained via `~astropop.catalogs.SourcesCatalog.table` function. This table contains the sources IDs, coordinates, proper motion (if available) and magnitudes (if available). No additional information is included. See the examples present in :ref:`manual_catalogs` section.

Object Matching
---------------

To identify a list of sources in a |SourcesCatalog|, you can use the `~astropop.catalogs.SourcesCatalog.match_objects` method. In this method, lists of ``ra`` and ``dec`` coordinates are passed and the catalog will find the closest sources within a ``limit_angle`` radius around them. A new |SourcesCatalog| instance is returned, containing only the matched sources in the exact order of the ``ra`` and ``dec`` lists passed to the method. To handle high proper motion, ``obstime`` argument can also be passed, but only is applied when proper motion is available in the catalog.

    >>> cat = SourcesCatalog(ids=['a', 'b', 'c'], ra=[1, 2, 3], dec=[4, 5, 6], unit='deg')
    >>> matched = cat.match_objects([1, 3], [4, 6], limit_angle='1 arcmin')
    >>> print(matched.table())
    id  ra dec
        deg deg
    --- --- ---
      a 1.0 4.0
      c 3.0 6.0

Magnitudes and additional ``query_table`` informations are also filtered and returned in the exact same order of the ``ra`` and ``dec`` lists passed to the method.

    >>> cat = SourcesCatalog(ids=['a', 'b', 'c'], ra=[1, 2, 3], dec=[4, 5, 6], unit='deg',
    ...                      mag={'g': [1, 2, 3], 'r': [4, 5, 6], 'i': [7, 8, 9]},
    ...                      query_table=Table({'id': ['a', 'b', 'c'], 'z': [1, 2, 3]}))
    >>> matched = cat.match_objects([1, 3], [4, 6], limit_angle='1 arcmin')
    >>> print(matched.table())
     id  ra dec  g  g_error  r  r_error  i  i_error
        deg deg
    --- --- --- --- ------- --- ------- --- -------
      a 1.0 4.0 1.0     0.0 4.0     0.0 7.0     0.0
      c 3.0 6.0 3.0     0.0 6.0     0.0 9.0     0.0
    >>> print(matched.query_table)
     id  z
    --- ---
      a   1
      c   3

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
