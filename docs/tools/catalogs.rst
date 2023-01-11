.. include:: ../references.txt

Catalogs Data
=============

|astropop| includes a simple interface to handle sources catalogs. It is mainly designed to perform astrometric RA and Dec coordinates and photometric magnitudes access and cross-matching between catalogs. It can be both offline or use online catalogs data based on the |astroquery| package.

This tools are not designed to be as complete as |astroquery|, but instead they are intended to provide a simple, direct and unified interface to interact with (RA, Dec) catalogs of sources. So, all the functionalities are limited by design.

The main class is |SourcesCatalog| which is a wrapper around |SkyCoord|, adding to it relevant informations of sources id (or names) and magnitudes. As it is *not* inherited from |SkyCoord|, it is not possible to use all |SkyCoord| methods directly on a |SourcesCatalog| object.

Manual (Offline) Catalog
------------------------

|SourcesCatalog| are constructing by passing, mainly, with the same arguments as |SkyCoord|, but also with a list of sources names and a dictionary of magnitudes. What the code does is to extract ``ids``, ``mags`` and ``query_table`` arguments and pass the rest to |SkyCoord|. But, is always recomended to use ``ra`` and ``dec`` compatible arguments, descried in the |SourcesCatalog| api to avoid problems in data accessing and cross-matching.

.. TODO:: continue the documentation.

Online Catalogs
---------------

Online catalogs queries are possible using the |astroquery| package. For a more coherent experience, |astropop| provides a wrapper around |astroquery| catalogs queries, which are available in the `~astropop.catalogs` module. As different catalogs have different formats, specialized classes for different sources are provides. Consult the following descriptions for more details:

.. toctree::
   :maxdepth: 1

   catalog/simbad.rst
   catalog/vizier.rst
   catalog/gaia.rst

.. automodapi:: astropop.catalogs
    :no-inheritance-diagram:




