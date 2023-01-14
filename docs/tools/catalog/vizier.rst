.. include:: ../../references.txt

.. |vizier| replace:: `~astropop.catalogs.vizier`
.. |VizierSourcesCatalog| replace:: `~astropop.catalogs.vizier.VizierSourcesCatalog`

Vizier Catalogs
===============

The VizieR is an astronomical catalog service that provides standard astronomical catalogs and tables in a uniform format. The VizieR is a service of the Centre de Données astronomiques de Strasbourg (CDS) and is operated by the Strasbourg Astronomical Data Center (CDS) and is available at http://vizier.u-strasbg.fr.

Several VizieR catalogs are useful for astrometric and photometric calibration. However, they do not allways have the same column names. So, the |vizier| module provides a set of classes to access the VizieR catalogs and to return the data in a |SourcesCatalog| instance.

To access an specific pre-defined catalog, just use ``vizier.<<catalog_name>>`` with the query parameters. Like, if you want to query the UCAC4 catalog for the sources in a 2 arcmin radius around the coordinates of the Crab Nebula, you can do:

.. ipython:: python
    :okwarning:

    from astropop.catalogs import vizier
    catalog = vizier.ucac4('M1', radius='2 arcmin', band=['B', 'V'])
    catalog.table()
    catalog.magnitude('B')
    catalog.sources_id()
    catalog.skycoord()

All catalogs have the same input parameters, that are a field ``center``, a search ``radius`` and the list of photometric ``band`` to query. They are derived from the |VizierSourcesCatalog| class, configured with a pre-defined configuration file to operate with the specific catalog. The list of all available pre-defined catalogs can be obtained with `astropop.catalogs.vizier.list_vizier_catalogs` method. Each catalog has a ``help`` method that returns a description of the catalog and the list of available photometric bands.

Currently available catalogs are:

- ``vizier.allwise``

  .. ipython:: python

      print(vizier.allwise.help())

- ``vizier.apass9``

  .. ipython:: python

      print(vizier.apass9.help())

- ``vizier.denis``

  .. ipython:: python

      print(vizier.denis.help())

- ``vizier.gsc242``

  .. ipython:: python

      print(vizier.gsc242.help())

- ``vizier.hip``

  .. ipython:: python

      print(vizier.hip.help())

- ``vizier.twomass``

  .. ipython:: python

      print(vizier.twomass.help())

- ``vizier.tycho2``

  .. ipython:: python

      print(vizier.tycho2.help())

- ``vizier.ucac4``

  .. ipython:: python

      print(vizier.ucac4.help())

- ``vizier.ucac5``

  .. ipython:: python

      print(vizier.ucac5.help())

- ``vizier.vsx``

  .. ipython:: python

      print(vizier.vsx.help())

- ``vizier.wise``

  .. ipython:: python

      print(vizier.wise.help())


Configure a New Catalog
-----------------------

To configure a new catalog, you can use a `YAML <https://yaml.org/>`_ configure file, to be passed to the |VizierSourcesCatalog| class. See the documentation for details. The file must have the following structure:

.. code-block:: yaml

    ---
    # caralog table in vizier
    table: 'I/322A/out'
    # small description for the catalog
    description: "UCAC4 Catalogue (Zacharias+, 2012)"
    # bibcode string reference
    bibcode: "2012yCat.1322....0Z"
    # list of columns to be downloaded in the query. Use '**' for all columns
    # Photometric columns will be added later based on selected
    # photometric bands.
    columns: ['+_r', 'UCAC4', 'RAJ2000', 'DEJ2000', 'pmRA', 'pmDE', 'EpRA']

    # dictionary of available photometric bands and their descriptions
    available_filters: {
      'J': '2MASS J magnitude (1.2um)',
      'H': '2MASS H magnitude (1.6um)',
      'K': '2MASS Ks magnitude (2.2um)',
      'B': 'B magnitude from APASS',
      'V': 'V magnitude from APASS',
      'g': 'g magnitude from APASS',
      'r': 'r magnitude from APASS',
      'i': 'i magnitude from APASS'
    }

    # coordinates configuration
    coordinates:
      ra_column: 'RAJ2000'  # RA column name
      dec_column: 'DEJ2000'  # DEC column name
      pm_ra_column: 'pmRA'  # proper motion RA column name (optional)
      pm_dec_column: 'pmDE'  # proper motion DEC column name (optional)
      frame: 'icrs'  # coordinate frame, default is 'icrs'

    # photometric configuration. Use {band} where the band name should be
    magnitudes:
      mag_column: '{band}mag'  # magnitude column name
      err_mag_column: 'e_{band}mag'  # magnitude error column name

    # epoch configuration. You can choose either column or value
    # but not both
    epoch:
      column: 'EpRA'  # epoch of coordinates column name
      # value = 'J2000.0'  # fixed epoch of coordinates value
      format: 'jyear'  # epoch format, default is 'jyear'

    # identifier configuration
    ids:
      prepend: 'UCAC4'  # prepend string to the identifier
      column: 'UCAC4'  # identifier column name.
      # multiple columns can be used, as a list. In this case, set also a separator
      # column: ['UCAC4', 'UCAC4_2']
      # separator: '-'  # separator between columns

.. automodapi:: astropop.catalogs.vizier
    :no-inheritance-diagram:
    :inherited-members:
