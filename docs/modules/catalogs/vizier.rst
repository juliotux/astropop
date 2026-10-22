.. include:: ../../references.txt

.. |vizier| replace:: `~astropop.catalogs.vizier`
.. |VizierSourcesCatalog| replace:: `~astropop.catalogs.vizier.VizierSourcesCatalog`

Vizier Catalogs
===============

The VizieR adapters map configured survey columns to the shared
|SourcesCatalog| interface. Use the packaged configuration name to query a
region, then inspect the returned identifiers, coordinates and bands. This
recipe makes a live query; its result count depends on the service:

.. code-block:: python

    from astropop.catalogs import vizier

    catalog = vizier.ucac4('M1', radius='2 arcmin', band=['B', 'V'])

    print('Sources:', len(catalog))
    print('Bands:', catalog.filters)
    print(catalog.table()[:5])

Constructors accept ``center``, ``radius`` and ``band``. ``band='all'`` selects
all configured filters; ``band=None`` disables normalized photometry; a list
selects named filters. Column requests are assembled from the configuration,
including its available magnitude columns, so selecting fewer bands does not
guarantee a smaller remote response. Use :doc:`selection` for
service selection, query provenance and failure handling.

Inspect packaged descriptions without a query:

.. code-block:: python

    print(vizier.list_vizier_catalogs())

    # Read the packaged catalog description without querying the service.
    print(vizier.ucac4.help())

The help below is generated from the same bundled YAML files. It describes the
installed configuration, not a live verification of the remote schema.

Currently available catalogs are:

- ``vizier.allwise``

  .. catalog-help:: allwise

- ``vizier.apass9``

  .. catalog-help:: apass9

- ``vizier.denis``

  .. catalog-help:: denis

- ``vizier.gsc242``

  .. catalog-help:: gsc242

- ``vizier.hip``

  .. catalog-help:: hip

- ``vizier.skymapper``

  .. catalog-help:: skymapper

- ``vizier.twomass``

  .. catalog-help:: twomass

- ``vizier.tycho2``

  .. catalog-help:: tycho2

- ``vizier.ucac4``

  .. catalog-help:: ucac4

- ``vizier.ucac5``

  .. catalog-help:: ucac5

- ``vizier.vsx``

  .. catalog-help:: vsx

- ``vizier.wise``

  .. catalog-help:: wise


Configure a New Catalog
-----------------------

Save the following schema as ``ucac4_custom.yml``. It maps the VizieR table
identifier, requested columns, coordinates, magnitudes, epoch and source IDs:

.. code-block:: yaml

    ---
    # VizieR table identifier
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
      # value: 2000.0  # alternative fixed Julian year
      format: 'jyear'  # explicit Astropy Time format

    # identifier configuration
    ids:
      prepend: 'UCAC4'  # prepend string to the identifier
      column: 'UCAC4'  # identifier column name.
      # multiple columns can be used, as a list. In this case, set also a separator
      # column: ['UCAC4', 'UCAC4_2']
      # separator: '-'  # separator between columns

Instantiate the configured adapter with that file before the field arguments.
This also makes a live query:

.. code-block:: python

    from astropop.catalogs.vizier import VizierSourcesCatalog

    custom = VizierSourcesCatalog('ucac4_custom.yml', (83.633, 22.0145),
                                  radius='2 arcmin', band=['B', 'V'])

    print('Configuration:', custom.name)
    print('Sources:', len(custom))
    print(custom.table()[:5])

Coordinate columns must carry angular units in the response. Proper motion is
used only when both configured motion columns are present in the schema;
``pm_ra_column`` represents RA motion multiplied by cos(Dec). Preserve the
catalog's coordinate frame and reference epoch rather than inferring them from
an image date or the spelling of a column name.

Schema variants
^^^^^^^^^^^^^^^

* ``column_aliases`` maps original response column names to local names.
  For example, ``column_aliases: {'2MASS': '_2MASS'}`` requests the original
  name and renames it for extraction when the alias is not already present.
  Use the local alias in ``columns``, ``ids`` and other extraction fields.
* ``epoch.column`` may be one name or a list of alternatives. The reader uses
  the **first column present**; it does not average epochs or choose a fallback
  per row. If none is present, no epoch is attached. Check missing values before
  propagating positions.
* ``epoch.value`` supplies a fixed epoch instead of a column. Specify
  ``epoch.format`` explicitly: for example ``value: 2000.0`` with
  ``format: jyear``, or a numeric MJD column with ``format: mjd``. String Julian
  years such as ``J2000.0`` require ``jyear_str``. Use either ``value`` or
  ``column``; if both are supplied, ``value`` takes precedence.
* For an astrometry-only configuration, omit ``available_filters`` and
  ``magnitudes``. Query with ``band=None`` (``'all'`` also selects no filters).
  Asking for a named band in that configuration raises ``ValueError``.
* For photometric configurations, provide both ``mag_column`` and
  ``err_mag_column`` templates. A configured error column absent from the
  response yields zero stored errors; zero does not establish an exact
  measurement. Inspect the original response before using such values as weights.
* ``ids.column`` accepts a name or a list of names joined with ``separator``
  (default ``'-'``). ``prepend`` adds a prefix separated by a space. ``ids: null``
  produces empty IDs, which cannot serve as the usual nonempty-ID matching
  success indicator; choose a real identifier for matching workflows.

The adapter retains the first returned table and exposes it as ``query_table``.
An empty table list raises ``RuntimeError``. This differs from a returned table
with zero rows; inspect its length before indexing. The schema does not apply
survey-specific quality cuts. Test a custom configuration against a small field
and inspect column names, units, missing values and quality flags before reuse.




Reference/API
-------------

.. automodapi:: astropop.catalogs.vizier
    :skip: list_vizier_catalogs
    :no-inheritance-diagram:
    :inherited-members:

.. currentmodule:: astropop.catalogs.vizier

.. py:function:: list_vizier_catalogs()

    Return a formatted string listing preconfigured catalog names, descriptions,
    and reference bibcodes from the packaged YAML files. Print the returned
    string to inspect the installed choices. No online query is performed.


.. currentmodule:: None
