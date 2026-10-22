.. include:: ../../references.txt

Catalogs, coordinates and magnitudes
====================================

|SourcesCatalog| stores source IDs, equatorial coordinates and optional
magnitudes. It wraps |SkyCoord| rather than subclassing it. Local and online
catalogs share this interface; :doc:`selection` covers service choice,
query failures and reuse, while this page explains the stored data and matching.

.. _manual_catalogs:

Manual (Offline) Catalog
------------------------

The |SourcesCatalog| constructor accepts coordinate arguments understood by
|SkyCoord|, together with source identifiers and optional photometry. It extracts
``ids``, ``mag`` and ``query_table`` and forwards the remaining arguments to
SkyCoord. Supply equatorial coordinates and explicit angular units for the
RA/Dec accessors and matching methods used below. One identifier must correspond
to each coordinate.

Create a catalog without photometry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For an astrometric source list, supply IDs and coordinates without ``mag``.
The summary table then contains only those identifiers and coordinates; it
does not invent photometric measurements::

    >>> import numpy as np
    >>> from astropy.table import Table
    >>> from astropop.catalogs import SourcesCatalog
    >>> from astropop.math import QFloat

    >>> astrometric = SourcesCatalog(ids=['a', 'b', 'c'], ra=[1, 2, 3],
    ...                             dec=[4, 5, 6], unit='deg')

    >>> print(astrometric.table())
     id  ra dec
        deg deg
    --- --- ---
      a 1.0 4.0
      b 2.0 5.0
      c 3.0 6.0

Supply magnitudes in several bands
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pass a ``mag`` dictionary with filter names as keys and one magnitude per
source as each value. The names are used to select photometry later; the
constructor does not infer a filter system from them. Plain lists provide
values without uncertainties. The summary includes an ``<band>_error`` column
for each band, filled with zeros in this case::

    >>> photometric = SourcesCatalog(ids=['a', 'b', 'c'], ra=[1, 2, 3],
    ...                              dec=[4, 5, 6], unit='deg',
    ...                              mag={'g': [12, 13, 14],
    ...                                   'r': [11, 12, 13],
    ...                                   'i': [10, 11, 12]})

    >>> print(photometric.table()['id', 'g', 'g_error', 'r', 'i'])
     id  g   g_error  r    i
    --- ---- ------- ---- ----
      a 12.0     0.0 11.0 10.0
      b 13.0     0.0 12.0 11.0
      c 14.0     0.0 13.0 12.0

Zero stored errors here mean that uncertainties were not supplied; they do not
establish exact measurements or justify infinite statistical weights.

Supply magnitudes with uncertainties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use a |QFloat| array for each dictionary value to provide magnitudes and their
standard uncertainties together. Source ordering must agree across every band.
The constructor separates each array into its magnitude and error columns::

    >>> uncertain = SourcesCatalog(ids=['a', 'b', 'c'], ra=[1, 2, 3],
    ...                            dec=[4, 5, 6], unit='deg',
    ...                            mag={'g': QFloat([12, 13, 14], [0.1, 0.2, 0.3]),
    ...                                 'r': QFloat([11, 12, 13], [0.2, 0.3, 0.4]),
    ...                                 'i': QFloat([10, 11, 12], [0.3, 0.4, 0.5])})

    >>> print(uncertain.table()['id', 'g', 'g_error', 'r', 'r_error', 'i', 'i_error'])
     id  g   g_error  r   r_error  i   i_error
    --- ---- ------- ---- ------- ---- -------
      a 12.0     0.1 11.0     0.2 10.0     0.3
      b 13.0     0.2 12.0     0.3 11.0     0.4
      c 14.0     0.3 13.0     0.4 12.0     0.5

``magnitude('g')`` reconstructs a QFloat containing that band's values and
uncertainties. See :doc:`../quantities/index` for working with QFloat;
the accessors below demonstrate the catalog's photometric interface.

Attach auxiliary query information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An optional ``query_table`` stores additional columns in the same source order.
It does not supply coordinates or magnitudes automatically. For example, keep
a quality code beside a catalog with one measured band::

    >>> auxiliary = Table({'quality': [0, 1, 0]})
    >>> cat = SourcesCatalog(ids=['a', 'b', 'c'], ra=[1, 2, 3],
    ...                      dec=[4, 5, 6], unit='deg',
    ...                      mag={'g': QFloat([12, 13, 14], [0.1, 0.2, 0.3])},
    ...                      query_table=auxiliary)

    >>> print('Sources:', len(cat))
    Sources: 3
    >>> print('Bands:', cat.filters)
    Bands: ['g']

``ids`` must be one-dimensional, and coordinate and magnitude lengths must
agree. The constructor does not validate the length of ``query_table``:
the caller must ensure its length and ordering match.

Read coordinates, identifiers and magnitudes
--------------------------------------------

Coordinates
^^^^^^^^^^^

``skycoord()`` returns the stored coordinate model as a SkyCoord;
``get_coordinates(obstime)`` attempts space-motion propagation to that epoch.
``ra()``, ``dec()`` and ``ra_dec_list()`` return numerical degrees, with the last
arranged as a NumPy array of RA/Dec pairs::

    >>> print('RA (deg):', cat.ra().tolist())
    RA (deg): [1.0, 2.0, 3.0]
    >>> print('RA/Dec pairs:', cat.ra_dec_list().tolist())
    RA/Dec pairs: [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

Source IDs
^^^^^^^^^^

``sources_id()`` returns the identifier array::

    >>> print('IDs:', cat.sources_id().tolist())
    IDs: ['a', 'b', 'c']

Magnitudes
^^^^^^^^^^

``magnitude(band)`` returns a |QFloat| in magnitudes. ``mag_list(band)`` returns
a NumPy array of magnitude/error pairs. Supply an available band explicitly;
absent photometry or an unknown band raises ``ValueError``::

    >>> print('g values:', cat.magnitude('g').nominal.tolist())
    g values: [12.0, 13.0, 14.0]
    >>> print('g errors:', cat.magnitude('g').uncertainty.tolist())
    g errors: [0.1, 0.2, 0.3]
    >>> print('g pairs:', cat.mag_list('g').tolist())
    g pairs: [[12.0, 0.1], [13.0, 0.2], [14.0, 0.3]]

Summary Table
^^^^^^^^^^^^^

``table()`` returns IDs, coordinates, available proper motions and magnitudes.
It excludes auxiliary query columns and does not serialize the complete motion
model, including its reference epoch, distance and radial velocity::

    >>> print('Summary columns:', cat.table().colnames)
    Summary columns: ['id', 'ra', 'dec', 'g', 'g_error']
    >>> print('Auxiliary columns:', cat.query_colnames)
    Auxiliary columns: ['quality']

Row selection, query columns and copying
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An integer, slice, index list or Boolean array selects rows into a new catalog.
Even an integer returns a one-row catalog. A string instead selects a column
from the auxiliary query table, not from ``table()``::

    >>> print('First row IDs:', cat[0].sources_id().tolist())
    First row IDs: ['a']
    >>> print('Query quality:', cat['quality'].tolist())
    Query quality: [0, 1, 0]
    >>> print('Selected IDs:', cat[np.array([True, False, True])].sources_id().tolist())
    Selected IDs: ['a', 'c']

    >>> exported = cat.table()

    >>> # The exported table is a copy; editing it leaves the catalog unchanged.
    >>> exported['id'][0] = 'z'

    >>> print('Original IDs:', cat.sources_id().tolist())
    Original IDs: ['a', 'b', 'c']

Without a query table, string indexing raises ``KeyError``. Access magnitudes
with ``magnitude('g')``, not ``cat['g']`` unless the query itself has a ``g``
column. ``table()`` and ``query_table`` return separate tables, but the object
is not an immutable store: ``sources_id()`` exposes its underlying array,
``copy()`` is shallow, and the constructor retains the supplied auxiliary
table. Copy IDs before editing them; use ``copy.deepcopy(cat)`` when independent
internal storage is required. Do not rely on shallow copies to isolate edits.

Empty catalogs and invalid coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An empty local catalog is valid storage, but not a nearest-neighbor reference::

    >>> empty = SourcesCatalog(ids=[], ra=[], dec=[], unit='deg')

    >>> print('Empty source count:', len(empty))
    Empty source count: 0
    >>> print('Empty summary rows:', len(empty.table()))
    Empty summary rows: 0

Check that a reference has rows before calling ``match_objects``. Handle an
empty detection list before invoking matching as well. Nonfinite coordinates
cannot be used by the nearest-neighbor search; filter or report them before
matching, keeping the original row indices so results can be reassembled.
Astropy also rejects invalid declinations and incompatible coordinate shapes.
Do not treat those input errors as successful unmatched observations.

Object Matching
---------------

`~astropop.catalogs.SourcesCatalog.match_objects` finds the nearest catalog
source independently for each input position, accepting it when its angular
separation is at most ``limit_angle``. Supply coordinate arrays in decimal
degrees and an explicit angular unit, for example ``'1 arcsec'``, for the limit.
The returned catalog has the **same length and order as the input positions**,
including unmatched rows. Multiple input positions can select the same catalog
source; this is not a one-to-one assignment.

Unmatched rows have an empty source ID, NaN coordinates and NaN magnitudes and
magnitude errors. Auxiliary ``query_table`` fields are reset to their dtype's
zero value (for example, numeric zero or an empty string), rather than NaN.
Use the source-ID success mask to select valid rows; a zero auxiliary value
alone does not identify a miss. The original catalog is unchanged::

    >>> import numpy as np
    >>> from astropy.table import Table
    >>> from astropop.catalogs import SourcesCatalog

    >>> cat = SourcesCatalog(ids=['A'], ra=[10], dec=[20], unit='deg',
    ...                      mag={'g': [12]},
    ...                      query_table=Table({'label': ['A'], 'count': [9]}))
    >>> matched = cat.match_objects([10, 180, 10], [20, -40, 20],
    ...                             limit_angle='1 arcsec')

    >>> print('len(matched):', len(matched))
    len(matched): 3
    >>> print('matched.sources_id().tolist():', matched.sources_id().tolist())
    matched.sources_id().tolist(): ['A', '', 'A']
    >>> print('Unmatched coordinates:', matched.ra_dec_list()[1])
    Unmatched coordinates: [nan nan]
    >>> print('Unmatched magnitude:', matched.magnitude('g').nominal[1])
    Unmatched magnitude: nan
    >>> print("matched.query_table['count'].tolist():", matched.query_table['count'].tolist())
    matched.query_table['count'].tolist(): [9, 0, 9]
    >>> print("matched.query_table['label'].tolist():", matched.query_table['label'].tolist())
    matched.query_table['label'].tolist(): ['A', '', 'A']

    >>> # Unmatched inputs retain a row with an empty identifier; select matches explicitly.
    >>> successful = matched.sources_id() != ''

    >>> print('len(matched[successful]):', len(matched[successful]))
    len(matched[successful]): 2
    >>> print('cat.sources_id().tolist():', cat.sources_id().tolist())
    cat.sources_id().tolist(): ['A']

``obstime`` selects the epoch used for nearest-neighbor searching when catalog
proper motion and a reference epoch are available. Supply input coordinates at
that same target epoch. It does not infer missing proper motions. This local
example creates a source moving approximately one arcsecond east per year::

    >>> from astropy import units as u
    >>> from astropy.time import Time

    >>> moving = SourcesCatalog(ids=['fast'], ra=[10] * u.deg, dec=[20] * u.deg,
    ...                         pm_ra_cosdec=[1000] * u.mas / u.yr,
    ...                         pm_dec=[0] * u.mas / u.yr,
    ...                         distance=[100] * u.pc,
    ...                         radial_velocity=[0] * u.km / u.s,
    ...                         obstime=Time('J2000'))
    >>> target_epoch = Time('J2010')

    >>> # Propagate the reference position to the observation epoch before matching.
    >>> target = moving.get_coordinates(target_epoch)
    >>> at_epoch = moving.match_objects(target.ra.deg, target.dec.deg,
    ...                                 '0.1 arcsec', obstime=target_epoch)

    >>> print('at_epoch.sources_id().tolist():', at_epoch.sources_id().tolist())
    at_epoch.sources_id().tolist(): ['fast']

    >>> at_original = moving.match_objects(target.ra.deg, target.dec.deg,
    ...                                    '0.1 arcsec')

    >>> print('at_original.sources_id().tolist():', at_original.sources_id().tolist())
    at_original.sources_id().tolist(): ['']

Matching selects rows; it is not a coordinate-correction operation. Recompute
coordinates explicitly with ``get_coordinates(target_epoch)`` when inspecting
propagated positions, rather than assuming the returned match coordinates are
the input positions.

.. _catalog-motion-snapshot:

Preserve a motion model for local reuse
---------------------------------------

Save the SkyCoord as an ECSV mixin column when the reference epoch and full
motion model are needed later. This retains the coordinate frame, time,
distance and velocity metadata of the example instead of reducing it to RA/Dec
columns. Reconstruct the catalog explicitly on reload::

    >>> from astropy.table import QTable
    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory

    >>> with TemporaryDirectory() as directory:
    ...     path = Path(directory) / 'moving_reference.ecsv'
    ...     snapshot = QTable({'id': moving.sources_id(), 'coords': moving.skycoord()})
    ...     snapshot.meta['description'] = 'Synthetic motion example, not an observed catalog'
    ...     snapshot.write(path, format='ascii.ecsv')
    ...     saved = QTable.read(path, format='ascii.ecsv')
    ...     restored = SourcesCatalog(saved['coords'], ids=saved['id'])
    ...     predicted = restored.get_coordinates(target_epoch)
    ...     print('Reference epoch:', restored.skycoord().obstime.jyear)
    ...     print(f'Reload difference: {predicted.separation(target).mas[0]:.3f} mas')
    Reference epoch: 2000.0
    Reload difference: 0.000 mas

Add magnitudes and auxiliary columns separately if they are needed, and record
catalog release, query region and retrieval provenance. The zero difference
above checks this serialization example; it is not a measure of physical
proper-motion accuracy.

An epoch argument cannot supply a missing motion model. For coordinates without
velocities, ``get_coordinates`` returns a copy of the original positions when
Astropy cannot propagate them; it does not relabel their epoch::

    >>> fixed = SourcesCatalog(ids=['fixed'], ra=[10], dec=[20], unit='deg')
    >>> unchanged = fixed.get_coordinates(target_epoch)

    >>> print('Position (deg):', unchanged.ra.deg.tolist(), unchanged.dec.deg.tolist())
    Position (deg): [10.0] [20.0]
    >>> print('Reference epoch:', unchanged.obstime)
    Reference epoch: None

Check that proper motion and a reference epoch are present before treating an
epoch-dependent match as propagated. The method catches Astropy ``ValueError``
and returns unchanged coordinates, so a returned object alone is not proof
that space motion was applied.

Online Catalogs
---------------

Service-specific adapters normalize their results into the same interface.
Use the following guides for configured bands and query options, and
:doc:`selection` for choosing and reusing queries.

Reference/API
-------------

.. automodapi:: astropop.catalogs
    :no-inheritance-diagram:


.. currentmodule:: None

Related topics
--------------

.. toctree::
    :maxdepth: 1

    selection
    simbad
    gaia
    vizier
