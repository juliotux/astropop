Choose and reuse catalog queries
================================

First decide whether you need names, astrometry/proper motion, or photometric
bands. The shared :class:`~astropop.catalogs.SourcesCatalog` interface contains
IDs, coordinates and optional magnitudes; it is not the full Astroquery API.
Learn its matching behavior with :doc:`../tutorials/local_catalog_matching`
before depending on a remote service.

.. list-table:: Query decisions
    :header-rows: 1
    :widths: 25 40 35

    * - Need
      - Starting point
      - Check before use
    * - Object identity and literature photometry
      - :doc:`../tools/catalog/simbad`
      - Object type, band availability, missing magnitudes, coordinate epoch.
    * - Gaia DR3 astrometry and available bands
      - :doc:`../tools/catalog/gaia`
      - Proper-motion epoch, row completeness and quality columns.
    * - A configured survey or custom VizieR table
      - :doc:`../tools/catalog/vizier`
      - Packaged table identifier, exact filter spelling and units.

Print :func:`~astropop.catalogs.vizier.list_vizier_catalogs` to inspect the catalog
names/descriptions shipped with your installed version. This reads packaged
configuration, not a live catalog. Each returned catalog exposes ``filters``;
use those names exactly. Missing errors or photometry do not establish a
high-quality measurement. Inspect finite magnitudes/errors and appropriate
survey quality columns before calibration.

Live queries and failures
--------------------------

Examples in the service guides require network access and are intentionally
static. Record the query region, release, filters, magnitude limits, epoch and
retrieval date. Reuse a reviewed ECSV/FITS snapshot for repeated local matching;
see the local tutorial. Preserve auxiliary columns separately when needed.

A service timeout differs from an empty scientific result. Catch the actual
exception from a failed request and retain its diagnostic details; do not
substitute an empty successful catalog. Empty-result behavior and exception
types can differ between services and dependency versions. Inspect the response
length and the service guide before indexing row zero. Verify table/column names
against the installed packaged configuration and service schema before changing
URLs or retrying indefinitely.

Gaia uses synchronous queries in this implementation to avoid the problematic
asynchronous job polling path. Synchronous transport still has timeouts and
service limits; it cannot guarantee availability. The Gaia guide documents
pagination and completeness checks. Restrict the query region/columns to the
scientific need, use bounded retries in the application, and keep query failures
visible. A successful recorded-response test checks our parser, not the remote
service's current health.

For high-proper-motion targets, propagate reference coordinates to the observation
epoch with available motion metadata before interpreting separations. Matching
is independently nearest-neighbor within ``limit_angle``; it preserves unmatched
rows and permits repeated reference sources. Neither astrometric proximity nor
a shared band name alone establishes a valid photometric zero-point reference.
