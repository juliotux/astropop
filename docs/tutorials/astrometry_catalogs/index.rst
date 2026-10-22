Astrometry and catalogs
=======================

Choose the route appropriate to the inputs. Local matching uses a supplied WCS;
local WCS fitting uses approximate pointing and identified references. Both
run offline on the shared data and save matches or a solved image. Astrometry.net
requires an installed solver and suitable indexes; online catalogs require
network access. Those external recipes are alternatives, not extra mandatory
steps after the local fit. See :doc:`../../getting_started/examples` for data setup.

.. toctree::
    :maxdepth: 1

    local_catalog_matching
    astrometric_calibration
    astrometry_net
    online_catalogs
