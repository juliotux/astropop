Diagnostics and data handling
=============================

These complete investigations follow inputs across several modules. The FITS
collection workflow creates temporary files and inspects its header index; the
quality workflow follows values and flags into saved products; the registration
workflow compares transforms and coordinate residuals across a simulated field.
They run offline and create their small synthetic inputs in the examples.

.. toctree::
    :maxdepth: 1

    fits_collections
    quality_propagation
    registration_diagnostics
