Tutorials
=========

Follow a complete example with fixed local inputs. CCD reduction and photometry
share a downloadable synthetic FITS sequence; generation is hidden from the
walkthroughs and checked with the examples. Each tutorial states what it executes
and which results to expect.

Start with :doc:`../installation` and the small example in
:doc:`../reduction/pipelines`. Follow :doc:`ccd_reduction` before
:doc:`photometry` to measure the image you reduced, or use the supplied
reference stack to begin with photometry. :doc:`local_catalog_matching` then
explains positional associations; it does not perform photometric calibration.
:doc:`polarimetry` starts from a separate set of ordinary/extraordinary beam
measurements rather than from the ordinary imaging stack.

.. toctree::
    :maxdepth: 1

    ccd_reduction
    photometry
    polarimetry
    local_catalog_matching
