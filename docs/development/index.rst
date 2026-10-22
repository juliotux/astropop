Development
===========

These pages describe project validation and internal extension points, rather
than instrument-specific reduction recipes.

Start a contribution
--------------------

1. Create an editable checkout with test and documentation extras using
   :doc:`../getting_started/installation`.
2. Reproduce the issue with a small local input and run the affected test file::

       python -m pytest tests/test_image_imcombine.py --remote-data=none

3. Check scientific contracts: units, coordinate origin, uncertainties, masks,
   copying and dtype behavior. Preserve the public API unless the change
   explicitly introduces a documented migration.
4. Add a focused regression test for a behavioral fix. Run the relevant offline
   suite and the signature check::

       python scripts/check_type_annotations.py
       python -m pytest tests docs --remote-data=none

5. Update the affected guide or API docstring and follow :doc:`documentation`
   for strict rendering, notebook and canonical-page checks.
6. Review the diff and open a PR explaining the problem, resulting behavior,
   validation and remaining limitations. Keep independent changes in separate
   commits with an area prefix such as ``image:`` or ``docs:``.

A proposal or local plan is not a release note. Describe confirmed public changes
in :doc:`../project/migration` with their actual version boundary; keep speculative design
notes separate from user-facing promises.

Contributor guides
------------------

.. toctree::
    :maxdepth: 1

    testing
    internal_interfaces
    testing_helpers
    documentation
    typing
    benchmarking
