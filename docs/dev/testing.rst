Development and validation
==========================

Install an editable checkout with the ``test`` and ``docs`` extras as described
in :doc:`../installation`. Run tests from the repository root so the checkout,
not another installed copy, provides the source under test.

Offline tests and live catalogs
-------------------------------

Ordinary tests must not depend on live astronomical services::

    python -m pytest tests docs --remote-data=none

The catalog suite shares test cases between recorded responses and live
conformance. Recorded responses are stored by catalog under
``tests/data/catalog_responses``. Missing recordings must be fixed explicitly;
they are not permission to access the network during an offline run.

For intentional live conformance, run the dedicated tox environment::

    tox -e py314-catalogs-online

Live tests depend on service availability and current schemas. Do not treat a
recorded-response pass as evidence that a remote service is working. To refresh
recordings, follow ``tests/data/catalog_responses/README.md`` and review the
changed response files before committing them.

Focused tests and numerical changes
-----------------------------------

Select existing modules or individual test classes while developing, for example::

    python -m pytest tests/test_image_imarith.py tests/test_image_imcombine.py

Check units, dtype, flags, absent versus zero uncertainty, in-place semantics,
input nonmutation, and all-rejected pixels when those contracts are affected.
Restore global configuration changes after each test. Test temporary storage
with a dedicated temporary directory. To inspect interpreter-exit behavior,
use a child process rather than executing all registered exit handlers inside
pytest, which would also affect coverage collection.

Documentation checks
--------------------

Runnable reStructuredText examples are part of pytest collection. Run changed
guides and build the HTML documentation::

    python -m pytest docs --remote-data=none
    python -m sphinx -b html -E -a -n -W --keep-going docs docs/_build/html
    python docs/development/check_docs.py --html docs/_build/html
    python docs/development/check_docs.py --notebooks

The ``-a -n`` options check references while rewriting every page, including
unchanged guides. Pandoc must be on ``PATH`` to render the stored notebooks.
Sphinx builds may also retrieve external reference inventories. Distinguish unreachable services
from local rendering or reference errors, and inspect the complete build log.
Notebook examples should not introduce implicit live catalog requests during
builds. Generated API pages and build output are ignored by Git.

``tox -e build_docs`` runs these checks on Python 3.14, including offline
prompted examples, stored-notebook schema/syntax checks and retained HTML URLs.
CI uses the same tox environment. Read the Docs selects Python 3.14 and fails
on Sphinx warnings; it renders stored notebooks without executing their external
workflows. See :doc:`../development/documentation` for example classification
and hosted preview review.
