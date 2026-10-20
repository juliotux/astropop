Maintain and review documentation
=================================

Reader paths and stable URLs
----------------------------

The home page separates tutorials, task guides, shared concepts and API
reference. Existing module guides remain supported entry points. Arithmetic
and combination have separate task pages; their old combined URL is a gateway.
API declarations live in ``reference/`` exactly once, while generated ``api/``
URLs retain their names. Do not copy generated API files into version control.

``docs/development/urls.json`` records the original 30 authored page URLs and
fragment identifiers at the pre-reorganization baseline. Retained guides keep
their content; extracted reference sections leave non-indexed compatibility
anchors. Keep those anchors while external users may link to them. A future
URL removal needs an explicit hosting redirect policy and verification.

After building, verify the retained URLs and anchors::

    python docs/development/check_docs.py --html docs/_build/html

This checks local built HTML. It does not prove that a hosting service redirects
removed URLs, or that an external website still resolves.

Example execution policy
-------------------------

Generate the per-example inventory from the current sources rather than
maintaining a second list of line numbers::

    python docs/development/check_docs.py --inventory

The inventory identifies pytest examples, stored console transcripts, plot
execution, static recipes, and stored notebooks. A function
call appearing in a code block does not mean the build executes it. Literal
``::`` blocks without prompts are application recipes; adapt their filenames
and input variables before running them. Core tutorials state their dependencies
and expected results and use local temporary data.

* Run prompted examples with ``python -m pytest docs --remote-data=none``.
  This does not execute stored console transcripts or arbitrary code blocks.
* Use standard ``code-block:: python`` directives for recipes and
  ``code-block:: text`` for recorded console output. These blocks are not
  executed during the build. Run plot directives by building Sphinx, using
  seeded random generators.
* Choose figures that answer a workflow question: which samples were rejected,
  which sources matched, or where alignment residuals remain. Use computed
  results, labeled axes and units, alternative text, and captions explaining
  the interpretation. For shared plotting code, use an external-file directive
  such as ``plot:: _ext/guide_figures.py plot_registration``. The
  ``guide_figures`` extension records that helper as a source dependency so
  edits to it invalidate the corresponding pages.
* Tutorial RST pages are the maintained numerical source. Hidden ``doctest``
  blocks only prepare temporary data and clean it up; visible examples print
  meaningful results for doctests to verify. Keep assertions and testing
  helpers in the test suite, rather than inside learning workflows.
* ``tutorial-data`` generates the shared offline archive in the build directory.
  It contains raw observations, local standards, reference products and nine
  notebook companions. ``tutorial-notebook:: page_name`` links the corresponding
  generated notebook. Edit the RST, not a second notebook copy.
* ``tutorial_notebooks.py`` converts prompted examples, expected console output,
  prose and plot directives. Setup cells locate the extracted bundle and set
  optional-array configuration before loading frames. Notebook figures use the
  current cell results. Sphinx figure wrappers replay the same visible RST
  calculations in temporary directories rather than duplicating the reduction.
* Run ``python docs/development/run_tutorials.py --output /tmp/tutorial-review``
  from the repository root with the docs/test dependencies installed. It executes
  all nine notebooks with fresh kernels, both sequentially using preceding
  products and independently using the supplied references. Executed notebooks
  and figures remain under that dedicated output directory. ``--mode sequence``
  or ``--mode independent`` selects one run for focused debugging.
* These helpers import Sphinx/Docutils only inside extension setup code.
  Ordinary test collection ignores ``_ext`` and ``_build`` so it does not need
  documentation dependencies or collect copies from generated bundles. Keep
  seeded data creation offline and bounded; input truth is for final recovery
  diagnostics, never a substitute for estimating a result from measurements.
* Catalog help is rendered from bundled YAML configuration using
  ``catalog-help``; it does not construct a catalog or query a remote service.
* Stored notebooks are rendered with ``nbsphinx_execute='never'``. The CCD
  notebook needs its external dataset, the astrometry notebook needs local
  solver/index installation and input files, and the online-catalog notebook
  needs available remote services. Schema validation and compilation do not
  establish execution success. Refresh one deliberately in a clean kernel,
  inspect every output and error, record inputs/environment, and review the diff.
* Large-image timing/memory studies belong in the studies repository. They are
  benchmarks, not stable correctness assertions or routine documentation builds.

API and editorial checklist
-----------------------------

Use NumPy-style docstrings for public callables. Annotations define parameter
and return types; omit redundant type declarations from annotated docstring
fields and let Sphinx render the types in signatures. Keep names and behavioral
explanations, including units, shapes and estimator limitations. State defaults, literal string
choices and what each choice does, shape/dtype/unit requirements, mutation and
array ownership, missing arrays, masks, return types, and expected failures.
Distinguish methods (called with parentheses) from properties. Use “standard
deviation” for stored uncertainty; distinguish it from variance and scatter.
Explain zero-based ``(x, y)`` separately from NumPy ``[y, x]`` indexing.

A tutorial teaches a complete local workflow; a task guide solves a specific
problem; a concept explains assumptions; a reference specifies exact contracts.
Link between them, preserving critical scientific limitations beside operations.
Do not claim speed, memory bounds or scientific validation without a specified
measurement or test. Follow :doc:`../dev/testing` for strict build commands and
ordinary versus live-service tests.

Strict checks and pull-request previews
---------------------------------------

``tox -e build_docs`` uses Python 3.14 and installs both ``docs`` and ``test``
extras. It checks stored-notebook schema/code syntax, runs local prompted
examples with remote data disabled, builds every page with strict reference
and warning checks, then verifies retained URLs and anchors. The GitHub docs
job selects that environment. Read the Docs also selects Python 3.14 and
``sphinx.fail_on_warning: true``; ``nitpicky=True`` in ``docs/conf.py`` checks
object references on both hosts. Hosted builds do not execute the external
notebook workflows. A rendering pass alone is not a scientific test.

Review changes by opening ``docs/_build/html/index.html`` after a local build.
For hosted previews, follow the Read the Docs status check on the pull request
and verify the displayed commit before reviewing the new task/reference paths.
Do not assume a preview exists just because the YAML file is present.

The Read the Docs project administrator enables previews in **Settings → Pull
request builds → Build pull requests for this project → Update**. Existing
pull requests need another pushed commit to trigger their first preview after
enablement. That project setting is not controlled by ``.readthedocs.yml`` and
its enabled state has not been verified in this checkout. See the
`official preview configuration guide
<https://docs.readthedocs.com/platform/stable/guides/pull-requests.html>`_
and `build configuration reference
<https://docs.readthedocs.com/platform/stable/config-file/v2.html>`_.

If strict builds fail, distinguish an unreachable external inventory from a
missing local target and inspect the full log. Fix broken local references or
examples; do not globally suppress warnings. Keep live catalog conformance in
its dedicated test job. Rendered notebook pages must not silently replace a
missing recorded response with a live request.

Annotation coverage
-------------------

Run ``python scripts/check_type_annotations.py`` to check package signatures.
This checks annotation presence, including complete PEP 484 comments for
Astropy model ``evaluate`` methods, whose inline annotations have unit semantics.
It excludes generated version metadata and does not establish static type
correctness. The test and documentation tox environments run this check.
