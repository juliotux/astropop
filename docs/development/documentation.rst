Maintain and review documentation
=================================

Reader paths and chapter ownership
----------------------------------

The home page has five destinations: Getting started, Tutorials, Modules and
reference, Development, and Project information. Each module chapter contains
its explanations, examples, technical background and Reference/API declarations.
Generated ``api/`` pages provide individual symbol contracts and retain their
URLs. Keep each indexed declaration in one owning chapter. Remove superseded
source pages and update their incoming links; do not leave orphan pages,
compatibility gateways or duplicate task/concept/reference trees.

Tutorials are organized beneath five real subsection indexes: CCD reduction,
photometry, astrometry/catalogs, polarimetry and diagnostics. Each index states
its inputs, outputs, dependencies and execution requirements. The half-wave
and quarter-wave tutorials are independent workflows and notebook companions.

After building into a clean output directory, verify the canonical pages::

    python docs/development/check_docs.py --html docs/_build/html

This requires built HTML for every authored page, rejects orphan source pages,
and checks annotation-derived parameter types. The strict Sphinx build checks
navigation and references. Remove the old build output after source moves so
stale HTML cannot mask missing pages. Hosting redirects for old external URLs
are separate from the maintained source tree.

Module explanations and complete tutorials
------------------------------------------

Module chapters are module-oriented. Each chapter should focus on a module or a
coherent API family, explaining its functions and classes with simple examples.
Cross-module, multi-step reductions belong in tutorials; technical concepts and
input/result diagnostics stay with the module that owns their contract.

Use :doc:`../modules/polarimetry/index` as the model: connect the scientific model
to the module's functions/classes, explain their options and results, and include
focused examples. Keep general observing/restart policy and complete cross-module
diagnostics in tutorials, and benchmarking instructions in contributor guides,
rather than filling module guides with unrelated advice.

Module guides must explain their functions and classes in clear, descriptive,
human-readable prose. Describe what an operation does, when it is useful, how
to supply its inputs and how to interpret the result. Explain important options,
units and scientific assumptions beside the corresponding operation. Generated
API declarations provide exact signatures within the same chapter; they
complement these explanations.

Every module guide must include simple examples of its functions and classes.
Introduce the purpose of each example and show meaningful output. Keep separate
examples for input forms that teach different behavior: for SourcesCatalog,
creating an astrometric catalog, supplying plain magnitudes in several bands,
and supplying QFloat magnitudes with uncertainties are all useful cases.
Related functions can share an example when each function's use remains clear.
When editing or reorganizing, preserve the explanations and behaviors those
examples teach. Removing repetition must not remove that instructional coverage.

Tutorials address more complex, multi-step tasks that combine APIs from several
modules. Take the reader from prepared inputs through processing and inspection to a usable
result. Module examples remain available for learning individual operations;
link them from the integrated workflow where useful.

Add technical details and references when they explain an algorithm, estimator,
convention or limitation. Prefer original articles, scientific literature and
technical descriptions from widely used software. Connect each citation to the
claim it supports, explain the relevant method in the page itself, and state
how Astropop's implementation follows or differs from it. A reference list
alone is not a description of the implemented API.

Example execution policy
-------------------------

Generate the per-example inventory from the current sources rather than
maintaining a second list of line numbers::

    python docs/development/check_docs.py --inventory

The inventory identifies pytest examples, stored console transcripts, plot
execution, static recipes, and generated notebook companions. A function
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
* Use inline ``.. mermaid::`` diagrams for workflows, dependencies and storage
  ownership. Let Mermaid lay out nodes and arrows; do not draw these schemes
  with Matplotlib. Split large diagrams into readable stages and use captions
  and alternative text. Mermaid renders SVG in the browser with light/dark
  themes and fullscreen viewing. Its versioned JavaScript loads from jsDelivr;
  viewing diagrams requires JavaScript and access to that CDN. A Sphinx build
  checks the directive, not Mermaid syntax or browser rendering: review the
  rendered diagrams in a browser after editing them.
* Choose scientific plots that answer a workflow question: which samples were rejected,
  which sources matched, or where alignment residuals remain. Use computed
  results, labeled axes and units, alternative text, and captions explaining
  the interpretation. Code these plots with ``.. plot::`` and explicitly set
  ``:include-source: false``. Keep plotting code inside the RST whenever
  possible. Ordinary imports of plotting libraries or the public API do not
  require a separate plotting script.
* Use an external Python script only when the plot needs functions defined in
  another file, such as the shared workflow/figure helpers. For that case,
  ``plot:: _ext/workflow_figures.py plot_polarimetry`` replays the tutorial
  and calls the same drawing function used by its notebook. The extension
  tracks those shared files as build dependencies. Keep self-contained task
  diagnostics inline instead.
* State in each caption whether a WCS, noise model or calibration frame is
  supplied rather than estimated. Use small deterministic inputs and the
  public operations being explained, whether the code is inline or shared.
* Tutorial RST pages are the maintained numerical source. Hidden ``doctest``
  blocks only prepare temporary data and clean it up; visible examples print
  meaningful results for doctests to verify. Keep assertions and testing
  helpers in the test suite, rather than inside learning workflows.
* ``tutorial-data`` generates the shared offline archive in the build directory.
  It contains raw observations, local standards, reference products and ten
  notebook companions. ``tutorial-notebook:: page_name`` links the corresponding
  generated notebook. Edit the RST, not a second notebook copy.
* ``tutorial_notebooks.py`` converts prompted examples, expected console output,
  prose and plot directives. Setup cells locate the extracted bundle and set
  optional-array configuration before loading frames. Notebook figures use the
  current cell results. Sphinx figure wrappers replay the same visible RST
  calculations in temporary directories rather than duplicating the reduction.
* Run ``python docs/development/run_tutorials.py --output /tmp/tutorial-review``
  from the repository root with the docs/test dependencies installed. It executes
  all ten notebooks with fresh kernels, both sequentially using preceding
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
* The former worked notebooks have been consolidated into the CCD tutorials
  and the Astrometry.net/online-catalog tutorials. Keep external recipes as
  standard Python blocks, with prerequisites stated explicitly. Those calls
  require user-supplied images and indexes or live services and do not execute
  during documentation builds. Do not keep a second numerical notebook copy.
  ``check_docs.py --notebooks`` checks the generated offline companions;
  ``run_tutorials.py`` separately executes them.
* Large-image timing/memory studies belong in the studies repository. They are
  benchmarks, not stable correctness assertions or routine documentation builds.

API and editorial checklist
-----------------------------

The documentation toolchain requires Python 3.12 or newer; the maintained tox
and Read the Docs builds use Python 3.14. This is separate from the runtime
Python minimum of the library.

Use NumPy-style docstrings for public callables. Annotations define parameter
and return types; omit redundant type declarations from annotated docstring
fields. ``sphinx_autodoc_typehints`` reads the annotations and fills the
NumPyDoc parameter and return descriptions during the build. Signatures keep
the parameter names and defaults, while type information appears beside the
explanatory text. Keep names and behavioral
explanations, including units, shapes and estimator limitations. State defaults, literal string
choices and what each choice does, shape/dtype/unit requirements, mutation and
array ownership, missing arrays, masks, return types, and expected failures.
Use :data:`~astropop._typing.ImageLike` for 2D pixel-array inputs rather than
generic ArrayLike, retaining explicit scalar alternatives for uncertainties.
Use ImageInput for frame/FITS/path conversions and keep coordinate vectors and
image stacks distinct. Concrete NumPy array outputs retain their NDArray
contracts. These aliases describe shape requirements without enforcing them.
Distinguish methods (called with parentheses) from properties. Use “standard
deviation” for stored uncertainty; distinguish it from variance and scatter.
Explain zero-based ``(x, y)`` separately from NumPy ``[y, x]`` indexing.

A tutorial teaches a complete multi-module workflow. A module chapter explains
its API family, scientific assumptions and focused operations, and integrates
the generated reference contracts.
Link between them, preserving critical scientific limitations beside operations.
Do not claim speed, memory bounds or scientific validation without a specified
measurement or test. Follow :doc:`testing` for strict build commands and
ordinary versus live-service tests.

Strict checks and pull-request previews
---------------------------------------

``tox -e build_docs`` uses Python 3.14 and installs both ``docs`` and ``test``
extras. It checks generated-notebook schema/code syntax, runs local prompted
examples with remote data disabled, builds every page with strict reference
and warning checks, then verifies canonical pages and annotation-derived
parameter types on representative function and class pages. The GitHub docs
job selects that environment. Read the Docs also selects Python 3.14 and
``sphinx.fail_on_warning: true``; ``nitpicky=True`` in ``docs/conf.py`` checks
object references on both hosts. The tox job executes the ten offline
companions in fresh kernels, sequentially and independently. External solver
and online-query recipes are not executed. A rendering pass alone is not a
scientific test.

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

Moving tutorial sources
-----------------------

``SOURCE_PAGES`` in ``_ext/tutorial_notebooks.py`` maps stable workflow IDs to
nested RST paths. Update it with page moves, along with notebook Markdown links,
bundle source copying and the recursive tutorial fixture in ``docs/conftest.py``.
Figure replay uses those same paths. The archive retains the source directory
structure so helpers work both in the checkout and in extracted bundles.
Execute the companions sequentially and independently after changing their paths
or splitting a workflow; schema validation alone cannot detect shared-state
assumptions.
