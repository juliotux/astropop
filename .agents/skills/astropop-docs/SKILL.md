---
name: astropop-docs
description: Write or repair Astropop reStructuredText guides, NumPy-style API docstrings, and Sphinx examples using the behavior implemented on the current branch.
---

# Astropop documentation

Read the implementation, nearby tests, and the relevant guide under `docs/`. Keep constructor APIs distinct from conversion helpers and describe actual defaults, mutation behavior, units, masks, and uncertainty semantics.

## Module guides and tutorials serve different readers

Module chapters combine explanatory documentation and their API reference. Organize their explanations
around the module's functions and classes, with local, simple examples and
useful results. A page that strings multiple modules together to produce a
complete reduction belongs in tutorials. Keep cross-cutting concepts and
troubleshooting identifiable as supporting material rather than treating them
as substitutes for a module guide.

Use `docs/modules/polarimetry/index.rst` as the editorial model: explain the
module's scientific model, then its classes/functions, their options, results
and focused examples. Make API ownership visible in section headings where
useful. Technical equations, algorithm details, figures and references belong
beside the function or class they explain. Do not turn task guides into general
reduction advice, observing-policy essays, troubleshooting checklists, benchmark
instructions or extended cross-module workflows. Move such material to the
appropriate tutorial, concept or development page, leaving a relevant link.
A navigation change alone does not fix a guide whose body still avoids
explaining the API. Preserve useful prose/examples at their maintained location
and update all incoming links when moving them. Remove superseded source pages;
do not keep orphan pages or compatibility gateways.

- **Module documentation must be clear, descriptive and human-readable about
  every function and class the module offers.** Explain what each does, when
  to use it, how to construct/call it and how to interpret its result. Include
  relevant inputs, units, defaults, choices, outputs and scientific limitations
  in connected prose. An API name list or generated signatures cannot replace
  these explanations.
- **Every module guide must contain simple examples.** Demonstrate the basic
  construction/use of each class and the module's functions; related functions
  may share a focused example if their individual use and results remain clear.
  Explain the examples before the code and show useful outputs afterward.
  Include distinct supported input forms when they teach different behavior.
  For SourcesCatalog, retain separate examples for an astrometry-only catalog,
  plain multi-band magnitudes and multi-band QFloat magnitudes with errors.
- **Preserve explanatory content while editing.** Before removing or merging an
  example, identify the behavior it teaches and retain that coverage in the
  revised page. If an extended example moves to another page, retain the basic
  module example locally and link to the extended treatment. Shorter text is not an
  improvement if it replaces a useful explanation/example with a terse summary.
  Moving API declarations to reference pages must not strip module guides of
  their explanatory prose or examples.
- **Tutorials must complete multi-step tasks using APIs from several modules together.**
  Include preparation, the full processing sequence, inspection and a usable
  result. Tutorials complement simple module examples; do not move all examples
  out of module guides or substitute disconnected API demonstrations for a
  complete tutorial workflow.
- **Add technical detail and references where relevant.** Explain algorithms,
  equations, estimator assumptions and conventions at the point they help the
  reader understand the operation. Prefer original articles, scientific
  references and technical descriptions from widely used software. Verify that
  each source supports the specific claim, link/cite it near that explanation,
  and distinguish the implemented Astropop behavior from the source's method.
  A bibliography or an external link alone does not explain our API.

## Structure and contracts

Annotations are the source of parameter and return types. For annotated
callables, omit duplicate types from NumPy-style docstring fields; retain
parameter/return names, behavior, units, shapes, defaults, errors and explicit
strategy explanations. Configure Sphinx to render annotation-derived types,
rather than hiding annotations or maintaining a second type definition. Render
annotation-derived types beside parameter/return descriptions. With this
project's NumPyDoc layout, use the configured ``sphinx_autodoc_typehints``
extension and verify the rendered fields with ``check_docs.py --html``;
types appearing only in a signature do not satisfy this requirement.
Use the shared `ImageLike` contract for 2D pixel-array inputs, including masks
and flag maps, instead of generic `ArrayLike`. Explain `(rows, columns)` and
operation-specific shape/dtype restrictions. Keep scalar uncertainty alternatives
explicit. `ImageInput` additionally covers frame/FITS/path conversion inputs;
coordinate vectors and image stacks are different contracts. NumPy array
outputs can retain concrete `NDArray` types and guaranteed dtypes. Do not claim
that the alias enforces dimensionality statically or at runtime.

Use `docs/getting_started/` for installation, first steps and example setup;
`docs/tutorials/` for complete multi-module workflows grouped beneath real
subsection landing pages; `docs/modules/` for explanatory module chapters with
integrated Reference/API; `docs/development/` for contributor guidance; and
`docs/project/` for migration, citation and license information. Do not restore
parallel task-guide, concepts and API-reference trees. Place technical concepts
with the module that owns their contract. Delete obsolete source paths after
moving their content and update internal links, downloads and helper registries.
Do not leave orphan pages, redirect stubs or compatibility-anchor placeholders.

Each module chapter introduces its subject, gives simple examples, explains
functions/classes and relevant scientific details, and contains Reference/API
summaries. Individual symbol pages remain generated under `docs/api/`; merging
API ownership does not require putting every signature on one giant page.

Tutorial groups cover CCD reduction, photometric measurements, astrometry and
catalogs, dual-beam polarimetry, and diagnostics/data handling. Their landing
pages must describe prerequisites, inputs, outputs, dependencies and offline,
network or external-program requirements. Keep half/quarter-wave workflows
independent and preserve cross-module process coverage.

Consult [documentation maintenance](../../../docs/development/documentation.rst)
for the maintained structure and validation policy.

Keep each indexed API declaration in one place. Generated `docs/api/` pages
remain generated; do not edit or commit them. Validate the current canonical
source pages and their built HTML with `check_docs.py --html`. Use a clean output
directory after moves so obsolete HTML cannot mask broken links. Hosting redirects
are separate from the maintained source tree; do not create orphan files for them.

For parameters with named strategies, explain every literal string explicitly:
for example, ``'or'`` combines bits by OR, ``'and'`` keeps common bits, and
``'no_merge'`` produces zero flags. Do not leave readers to infer which behavior
matches which name from their order in a sentence.

## Examples and scientific behavior

Use standard `code-block:: python` directives for recipes and `code-block:: text` for stored console transcripts. Do not add IPython directives or execution-only options. Keep deterministic prompted examples and plot directives where execution is useful; static code blocks are not executed. Render VizieR help with `catalog-help`, using packaged configuration without constructing a catalog or querying services.

Separate logical steps in examples with blank lines, including the bodies of
long loops. Add comments immediately beside small steps whose purpose, units,
coordinate direction, selection rule or scientific assumption needs explanation.
Explain why that operation is needed; do not merely restate the next line or
add a title comment at the start of each block. Simple self-explanatory snippets
need no forced commentary. Preserve existing outputs and keep these comments
and blank lines in generated notebook companions too.

Seed random generators in executable examples and plots. Distinguish pytest
examples, Sphinx plot execution, stored transcripts, static recipes and
stored notebooks. Rendering or syntax-checking a notebook does not execute its
workflow; do not report it as executed.

For FrameData guides, check shape/dtype constraints, copying and array sharing, metadata/WCS, FITS extension selection, overwrite defaults, and whether interchange preserves Boolean masks or full flag bits. Document only storage/configuration features available on this branch.

## Scientific plots and diagrams

- Use inline `.. mermaid::` diagrams for workflows, dependencies, architecture
  and storage ownership. Let the diagram engine lay out nodes and arrows;
  do not draw these schemes with Matplotlib. Use captions and alternative text, and split large diagrams into readable
  stages. Review browser rendering, light/dark themes and fullscreen viewing;
  a Sphinx build alone does not validate Mermaid syntax.
- Use Matplotlib for scientific plots, image displays and spatial/data
  annotations. Code these with the Sphinx `.. plot::` directive
  and set `:include-source: false` explicitly.
- Keep the plotting code inside the `.rst` page whenever possible. A simple,
  self-contained figure must not be moved into a separate Python script merely
  to shorten the RST. Ordinary imports of NumPy, Matplotlib or the public API do
  not by themselves require an external plotting script.
- Use an external Python script only when the plot needs to call functions
  defined in another file, such as a shared workflow or reusable figure helper.
  Explain the shared dependency and keep the source tracked for rebuilds.
- Keep labels, units, alternative text and an explanatory caption. Seed random
  inputs and validate the rendered figure in the Sphinx build.

For a self-contained figure, use this form:

```rst
.. plot::
    :include-source: false
    :alt: Signal measured in three successive exposures.
    :caption: A small illustrative sequence, plotted directly from the RST.

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [10, 12, 11], marker='o')
    ax.set(xlabel='Exposure number', ylabel='Signal (ADU)')
```

## Build and validation

For build failures:

- Check reStructuredText section hierarchy, blank lines before lists, indentation, and balanced inline markup.
- Keep the local checkout first on the documentation import path so autodoc describes the changed source.
- Inspect `docs/conf.py` and docs extras rather than installing arbitrary extensions. Notebook rendering may require the Pandoc executable in addition to a Python package.
- Distinguish rendering/autodoc errors, external inventory/network failures, and existing warnings. Do not suppress warnings globally to produce a green build.

For a focused edit, run pytest on the actual changed guide with
`--remote-data=none`. For documentation CI, use `tox -e build_docs`; inspect
`tox.ini` for its current interpreter, extras and command sequence. To run the
underlying checks directly from the repository root:

```sh
python -m pytest docs --remote-data=none
python docs/development/check_docs.py --notebooks
python -m sphinx -b html -E -a -n -W --keep-going docs /tmp/astropop-docs-build
python docs/development/check_docs.py --html /tmp/astropop-docs-build
```

When moving tutorial sources, update SOURCE_PAGES in tutorial_notebooks.py,
the bundle source copying, figure replay, pytest fixtures and notebook links
together. Execute both sequential and independent notebook modes.

The notebook check validates schema, code syntax and stored errors without
execution. `--inventory` lists authored-example validation modes. Use a fresh
Sphinx environment (`-E`) after moving pages or API declarations to avoid stale indexed
objects; `-a -n -W` rewrites all pages and treats missing references/warnings as
failures. Inspect relevant rendered pages when navigation or layout changes.
Read the Docs PR-preview enablement is a project setting, not proof supplied by
`.readthedocs.yml`; distinguish configured build checks from verified hosted
publication.

Keep generated HTML, temporary images, and build caches out of commits. Use `docs:` for documentation commits and name the affected module when docstring updates are part of its implementation fix.
