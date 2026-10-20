---
name: astropop-docs
description: Write or repair Astropop reStructuredText guides, NumPy-style API docstrings, and Sphinx examples using the behavior implemented on the current branch.
---

# Astropop documentation

Read the implementation, nearby tests, and the relevant guide under `docs/`. Keep constructor APIs distinct from conversion helpers and describe actual defaults, mutation behavior, units, masks, and uncertainty semantics.

## Structure and contracts

Annotations are the source of parameter and return types. For annotated
callables, omit duplicate types from NumPy-style docstring fields; retain
parameter/return names, behavior, units, shapes, defaults, errors and explicit
strategy explanations. Configure Sphinx to render annotation-derived types,
rather than hiding annotations or maintaining a second type definition.

Use `docs/tutorials/` for complete learning workflows, `docs/how_to/` for tasks,
`docs/concepts/` for shared assumptions, `docs/reference/` for API declarations,
and `docs/development/` for contributor guidance. Existing module guides remain
supported entry points. Consult [documentation maintenance](../../../docs/development/documentation.rst)
for the maintained structure and example policy.

Keep each indexed API declaration in one place. Generated `docs/api/` pages
remain generated; do not edit or commit them. When extracting sections, preserve
old URLs and anchors using `docs/development/urls.json` and compatibility links.
Do not weaken the URL manifest merely to make a broken migration pass.

For parameters with named strategies, explain every literal string explicitly:
for example, ``'or'`` combines bits by OR, ``'and'`` keeps common bits, and
``'no_merge'`` produces zero flags. Do not leave readers to infer which behavior
matches which name from their order in a sentence.

## Examples and scientific behavior

Use runnable examples for deterministic local behavior where the existing documentation tooling supports them. Use static Python blocks for live catalog examples so documentation builds do not depend on external catalog queries. Remove IPython-only options when converting a directive to `code-block`.
Seed random generators in executable examples and plots. Distinguish pytest
examples, Sphinx IPython/plot execution, verbatim transcripts, static recipes and
stored notebooks. Rendering or syntax-checking a notebook does not execute its
workflow; do not report it as executed.

For FrameData guides, check shape/dtype constraints, copying and array sharing, metadata/WCS, FITS extension selection, overwrite defaults, and whether interchange preserves Boolean masks or full flag bits. Document only storage/configuration features available on this branch.

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

The notebook check validates schema, code syntax and stored errors without
execution. `--inventory` lists authored-example validation modes. Use a fresh
Sphinx environment (`-E`) after moving API declarations to avoid stale indexed
objects; `-a -n -W` rewrites all pages and treats missing references/warnings as
failures. Inspect relevant rendered pages when navigation or layout changes.
Read the Docs PR-preview enablement is a project setting, not proof supplied by
`.readthedocs.yml`; distinguish configured build checks from verified hosted
publication.

Keep generated HTML, temporary images, and build caches out of commits. Use `docs:` for documentation commits and name the affected module when docstring updates are part of its implementation fix.
