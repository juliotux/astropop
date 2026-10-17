---
name: astropop-docs
description: Write or repair Astropop reStructuredText guides, NumPy-style API docstrings, and Sphinx examples using the behavior implemented on the current branch.
---

# Astropop documentation

Read the implementation, nearby tests, and the relevant guide under `docs/`. Keep constructor APIs distinct from conversion helpers and describe actual defaults, mutation behavior, units, masks, and uncertainty semantics.

Use runnable examples for deterministic local behavior where the existing documentation tooling supports them. Use static Python blocks for live catalog examples so documentation builds do not depend on external catalog queries. Remove IPython-only options when converting a directive to `code-block`.

For FrameData guides, check shape/dtype constraints, copying and array sharing, metadata/WCS, FITS extension selection, overwrite defaults, and whether interchange preserves Boolean masks or full flag bits. Document only storage/configuration features available on this branch.

For build failures:

- Check reStructuredText section hierarchy, blank lines before lists, indentation, and balanced inline markup.
- Keep the local checkout first on the documentation import path so autodoc describes the changed source.
- Inspect `docs/conf.py` and docs extras rather than installing arbitrary extensions. Notebook rendering may require the Pandoc executable in addition to a Python package.
- Distinguish rendering/autodoc errors, external inventory/network failures, and existing warnings. Do not suppress warnings globally to produce a green build.

Validation depends on the change:

```sh
python -m pytest docs/path/to/changed-guide.rst
python -m sphinx -b html docs /tmp/astropop-docs-build
```

Replace the guide path with the actual file. Use the configured tox docs environment when reproducing CI. For a warning audit, add `-W --keep-going` and report its result separately from the ordinary build. Inspect relevant rendered examples when layout or generated API content changes.

Keep generated HTML, temporary images, and build caches out of commits. Use `docs:` for documentation commits and name the affected module when docstring updates are part of its implementation fix.
