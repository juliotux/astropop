---
name: astropop-ci
description: Diagnose and fix Astropop test, coverage, dependency, tox, and GitHub Actions failures while keeping ordinary CI independent of live catalogs.
---

# Astropop CI

Inspect the failing job, current configuration, and failing test before choosing a fix. Separate a collection/import failure, assertion failure, service outage, dependency incompatibility, and coverage-measurement failure; they need different remedies.

## Sources and checks

- Dependency bounds and pytest/coverage settings: `pyproject.toml`.
- Environment factors, extras, and commands: `tox.ini`.
- Runner/platform and live/offline topology: `.github/workflows/`.
- Catalog replay and custom options, if implemented: `tests/conftest.py` and `tests/data/catalog_responses/README.md`.

Use the environment actually named in the failing job. Inspect `tox list` and `tox config -e <environment>` before assuming factors expand correctly. Check Python support and oldest dependency compatibility together. A version pin should explain the concrete incompatible API; it is not a substitute for claiming a migration is complete.

Typical checks, adjusted to the installed tooling and changed files:

```sh
python -m pytest tests/test_affected_module.py
python -m pytest tests docs --remote-data=none --cov=astropop --cov-report=term-missing
python -m ruff check astropop tests
git diff --check
```

The first path is a placeholder for the actual test module. If Ruff is not configured/installed in this checkout, inspect the repository's lint configuration first.

## Documentation CI

For documentation failures, inspect `tox config -e build_docs` and reproduce
with `tox -e build_docs`. Its checks cover offline prompted examples, stored
notebook schema/syntax/errors, a clean strict Sphinx build, and canonical HTML
pages without orphan sources through `docs/development/check_docs.py`. Check the interpreter
and extras against the GitHub docs job and `.readthedocs.yml`; use their current
values rather than assuming the ordinary test matrix covers the docs environment.

Distinguish a failed example, missing canonical page, broken reference, unavailable
external inventory, and missing rendering tools such as Pandoc. A notebook
validation/rendering pass does not prove execution of its external workflow.
Do not suppress warnings, omit maintained pages from validation, add orphan
compatibility pages, or enable live queries to hide a failure. For page/API moves and example policy, follow the
[documentation maintenance guide](../../../docs/development/documentation.rst).
Report an isolated tox run separately from checks run in an existing environment,
and configured hosted builds separately from observed remote results.

## Coverage and resources

For a coverage collapse, inspect collection/skips and whether any test executes process-wide shutdown handlers. Preserve the existing cache API while repairing harness isolation unless cache redesign is explicitly in scope. Test exit behavior with a subprocess, not pytest's own exit handlers. Use per-test cache directories and close file handles, including on Windows.

## Catalog topology

When the replay infrastructure exists, regular Python/platform/dependency jobs and coverage use recordings. Keep exactly one separate live catalog conformance job, including Gaia; a fast/offline Python job must still exist alongside it. Inspect the YAML embedded inside reusable-workflow `envs` strings as well as the outer workflow.

Do not hide conformance failures by switching the live job to replay or silently skipping a service. Record an outage separately from a deterministic code failure. Use the catalog skill for query/schema and fixture changes.

Validate relevant local checks, then report which remote/platform jobs remain unrun. Check the staged diff for unrelated feature code before making a module-prefixed commit.
