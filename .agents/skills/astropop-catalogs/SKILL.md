---
name: astropop-catalogs
description: Maintain Astropop SIMBAD, VizieR, and Gaia queries and their shared recorded/live tests, including schema changes, timeouts, and complete query results.
---

# Astropop catalogs

Read the affected implementation in `astropop/catalogs/`, the corresponding tests in `tests/test_catalogs_online.py`, and the applicable VizieR YAML. Verify the installed Astroquery API and actual response shape before changing an endpoint or column name.

## Compatibility invariants

- SIMBAD can return degree-valued lowercase coordinates or legacy sexagesimal uppercase coordinates. Never infer hour-angle units from the semantic name RA alone. Handle both `None` and empty tables before indexing.
- TAP photometry may have multiple rows per source/filter. Preserve source identity, band case, masks, proper motions, and bibliographic information when normalizing it.
- Choose cone-search matches by separation rather than trusting response order. Identifier priority and prefix boundaries matter; substring matching can select a different catalog designation.
- VizieR column aliases must work in outgoing column selection and incoming response normalization without renaming an already normalized column twice. Check sort-column names and epoch values against the configured time format.
- Gaia asynchronous queue delays do not establish that a URL changed. Preserve the requested query mode and document timeout limitations honestly.
- Where synchronous Gaia pagination is implemented, preserve integer source IDs, magnitude/cone filters on every page, exact column matching, final angular-distance ordering, and failure rather than partial results. Include page boundaries and IDs above `2**53` in deterministic coverage.
- Keep timeout changes scoped to the client's connections, restore patched methods, close connections, and share a deadline across retries/pages. A socket timeout is not a hard deadline covering every DNS, streaming, or parsing step.

## Recorded and live tests

First check whether `tests/data/catalog_responses/README.md` and replay options exist on this branch. Read that README before recording. On branches without replay infrastructure, do not report skipped remote tests as offline conformance.

Where supported:

```sh
python -m pytest tests/test_catalogs_online.py --remote-data=none
python -m pytest tests/test_catalogs_online.py --remote-data=any
```

Keep assertions and collected test IDs shared between modes. Mock at the HTTP layer so Astroquery still builds queries and parses responses. Store compact fixtures per catalog, not per test. Offline mode must fail for unknown requests; live mode must bypass replay and result caches. Account for VizieR's explicit cache argument and import-time Gaia announcements.

Refresh fixtures only when needed, using the documented explicit recording mode in one process. Preserve strict matching of query criteria, coordinates, filters, and pagination cursors. Normalize ordering only where semantically harmless, such as simple SELECT column lists. Review recordings for credentials/cookies and unrelated changes; rerun offline after recording.

Keep deterministic edge-case tests alongside the catalog tests, with class-owned helpers where practical. Assert intentional no-result or proper-motion warnings around the precise call instead of suppressing them globally. A live Gaia skip option, if present, must leave offline and deterministic Gaia tests enabled and must not disable the dedicated conformance job's Gaia checks.
