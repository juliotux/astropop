# Development rules

## Scope and branches

- Inspect `git status`, the current branch, and nearby code/tests before editing. Preserve unrelated work.
- Keep independent fixes and features separate. A CI repair is not permission to redesign cache ownership, uncertainty semantics, or registration algorithms.
- When asked for a new independent branch, use the requested base; otherwise prefer local `main` and state the choice. Use a feature branch as a base only when the work depends on it. Do not silently reset, discard work, or rewrite history.
- Complete authorized local work without repeated confirmation. Local edits or commits do not themselves authorize a push, PR publication, merge, or release.

## Implementation

- Follow neighboring Python conventions and NumPy-style docstrings. Prefer the smallest change that fixes the cause while preserving public behavior.
- Preserve physical units, coordinate frames/epochs, masks, uncertainties, shapes, and in-place semantics unless the task explicitly changes them.
- Add regression coverage for behavioral fixes. Do not delete operations from exception or nonmutation tests just because their return values are unused.
- Use `tmp_path` for disposable files/directories. Never hand a shared directory such as the system temporary root to an object that deletes its cache directory.
- Test interpreter shutdown in a child process. Never call `atexit._run_exitfuncs()` inside pytest: it can stop coverage measurement and execute unrelated handlers.
- Keep helpers inside their owning test classes where practical. Put genuinely shared fixtures in `conftest.py`; avoid one-off test files when an existing module fits.

## Documentation

- Module chapters combine clear function/class explanations, simple examples,
  technical details and an integrated Reference/API section. Keep one indexed
  declaration per symbol; retain generated symbol pages under docs/api.
- Group multi-module tutorials under real subsection indexes with inputs,
  outputs and prerequisites. Place shared concepts with their owning modules;
  do not restore parallel task-guide, concepts and API-reference trees.
  Remove superseded source pages and update incoming links; do not retain
  orphan pages or compatibility gateways.
- Use `ImageLike` for 2D pixel-array inputs, with explicit scalar alternatives
  where supported. Keep coordinate vectors, stacks and broad `ImageInput`
  conversion contracts distinct.
- Module guides must give clear, descriptive, human-readable explanations of
  the module's functions and classes, with simple examples and useful outputs.
  Preserve distinct construction/input examples when reorganizing or shortening
  text; do not replace their teaching content with brief descriptions or API lists.
- Tutorials must combine several APIs to complete a more complex task from
  inputs through an inspected result. They complement module-level examples.
- Include relevant technical details, algorithm assumptions and references to
  original articles, scientific literature or technical documentation of widely
  used software. Tie references to the actual implemented behavior.
- Use inline `.. mermaid::` for workflow, dependency and ownership diagrams;
  do not draw these schemes with Matplotlib. Keep them readable and captioned.
- Use `.. plot::` with `:include-source: false` for scientific plots. Keep
  plotting code in the RST whenever possible; use external Python scripts only
  when functions defined in another file are needed.
- Follow [the documentation skill](../skills/astropop-docs/SKILL.md) for the full
  writing and validation requirements when changing documentation.

## Validation

- Treat `pyproject.toml`, `tox.ini`, and `.github/workflows/` in this checkout as authoritative. Do not assume a test option or feature from another branch exists here.
- Use an isolated Python environment with the appropriate test/docs extras. If using `XDG_CACHE_HOME` or `XDG_CONFIG_HOME`, create the directories first; Astropy may ignore nonexistent paths.
- Start with the affected tests and broaden to the relevant integration checks. Once checks pass, repeat them only for new changes, failures, or unresolved concerns.
- Keep ordinary CI offline and live catalog conformance in one explicit job when that infrastructure is present. Never make missing recorded responses silently fall through to the network.
- Report commands/results, skips, and outstanding checks accurately. A local Linux pass does not prove Windows, macOS, oldest dependencies, development dependencies, or remote CI passed.
- Do not globally suppress warnings or weaken tolerances to hide a failure. Expect deliberate warnings locally and fix unintended behavior at its source.

## Commits and handoff

Follow the module-prefix convention in [AGENTS.md](../AGENTS.md). Review the staged diff and commit in small batches, grouping implementation with its applicable tests. Stage named paths or hunks so unrelated changes are not included.

PR descriptions should explain the concrete failure and resulting behavior, then give relevant validation and known limitations. Distinguish implemented fixes from future work and verified service behavior from recordings or historical observations.
