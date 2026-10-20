# Working on Astropop

Read [.agents/rules/development.md](rules/development.md) before changing this repository. These instructions apply throughout the checkout; explicit user instructions take precedence.

## Commit messages

Begin every commit subject with the module or area being changed, followed by a colon and a short description. Prefer a space after the colon:

- `catalogs: handle empty SIMBAD responses`
- `framedata: isolate cache shutdown tests`
- `image: exclude rejected samples from summed errors`
- `docs: explain FrameData conversion`
- `ci: run live catalog checks in a separate job`
- `agents: add repository skills`

Make small, coherent commits. Use the affected module for its tests too; reserve `ci:` for workflows, environments, dependencies, and cross-cutting test infrastructure. Split unrelated changes rather than hiding them behind a broad prefix. Do not rewrite existing commits solely to enforce this convention unless asked.

## Repository skills

Read the relevant skill when its workflow applies. Ordinary small edits do not require loading every skill.

| Skill | Use when |
| --- | --- |
| [astropop-ci](skills/astropop-ci/SKILL.md) | Diagnosing test, coverage, dependency, or GitHub Actions failures. |
| [astropop-catalogs](skills/astropop-catalogs/SKILL.md) | Changing SIMBAD, VizieR, Gaia, or recorded/live catalog tests. |
| [astropop-scientific-code](skills/astropop-scientific-code/SKILL.md) | Changing image reduction, FrameData, units, uncertainties, photometry, or polarimetry. |
| [astropop-docs](skills/astropop-docs/SKILL.md) | Editing API docs, scientific examples, or fixing Sphinx builds. |
| [astropop-typing](skills/astropop-typing/SKILL.md) | Adding or reviewing package type annotations without changing runtime contracts. |
