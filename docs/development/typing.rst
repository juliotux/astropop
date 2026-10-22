Type annotation conventions
===========================

Annotations define parameter and return types; NumPy-style docstrings explain
behavior, units, shapes, defaults and errors. Sphinx renders the types beside
their descriptions. Do not duplicate annotations in docstring type fields.

Use Python 3.11-compatible annotations and the shared contracts in
:doc:`../modules/utilities/types`. ImageLike describes 2D pixel inputs, while
ImageInput additionally covers conversion inputs such as frames and FITS paths.
Coordinates, scalar-or-image uncertainties and image stacks need their own
contracts. Shape metadata does not enforce runtime or static dimensionality.

Preserve runtime behavior when adding hints. Astropy model evaluate methods
interpret inline annotations as physical units; those methods use complete
PEP 484 function type comments instead. Keep imports used only for hints under
TYPE_CHECKING when importing them at runtime would cause a cycle.

Run the signature-presence check::

    python scripts/check_type_annotations.py

It excludes generated version metadata and does not constitute a full static
type-check pass. Review callers and return branches, run affected tests, and
check the rendered parameter types when changing public contracts. Broad typing
passes should compare executable ASTs after removing annotation-only scaffolding.
