---
name: astropop-typing
description: Add or review Astropop type annotations for numerical arrays, astronomy objects, optional data, decorators and public APIs without changing runtime behavior.
---

# Astropop typing

Read the implementation, callers and tests before annotating. Cover module
functions, constructors, methods, properties and nested helpers; `self` and
`cls` normally need no explicit annotation. Exclude generated `version.py`.
Preserve signatures, accepted inputs, defaults, mutation and scientific behavior.
Annotations do not authorize adding coercion or runtime validation.
Astropy models interpret inline annotations on `evaluate` as physical units.
Use complete PEP 484 function type comments on those methods, preserving their
original runtime annotations and unit behavior; ordinary methods use inline hints.

Annotations are the source of parameter and return types. For annotated
callables, omit duplicate types from NumPy-style docstring fields; retain
parameter/return names, behavior, units, shapes, defaults, errors and explicit
strategy explanations. Configure Sphinx to render annotation-derived types,
rather than hiding annotations or maintaining a second type definition. Render
annotation-derived types beside parameter/return descriptions. With this
project's NumPyDoc layout, use the configured ``sphinx_autodoc_typehints``
extension and verify the rendered fields with ``check_docs.py --html``;
types appearing only in a signature do not satisfy this requirement.

Use Python 3.11-compatible types and postponed annotations where necessary.
Use the shared `ImageLike` alias for 2D pixel-array inputs (including spatial
mask, flag and uncertainty maps), rather than generic `ArrayLike`. It describes
`(rows, columns)` using `Annotated` metadata; it does not enforce shape. Keep
scalar-or-image uncertainties explicitly unioned with scalar types. Retain
`ImageInput` for conversion helpers accepting frames, FITS objects and paths.
Do not use `ImageLike` for source-coordinate vectors, 3D image stacks, or
arbitrary-dimensional numerical helpers. Use `numpy.typing.ArrayLike` for
those generic sequence/scalar inputs and `NDArray` for concrete array outputs;
specify a dtype only when the implementation guarantees it. Astropy quantities, units, coordinates, times, tables, headers and HDUs are
separate contracts, not interchangeable numerical arrays. Keep missing arrays
explicitly optional. Do not imply shape validation that the type system cannot
express. Reuse private aliases for recurring domain contracts rather than
repeating large unions.

Use `Literal` for closed string strategies, unions/overloads for distinct
supported inputs or outputs, `Self` for instance-preserving methods where
accurate, and `ParamSpec`/`TypeVar` for wrappers and decorators. A decorator that
can skip execution must include `None` in its return type. Use `object` for
opaque metadata and truly unrestricted values; reserve `Any` for interfaces
that actually require dynamic access. Do not fill missing annotations with
blanket `Any` or infer a contract only from a default value.

Keep imports needed only by annotations under `TYPE_CHECKING` when runtime
imports would cause a cycle or load an optional dependency. Preserve source
formatting and line endings. Update public docstrings when an existing contract
is ambiguous, without silently narrowing accepted input types to match a hint.

Use `astropop/_typing.py` for shared scientific aliases. Keep the `py.typed`
marker in packaged distributions. Run `python scripts/check_type_annotations.py`
for signature coverage; it accepts complete PEP 484 comments for Astropy models.

Validate annotation completeness and syntax separately from type correctness.
Use the repository's configured static checks when present; inspect their scope
before claiming a type-check pass. Run affected module tests and documentation
checks for changed public signatures. For broad annotation passes, compare
executable ASTs with annotations/type-only scaffolding removed to catch accidental
behavior changes. Report excluded generated code and external-library typing
limitations. Commit coherent module-sized batches with the affected prefix.
