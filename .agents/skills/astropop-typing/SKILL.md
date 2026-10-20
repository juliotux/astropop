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

Use Python 3.11-compatible types and postponed annotations where necessary.
Use `numpy.typing.ArrayLike` for inputs accepting sequences/scalars and
`NDArray` for arrays; specify a dtype only when the implementation guarantees
it. Astropy quantities, units, coordinates, times, tables, headers and HDUs are
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

Validate annotation completeness and syntax separately from type correctness.
Use the repository's configured static checks when present; inspect their scope
before claiming a type-check pass. Run affected module tests and documentation
checks for changed public signatures. For broad annotation passes, compare
executable ASTs with annotations/type-only scaffolding removed to catch accidental
behavior changes. Report excluded generated code and external-library typing
limitations. Commit coherent module-sized batches with the affected prefix.
