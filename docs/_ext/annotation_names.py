"""Resolve private import names when Sphinx falls back to string annotations."""

from __future__ import annotations

import re
import sys
import types
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from docutils import nodes
    from sphinx import addnodes
    from sphinx.application import Sphinx
    from sphinx.environment import BuildEnvironment


def format_signature(
    app: Sphinx, what: str, name: str, obj: Any, options: Any,
    signature: str | None, return_annotation: str | None,
) -> tuple[str | None, str | None]:
    """Keep alias names readable without changing the runtime annotations.

    Sphinx can retain strings when an annotation contains a TYPE_CHECKING-only
    import. Nested aliases can also render as TypeAliasForwardRef(...) in
    Sphinx 9. Resolve those names using the documented object's own namespace.
    """
    namespace = vars(sys.modules[obj.__module__]) if hasattr(obj, '__module__') else {}
    aliases = dict(app.config.autodoc_type_aliases)
    for key, value in namespace.items():
        if isinstance(value, types.ModuleType):
            aliases[key] = value.__name__
        elif isinstance(value, type):
            aliases[key] = value.__module__ + '.' + value.__qualname__

    def qualify(match: re.Match[str]) -> str:
        """Expand an imported name using the documented object's namespace."""
        token = match.group()
        if token in aliases:
            return aliases[token]
        head, dot, tail = token.partition('.')
        return aliases.get(head, head) + dot + tail

    def format_annotation(text: str | None) -> str | None:
        """Resolve annotation names while preserving quoted values and defaults."""
        if text is None:
            return None
        text = re.sub(r"TypeAliasForwardRef\(['\"]([^'\"]+)['\"]\)", r'\1', text)
        # NumPy dtype classes are defaults; their repr is not Python syntax.
        text = re.sub(r"<class '([^']+)'>", r'\1', text)
        # Defaults and Literal choices are values, not names to resolve.
        parts = re.split(r'''('(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*")''', text)
        for i in range(0, len(parts), 2):
            parts[i] = re.sub(r'\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\b', qualify, parts[i])
        return ''.join(parts)

    return format_annotation(signature), format_annotation(return_annotation)


def resolve_alias(
    app: Sphinx, env: BuildEnvironment, node: addnodes.pending_xref,
    contnode: nodes.TextElement,
) -> nodes.reference | None:
    """Link type aliases registered as data/attributes instead of classes."""
    from sphinx.ext.intersphinx import missing_reference

    aliases = app.config.autodoc_type_aliases
    target = node['reftarget']
    if node.get('refdomain') != 'py' or not (
        target in aliases or target in aliases.values() or target.startswith('numpy.')
    ):
        return None
    reference = node.copy()
    reference['reftarget'] = aliases.get(target, target)
    reference['reftype'] = 'obj'
    local = env.domains['py'].resolve_xref(
        env, node['refdoc'], app.builder, 'obj', reference['reftarget'], reference, contnode
    )
    return local or missing_reference(app, env, reference, contnode)


def setup(app: Sphinx) -> dict[str, bool]:
    """Register signature formatting and type-alias reference resolution."""
    app.setup_extension('sphinx.ext.autodoc')
    app.connect('autodoc-process-signature', format_signature, priority=800)
    app.connect('missing-reference', resolve_alias, priority=800)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
