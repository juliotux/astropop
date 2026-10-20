"""Render catalog help from packaged configuration without querying services."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from docutils import nodes
from sphinx.util.docutils import SphinxDirective

if TYPE_CHECKING:
    from sphinx.application import Sphinx


class CatalogHelp(SphinxDirective):
    """Insert the same local help text returned by a configured VizieR class."""

    required_arguments = 1
    has_content = False

    def run(self) -> list[nodes.Node]:
        """Track the YAML dependency and display help without creating a catalog."""
        from astropop.catalogs import vizier

        name = self.arguments[0]
        config = Path(vizier.__file__).parent / 'vizier_catalogs' / f'{name}.yml'
        if not name.isidentifier() or not config.is_file():
            raise self.error(f'Unknown bundled VizieR catalog: {name}')
        self.env.note_dependency(str(config))
        self.env.note_dependency(vizier.__file__)
        text = getattr(vizier, name).help()
        return [nodes.literal_block(text, text, language='text')]


def setup(app: Sphinx) -> dict[str, bool]:
    """Register the local catalog-help directive for Sphinx builds."""
    app.add_directive('catalog-help', CatalogHelp)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
