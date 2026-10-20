"""Inspect documentation examples and check retained HTML URLs.

Run from the repository root; this script does not execute examples or contact
services. Use pytest and Sphinx for execution and rendering respectively.
"""

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import sys


class Anchors(HTMLParser):
    """Collect HTML IDs for retained documentation-link checks."""

    def __init__(self):
        """Initialize the HTML parser with an empty set of anchor IDs."""
        super().__init__()
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        """Record IDs from a start tag, ignoring its other attributes."""
        self.ids.update(value for key, value in attrs if key == 'id')


def check_urls(html):
    """Verify retained pages and anchors, exiting with all missing targets."""
    manifest = json.loads(Path(__file__).with_name('urls.json').read_text())
    failures = []
    for name, contract in manifest['pages'].items():
        page = html / name
        if not page.is_file():
            failures.append(f'Missing page: {name}')
            continue
        parser = Anchors()
        parser.feed(page.read_text())
        for anchor in contract['anchors']:
            if anchor not in parser.ids:
                failures.append(f'Missing anchor: {name}#{anchor}')
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f"Verified {len(manifest['pages'])} retained page URLs and anchors")


_DIRECTIVE_MODES = {
    '.. code-block:: text': 'stored console transcript; not executed',
    '.. catalog-help::': 'local catalog help; rendered without online queries',
    '.. tutorial-notebook::': 'generated RST companion; execute with development/run_tutorials.py',
    '.. tutorial-data::': 'generated local FITS download; no online queries',
    '.. plot::': 'Sphinx plot execution',
    '.. code-block:: python': 'static recipe; not executed',
}


def example_validation(lines: list[str], index: int) -> str | None:
    """Classify an example's opening line without counting continuation prompts."""
    stripped = lines[index].strip()
    if stripped.startswith('>>>'):
        if index and lines[index - 1].strip().startswith(('>>>', '...')):
            return None
        return 'pytest example'
    for prefix, mode in _DIRECTIVE_MODES.items():
        if stripped.startswith(prefix):
            return mode
    if stripped.endswith('::') and not stripped.startswith('..'):
        following = next((text.strip() for text in lines[index + 1:]
                          if text.strip()), '')
        if not following.startswith('>>>'):
            return 'static literal recipe; not executed'
    return None


def example_inventory():
    """Print a JSON inventory of authored examples and their validation modes."""
    docs = Path(__file__).resolve().parents[1]
    examples = []
    for path in sorted(docs.rglob('*.rst')):
        if any(part in ('api', '_build') for part in path.relative_to(docs).parts):
            continue
        lines = path.read_text().splitlines()
        for index in range(len(lines)):
            mode = example_validation(lines, index)
            if mode:
                examples.append({'page': str(path.relative_to(docs)),
                                 'line': index + 1, 'validation': mode})
    print(json.dumps(examples, indent=2))


def check_notebooks():
    """Validate generated notebook schemas and Python syntax without execution."""
    import nbformat

    docs = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(docs / '_ext'))
    from tutorial_notebooks import WORKFLOWS, build_notebook

    cells = 0
    for page, name in WORKFLOWS:
        notebook = build_notebook(docs / 'tutorials' / f'{page}.rst')
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != 'code':
                continue
            compile(cell.source, f'{name}.ipynb:cell {index}', 'exec')
            cells += 1
    print(f'Validated {len(WORKFLOWS)} generated notebook schemas and {cells} code cells; '
          'not executed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--html', type=Path, help='Sphinx HTML output directory')
    parser.add_argument('--inventory', action='store_true', help='List validation modes')
    parser.add_argument('--notebooks', action='store_true',
                        help='Check generated notebook schemas and Python syntax; no execution')
    args = parser.parse_args()
    if args.html:
        check_urls(args.html)
    if args.inventory:
        example_inventory()
    if args.notebooks:
        check_notebooks()
    if not args.html and not args.inventory and not args.notebooks:
        parser.error('Choose --html, --inventory or --notebooks')
