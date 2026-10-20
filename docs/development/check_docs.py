"""Inspect documentation examples and check retained HTML URLs.

Run from the repository root; this script does not execute examples or contact
services. Use pytest and Sphinx for execution and rendering respectively.
"""

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path


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


def example_inventory():
    """Print a JSON inventory of authored examples and their validation modes."""
    docs = Path(__file__).resolve().parents[1]
    examples = []
    for path in sorted(docs.rglob('*.rst')):
        if any(part in ('api', '_build') for part in path.relative_to(docs).parts):
            continue
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            mode = None
            if stripped.startswith('>>>'):
                if index and lines[index - 1].strip().startswith(('>>>', '...')):
                    continue
                mode = 'pytest example'
            elif stripped.startswith('.. code-block:: text'):
                mode = 'stored console transcript; not executed'
            elif stripped.startswith('.. catalog-help::'):
                mode = 'local catalog help; rendered without online queries'
            elif stripped.startswith('.. plot::'):
                mode = 'Sphinx plot execution'
            elif stripped.startswith('.. code-block:: python'):
                mode = 'static recipe; not executed'
            elif stripped.endswith('::') and not stripped.startswith('..'):
                following = next((text.strip() for text in lines[index + 1:]
                                  if text.strip()), '')
                if not following.startswith('>>>'):
                    mode = 'static literal recipe; not executed'
            if mode:
                examples.append({'page': str(path.relative_to(docs)),
                                 'line': index + 1, 'validation': mode})
    for path in sorted((docs / 'ipynb').glob('*.ipynb')):
        examples.append({'page': str(path.relative_to(docs)),
                         'validation': 'stored notebook; external inputs; not executed'})
    print(json.dumps(examples, indent=2))


def check_notebooks():
    """Validate notebook schemas, compile code and reject stored error outputs.

    IPython syntax is transformed before compilation; cells are not executed.
    """
    import nbformat
    from IPython.core.inputtransformer2 import TransformerManager

    docs = Path(__file__).resolve().parents[1]
    notebooks = list((docs / 'ipynb').glob('*.ipynb'))
    cells = 0
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != 'code':
                continue
            code = TransformerManager().transform_cell(cell.source)
            compile(code, f'{path.name}:cell {index}', 'exec')
            if any(output.output_type == 'error' for output in cell.outputs):
                raise SystemExit(f'Stored error output: {path.name}:cell {index}')
            cells += 1
    print(f'Validated {len(notebooks)} notebook schemas and {cells} code cells; '
          'not executed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--html', type=Path, help='Sphinx HTML output directory')
    parser.add_argument('--inventory', action='store_true', help='List validation modes')
    parser.add_argument('--notebooks', action='store_true',
                        help='Check schema, code syntax and stored errors; no execution')
    args = parser.parse_args()
    if args.html:
        check_urls(args.html)
    if args.inventory:
        example_inventory()
    if args.notebooks:
        check_notebooks()
    if not args.html and not args.inventory and not args.notebooks:
        parser.error('Choose --html, --inventory or --notebooks')
