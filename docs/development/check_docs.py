"""Inspect documentation examples and check canonical HTML pages.

Run from the repository root; this script does not execute examples or contact
services. Use pytest and Sphinx for execution and rendering respectively.
"""

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import sys


def check_pages(html: Path) -> None:
    """Require built HTML for every authored page and reject orphan placeholders."""
    docs = Path(__file__).resolve().parents[1]
    pages = [page for page in docs.rglob('*.rst')
             if not any(part in ('api', '_build', '_templates')
                        for part in page.relative_to(docs).parts)]
    failures = []
    for page in sorted(pages):
        relative = page.relative_to(docs)
        if ':orphan:' in page.read_text().splitlines():
            failures.append(f'Orphan source page: {relative}')
        if not (html / relative.with_suffix('.html')).is_file():
            failures.append(f'Missing built page: {relative}')
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f'Verified {len(pages)} canonical documentation pages; no orphan sources')


class ParameterTypes(HTMLParser):
    """Collect nonempty types rendered beside parameter names in API fields."""

    def __init__(self) -> None:
        """Initialize per-paragraph state and the set of typed parameters."""
        super().__init__()
        self.typed: set[str] = set()
        self._name: list[str] | None = None
        self._parameter = ''
        self._has_type = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Recognize parameter labels and the annotation renderer's type span."""
        if tag == 'strong':
            self._name = []
        if tag == 'span' and 'sphinx_autodoc_typehints-type' in dict(attrs).get('class', '').split():
            self._has_type = True

    def handle_data(self, data: str) -> None:
        """Read a bold parameter name and confirm the following type has text."""
        if self._name is not None:
            self._name.append(data)
        elif self._parameter and self._has_type and data.strip():
            self.typed.add(self._parameter)

    def handle_endtag(self, tag: str) -> None:
        """Finish labels and prevent parameter state leaking into later prose."""
        if tag == 'strong' and self._name is not None:
            self._parameter = ''.join(self._name).strip()
            self._name = None
        elif tag == 'p':
            self._parameter = ''
            self._has_type = False


def check_parameter_types(html: Path) -> None:
    """Require annotation-derived types in representative function/class fields."""
    required = {
        'astropop.framedata.FrameData': {'data', 'dtype', 'uncertainty'},
        'astropop.image.imcombine.imcombine': {'frames', 'method', 'dtype'},
        'astropop.image.imarith.imarith': {'operand1', 'operand2', 'merge_flags'},
        'astropop.astrometry.guess_coordinates': {'ra', 'dec', 'skycoord'},
        'astropop.catalogs.SourcesCatalog': {'ids', 'mag'},
    }
    failures = []
    for name, parameters in required.items():
        page = html / 'api' / f'{name}.html'
        if not page.is_file():
            failures.append(f'Missing API page for parameter types: {page.name}')
            continue
        parser = ParameterTypes()
        parser.feed(page.read_text())
        missing = parameters - parser.typed
        if missing:
            failures.append(f'Missing parameter-description types in {page.name}: {sorted(missing)}')
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f'Verified annotation-derived parameter types on {len(required)} API pages')


_DIRECTIVE_MODES = {
    '.. code-block:: text': 'stored console transcript; not executed',
    '.. catalog-help::': 'local catalog help; rendered without online queries',
    '.. tutorial-notebook::': 'generated RST companion; execute with development/run_tutorials.py',
    '.. tutorial-data::': 'generated local FITS download; no online queries',
    '.. plot::': 'Sphinx plot execution',
    '.. mermaid::': 'browser-rendered Mermaid diagram; review in a browser',
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
    from tutorial_notebooks import WORKFLOWS, SOURCE_PAGES, build_notebook

    cells = 0
    for page, name in WORKFLOWS:
        notebook = build_notebook(docs / 'tutorials' / f'{SOURCE_PAGES[page]}.rst')
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
        check_pages(args.html)
        check_parameter_types(args.html)
    if args.inventory:
        example_inventory()
    if args.notebooks:
        check_notebooks()
    if not args.html and not args.inventory and not args.notebooks:
        parser.error('Choose --html, --inventory or --notebooks')
