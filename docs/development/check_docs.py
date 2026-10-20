"""Inspect documentation examples and check retained HTML URLs.

Run from the repository root; this script does not execute examples or contact
services. Use pytest and Sphinx for execution and rendering respectively.
"""

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path


class Anchors(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        self.ids.update(value for key, value in attrs if key == 'id')


def check_urls(html):
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
            elif stripped.startswith('.. ipython::'):
                options = lines[index + 1:index + 5]
                mode = ('stored transcript' if any(':verbatim:' in x for x in options)
                        else 'Sphinx IPython execution')
            elif stripped.startswith('.. plot::'):
                mode = 'Sphinx plot execution'
            elif stripped.startswith('.. code-block:: python'):
                mode = 'static recipe; not executed'
            if mode:
                examples.append({'page': str(path.relative_to(docs)),
                                 'line': index + 1, 'validation': mode})
    for path in sorted((docs / 'ipynb').glob('*.ipynb')):
        examples.append({'page': str(path.relative_to(docs)),
                         'validation': 'stored notebook; external inputs; not executed'})
    print(json.dumps(examples, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--html', type=Path, help='Sphinx HTML output directory')
    parser.add_argument('--inventory', action='store_true', help='List validation modes')
    args = parser.parse_args()
    if args.html:
        check_urls(args.html)
    if args.inventory:
        example_inventory()
    if not args.html and not args.inventory:
        parser.error('Choose --html or --inventory')
