"""Generate executable tutorial companions from the maintained RST workflows."""

from __future__ import annotations

import contextlib
import doctest
import io
from pathlib import Path
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nbformat import NotebookNode

WORKFLOWS = [
    ('observations', '01_observations'),
    ('master_bias', '02_master_bias'),
    ('master_flat', '03_master_flat'),
    ('science_reduction', '04_science_reduction'),
    ('photometry', '05_stacked_photometry'),
    ('light_curve', '06_light_curve'),
    ('polarimetry', '07_polarimetry'),
    ('astrometric_calibration', '08_astrometry'),
    ('photometric_calibration', '09_photometric_calibration'),
]

BOOTSTRAP = '''from pathlib import Path
import sys
import os
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')

bundle = Path.cwd()
if not (bundle / 'tutorial_data.py').exists():
    bundle = bundle.parent
os.chdir(bundle)
sys.path.insert(0, str(bundle))
from tutorial_data import generate_dataset
from astropop.config import AstropopConfig
AstropopConfig.FRAMEDATA_DISABLE_UNCERTAINTY = False
AstropopConfig.FRAMEDATA_DISABLE_FLAGS = False

data = bundle / 'data'
if not data.exists():
    generate_dataset(data)
'''


def visible_source(source: str) -> str:
    """Remove hidden fixture setup and cleanup, retaining user-facing calculations."""
    return re.sub(r'^\.\. doctest::\n(?:[ \t].*\n|\n)*', '', source, flags=re.M)


def execute_workflow(page: str, data: Path) -> dict[str, object]:
    """Replay a guide's visible examples to prepare the corresponding Sphinx figure."""
    sources = Path(__file__).resolve().parent / 'tutorial_sources'
    if not sources.exists():
        sources = Path(__file__).resolve().parents[1] / 'tutorials'
    text = visible_source((sources / f'{page}.rst').read_text())
    namespace = {'data': data, '__name__': '__tutorial__'}
    with contextlib.redirect_stdout(io.StringIO()):
        for example in doctest.DocTestParser().get_examples(text):
            exec(compile(example.source, f'{page}.rst', 'exec'), namespace)
    return namespace


def markdown(text: str) -> str:
    """Translate the limited inline RST vocabulary used in these tutorial pages."""
    def document_link(match: re.Match[str]) -> str:
        """Keep workflow links local to the generated notebook collection."""
        target = match[1]
        label = target.replace('_', ' ')
        if '<' in target:
            label, target = target.rstrip('>').split('<', 1)
        names = dict(WORKFLOWS)
        if target in names:
            return f'[{label.strip()}]({names[target]}.ipynb)'
        import posixpath
        path = posixpath.normpath('tutorials/' + target)
        return f'[{label.strip()}](https://astropop.readthedocs.io/en/latest/{path}.html)'

    text = re.sub(r':doc:`([^`]+)`', document_link, text)
    text = re.sub(r':(?:ref|download):`([^`]+)`', r'\1', text)
    text = re.sub(r':(?:class|func|mod):`~?([^`]+)`', r'`\1`', text)
    text = text.replace('``', '`')
    lines = text.splitlines()
    converted = []
    for i, line in enumerate(lines):
        if i + 1 < len(lines) and re.fullmatch(r'[=\-^]{3,}', lines[i + 1]):
            converted.append(('# ' if lines[i + 1][0] == '=' else '## ') + line)
        elif re.fullmatch(r'[=\-^]{3,}', line):
            continue
        else:
            converted.append(line[:-1] if line.endswith('::') else line)
    return '\n'.join(converted).strip()


def build_notebook(page: Path) -> NotebookNode:
    """Convert narrative, prompted examples and plots without duplicating calculations."""
    import nbformat as nbf

    text = visible_source(page.read_text())
    lines = text.splitlines()
    cells = [nbf.v4.new_code_cell(BOOTSTRAP, metadata={
        'tags': ['hide-input'], 'jupyter': {'source_hidden': True}})]
    narrative: list[str] = []

    def flush() -> None:
        """Emit accumulated explanatory text as one Markdown cell."""
        if value := markdown('\n'.join(narrative)):
            cells.append(nbf.v4.new_markdown_cell(value))
        narrative.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(('.. tutorial-data::', '.. tutorial-notebook::', '.. include::')):
            i += 1
            continue
        if line.lstrip().startswith('>>>'):
            flush()
            block = []
            while i < len(lines) and (lines[i].startswith('    ') or lines[i] == ''):
                block.append(lines[i])
                i += 1
            examples = doctest.DocTestParser().get_examples('\n'.join(block))
            code = '\n'.join(example.source.rstrip() for example in examples)
            expected = ''.join(example.want for example in examples).replace('<BLANKLINE>', '')
            cells.append(nbf.v4.new_code_cell(code))
            if expected:
                cells.append(nbf.v4.new_markdown_cell('Expected output:\n\n```text\n' + expected.rstrip() + '\n```'))
            continue
        if line.startswith('.. plot::'):
            flush()
            args = line.partition('::')[2].strip().split()
            options: dict[str, str] = {}
            body = []
            i += 1
            while i < len(lines) and (lines[i].startswith('    ') or lines[i] == ''):
                part = lines[i][4:] if lines[i].startswith('    ') else ''
                if part.startswith(':'):
                    key, _, value = part[1:].partition(':')
                    options[key] = value.strip()
                else:
                    body.append(part)
                i += 1
            if args:
                module = Path(args[0]).stem
                function = args[1]
                if module == 'workflow_figures':
                    from workflow_figures import NOTEBOOK_PLOTS
                    call = NOTEBOOK_PLOTS[function]
                    code = f'from workflow_figures import {call.partition("(")[0]}\n{call}'
                elif function == 'plot_polarimetry':
                    code = 'from guide_figures import plot_polarimetry\nplot_polarimetry(psi, result)'
                else:
                    code = f'from {module} import {function}\n{function}()'
            else:
                code = '\n'.join(body).strip()
            code += '\nimport matplotlib.pyplot as plt\nplt.show()'
            cells.append(nbf.v4.new_code_cell(code, metadata={
                'tags': ['hide-input'], 'jupyter': {'source_hidden': True}}))
            if options.get('caption'):
                cells.append(nbf.v4.new_markdown_cell(options['caption']))
            continue
        if line.startswith('.. code-block::'):
            language = line.partition('::')[2].strip()
            block = []
            i += 1
            while i < len(lines) and (lines[i].startswith('    ') or lines[i] == ''):
                block.append(lines[i][4:] if lines[i].startswith('    ') else '')
                i += 1
            if '\n'.join(block).strip() == "data = Path('data')":
                narrative.append('The hidden setup cell locates the bundled data automatically.')
            else:
                narrative.extend(['```' + language, '\n'.join(block).strip(), '```', ''])
            continue
        narrative.append(line)
        i += 1
    flush()
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
        'language_info': {'name': 'python'}, 'astropop_source': page.name})
    return notebook


def build_notebooks(source: Path, output: Path) -> list[Path]:
    """Write the nine generated notebooks, leaving their numerical source in RST."""
    import nbformat

    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for page, name in WORKFLOWS:
        notebook = build_notebook(source / f'{page}.rst')
        path = output / f'{name}.ipynb'
        nbformat.write(notebook, path)
        paths.append(path)
    return paths
