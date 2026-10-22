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
    ('quarterwave', '07b_quarterwave'),
    ('astrometric_calibration', '08_astrometry'),
    ('photometric_calibration', '09_photometric_calibration'),
]

SOURCE_PAGES = {
    'observations': 'ccd_reduction/observations',
    'master_bias': 'ccd_reduction/master_bias',
    'master_flat': 'ccd_reduction/master_flat',
    'science_reduction': 'ccd_reduction/science_reduction',
    'dark_correction': 'ccd_reduction/dark_correction',
    'photometry': 'photometry/photometry',
    'photometric_calibration': 'photometry/photometric_calibration',
    'light_curve': 'photometry/light_curve',
    'local_catalog_matching': 'astrometry_catalogs/local_catalog_matching',
    'astrometric_calibration': 'astrometry_catalogs/astrometric_calibration',
    'astrometry_net': 'astrometry_catalogs/astrometry_net',
    'online_catalogs': 'astrometry_catalogs/online_catalogs',
    'fits_collections': 'diagnostics/fits_collections',
    'quality_propagation': 'diagnostics/quality_propagation',
    'registration_diagnostics': 'diagnostics/registration_diagnostics',
    'polarimetry': 'polarimetry/halfwave',
    'quarterwave': 'polarimetry/quarterwave',
}

BOOTSTRAP = '''from pathlib import Path
import sys
import os
import matplotlib

matplotlib.use('module://matplotlib_inline.backend_inline')

# Accept launches from either the bundle root or its notebooks directory.
bundle = Path.cwd()
if not (bundle / 'tutorial_data.py').exists():
    bundle = bundle.parent
os.chdir(bundle)
sys.path.insert(0, str(bundle))
from tutorial_data import generate_dataset
from astropop.config import AstropopConfig

# Set allocation policy before the dataset helper reads or creates frames.
AstropopConfig.FRAMEDATA_DISABLE_UNCERTAINTY = False
AstropopConfig.FRAMEDATA_DISABLE_FLAGS = False

# Reuse existing inputs so rerunning a notebook does not regenerate them.
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
    text = visible_source((sources / f'{SOURCE_PAGES[page]}.rst').read_text())
    namespace = {'data': data, '__name__': '__tutorial__'}
    with contextlib.redirect_stdout(io.StringIO()):
        for example in doctest.DocTestParser().get_examples(text):
            exec(compile(example.source, f'{page}.rst', 'exec'), namespace)
    return namespace


def markdown(text: str, source: str = '') -> str:
    """Translate the limited inline RST vocabulary used in these tutorial pages."""
    def document_link(match: re.Match[str]) -> str:
        """Keep workflow links local to the generated notebook collection."""
        target = match[1]
        label = target.replace('_', ' ')
        if '<' in target:
            label, target = target.rstrip('>').split('<', 1)
        import posixpath
        path = posixpath.normpath(posixpath.join(
            'tutorials', posixpath.dirname(source), target))
        names = {'tutorials/' + SOURCE_PAGES[key]: name for key, name in WORKFLOWS}
        if path in names:
            return f'[{label.strip()}]({names[path]}.ipynb)'
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


def indented_block(lines: list[str], start: int) -> tuple[list[str], int]:
    """Read an indented RST body, including blank lines, and return its end."""
    end = start
    while end < len(lines) and (lines[end].startswith('    ') or lines[end] == ''):
        end += 1
    return lines[start:end], end


def example_code(block: list[str]) -> str:
    """Keep prompted code, comments and spacing without copying console output."""
    code = []
    in_input = False
    for line in block:
        prompt = re.match(r'^\s*(>>>|\.\.\.)(?: (.*))?$', line)
        if prompt and (prompt[1] == '>>>' or in_input):
            code.append(prompt[2] or '')
            in_input = True
        else:
            in_input = False
            if not line.strip():
                code.append('')
    return '\n'.join(code).strip('\n')


def example_cells(block: list[str]) -> list[NotebookNode]:
    """Convert prompted examples to code and a separate expected-output cell."""
    import nbformat as nbf

    examples = doctest.DocTestParser().get_examples('\n'.join(block))
    code = example_code(block)
    expected = ''.join(example.want for example in examples).replace('<BLANKLINE>', '')
    cells = [nbf.v4.new_code_cell(code)]
    if expected:
        cells.append(nbf.v4.new_markdown_cell('Expected output:\n\n```text\n' + expected.rstrip() + '\n```'))
    return cells


def plot_code(args: list[str], body: list[str]) -> str:
    """Resolve a plot directive to a call using the notebook's live variables."""
    if not args:
        return '\n'.join(body).strip()
    module = Path(args[0]).stem
    function = args[1]
    if module == 'workflow_figures':
        from workflow_figures import NOTEBOOK_PLOTS
        call = NOTEBOOK_PLOTS[function]
        return f'from workflow_figures import {call.partition("(")[0]}\n{call}'
    if function == 'plot_polarimetry':
        return 'from guide_figures import plot_polarimetry\nplot_polarimetry(psi, result)'
    return f'from {module} import {function}\n{function}()'


def plot_cells(directive: str, block: list[str]) -> list[NotebookNode]:
    """Convert a plot body and its options to hidden code and a visible caption."""
    import nbformat as nbf

    args = directive.partition('::')[2].strip().split()
    options: dict[str, str] = {}
    body = []
    for line in block:
        part = line[4:] if line.startswith('    ') else ''
        if part.startswith(':'):
            key, _, value = part[1:].partition(':')
            options[key] = value.strip()
        else:
            body.append(part)
    code = plot_code(args, body) + '\nimport matplotlib.pyplot as plt\nplt.show()'
    cells = [nbf.v4.new_code_cell(code, metadata={
        'tags': ['hide-input'], 'jupyter': {'source_hidden': True}})]
    if options.get('caption'):
        cells.append(nbf.v4.new_markdown_cell(options['caption']))
    return cells


def recipe_text(directive: str, block: list[str]) -> list[str]:
    """Render a static recipe, replacing the data path supplied by notebook setup."""
    language = directive.partition('::')[2].strip()
    text = '\n'.join(line[4:] if line.startswith('    ') else '' for line in block).strip()
    if text == "data = Path('data')":
        return ['The hidden setup cell locates the bundled data automatically.']
    return ['```' + language, text, '```', '']


def build_notebook(page: Path) -> NotebookNode:
    """Convert narrative, prompted examples and plots without duplicating calculations."""
    import nbformat as nbf

    source_path = next((v for v in SOURCE_PAGES.values()
                        if page.as_posix().endswith(v + '.rst')), page.stem)
    text = visible_source(page.read_text())
    lines = text.splitlines()
    cells = [nbf.v4.new_code_cell(BOOTSTRAP, metadata={
        'tags': ['hide-input'], 'jupyter': {'source_hidden': True}})]
    narrative: list[str] = []

    def flush() -> None:
        """Emit accumulated explanatory text as one Markdown cell."""
        if value := markdown('\n'.join(narrative), source_path):
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
            block, i = indented_block(lines, i)
            cells.extend(example_cells(block))
            continue
        if line.startswith('.. plot::'):
            flush()
            block, i = indented_block(lines, i + 1)
            cells.extend(plot_cells(line, block))
            continue
        if line.startswith('.. code-block::'):
            block, i = indented_block(lines, i + 1)
            narrative.extend(recipe_text(line, block))
            continue
        narrative.append(line)
        i += 1
    flush()
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
        'language_info': {'name': 'python'}, 'astropop_source': source_path + '.rst'})
    return notebook


def build_notebooks(source: Path, output: Path) -> list[Path]:
    """Write the generated notebooks, leaving their numerical source in RST."""
    import nbformat

    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for page, name in WORKFLOWS:
        notebook = build_notebook(source / f'{SOURCE_PAGES[page]}.rst')
        path = output / f'{name}.ipynb'
        nbformat.write(notebook, path)
        paths.append(path)
    return paths
