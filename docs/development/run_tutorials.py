"""Generate and execute the offline tutorials both in sequence and independently."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
from typing import Literal


def run_tutorials(output: Path, kernel: str = 'python3',
                  mode: Literal['sequence', 'independent', 'both'] = 'both') -> None:
    """Build a bundle and retain notebook executions with fresh kernels in each mode."""
    import nbformat
    from nbclient import NotebookClient

    docs = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(docs.parent))
    sys.path.insert(0, str(docs / '_ext'))
    from tutorial_data import build_bundle

    output = output.resolve()
    bundle = output / 'tutorial_assets'
    build_bundle(bundle)
    modes = ['sequence', 'independent'] if mode == 'both' else [mode]
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(docs.parent) + os.pathsep + environment.get('PYTHONPATH', '')
    for execution_mode in modes:
        destination = output / 'executed' / execution_mode
        destination.mkdir(parents=True, exist_ok=True)
        for path in sorted((bundle / 'notebooks').glob('*.ipynb')):
            working = bundle
            if execution_mode == 'independent':
                working = output / 'independent_inputs' / path.stem
                if working.exists():
                    shutil.rmtree(working)
                working.mkdir(parents=True)
                shutil.copytree(bundle / 'data', working / 'data')
                for helper in bundle.glob('*.py'):
                    shutil.copyfile(helper, working / helper.name)
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            client = NotebookClient(notebook, timeout=180, kernel_name=kernel,
                                     resources={'metadata': {'path': str(working)}})
            client.execute(env=environment)
            nbformat.write(notebook, destination / path.name)
            figures = sum('image/png' in item.get('data', {})
                          for cell in notebook.cells for item in cell.get('outputs', []))
            if not figures:
                raise RuntimeError(f'No rendered figure in {path.name}')
            code_cells = sum(cell.cell_type == 'code' for cell in notebook.cells)
            print(f'{execution_mode}: {path.name}: executed {code_cells} code cells, '
                  f'{figures} figures', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='Dedicated directory for generated data and executed notebooks')
    parser.add_argument('--kernel', default='python3', help='Jupyter kernel in the docs environment')
    parser.add_argument('--mode', choices=['sequence', 'independent', 'both'], default='both')
    args = parser.parse_args()
    run_tutorials(args.output, args.kernel, args.mode)
