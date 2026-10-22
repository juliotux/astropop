"""Generate shared offline tutorial observations, reference products and figures."""

from __future__ import annotations

from datetime import datetime, timedelta
import os
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import Table

if TYPE_CHECKING:
    from sphinx.application import Sphinx


def generate_dataset(directory: Path, include_extras: bool = True) -> None:
    """Write seeded raw FITS exposures, source truth and reduced reference data.

    Counts are in ADU with a gain of one electron per ADU. Gaussian read noise
    is added to Poisson photoelectron counts. This is a teaching camera model,
    without dark current, saturation, overscan or instrumental photometric effects.

    Parameters
    ----------
    directory
        Dedicated output directory for the generated observations and references.
        Existing generated filenames are overwritten.
    include_extras
        Add the timestamped variable-star sequence, local standards, approximate
        pointing and solved reference image. Defaults to True; short CCD-only
        examples omit these additional inputs.
    """
    from astropop.framedata import FrameData

    directory.mkdir(parents=True, exist_ok=True)
    raw = directory / 'raw'
    raw.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    y, x = np.indices((256, 256))
    bias = 100 + 0.02 * x + 0.8 * np.sin(y / 11)
    response = 0.85 + 0.3 * x / 255
    response *= 1 - 0.08 * np.exp(-((x - 170)**2 + (y - 90)**2) / (2 * 18**2))
    response /= np.median(response)
    sky = 160 + 0.3 * x + 0.2 * y
    sx, sy = np.meshgrid([48., 128., 208.], [48., 128., 208.])
    truth = Table({'x': sx.ravel() + rng.uniform(-10, 10, 9),
                   'y': sy.ravel() + rng.uniform(-10, 10, 9),
                   'flux': [18000., 28000., 45000., 22000., 60000.,
                            35000., 25000., 40000., 30000.]})
    truth.write(directory / 'sources.ecsv', format='ascii.ecsv', overwrite=True)

    def write(name: str, pixels: np.ndarray, kind: str, exposure: float) -> None:
        """Store a raw exposure with detector and observation metadata."""
        header = fits.Header({'BUNIT': 'adu', 'IMAGETYP': kind,
                              'EXPTIME': exposure, 'GAIN': 1., 'RDNOISE': 4.,
                              'FILTER': 'R', 'DATE-OBS': '2025-01-01T00:00:00'})
        index = int(name.rsplit('_', 1)[1].split('.')[0])
        header['DATE-OBS'] = (datetime(2025, 1, 1) + timedelta(seconds=60 * index)).isoformat()
        header['TIMESYS'] = 'UTC'
        fits.writeto(raw / name, pixels.astype('float32'), header, overwrite=True)

    for i in range(8):
        write(f'bias_{i:02d}.fits', bias + rng.normal(0, 4, bias.shape), 'BIAS', 0)
    for i, level in enumerate([9000., 10000., 11000., 12000., 9500., 10500.]):
        flat = bias + rng.poisson(level * response) + rng.normal(0, 4, bias.shape)
        write(f'flat_{i:02d}.fits', flat, 'FLAT', 2)
    for i in range(6):
        signal = sky.copy()
        for source in truth:
            radius2 = (x - source['x'] - 0.4 * i)**2 + (y - source['y'] + 0.25 * i)**2
            signal += source['flux'] / (2 * np.pi * 1.8**2) * np.exp(-radius2 / (2 * 1.8**2))
        pixels = bias + rng.poisson(signal * response) + rng.normal(0, 4, bias.shape)
        pixels[20, 50 + 30 * i] += 8000  # isolated events outside the registration window
        write(f'science_{i:02d}.fits', pixels, 'LIGHT', 30)
    FrameData(bias, unit='adu').write(directory / 'true_bias.fits', overwrite=True)
    FrameData(response).write(directory / 'true_flat.fits', overwrite=True)
    reduce_dataset(directory, directory / 'reference')
    if include_extras:
        from workflow_data import extend_dataset
        extend_dataset(directory)


def reduce_dataset(data: Path, output: Path) -> None:
    """Create the reference products with the same steps shown in the CCD guide."""
    from astropop.framedata import read_framedata
    from astropop.image import imarith, imcombine
    from astropop.image.register import register_framedata_list
    from astropop.image.processing import subtract_bias, flat_correct

    output.mkdir(exist_ok=True)
    master_bias = imcombine(sorted((data / 'raw').glob('bias_*.fits')), method='median', dtype='float32')
    flats = []
    for path in sorted((data / 'raw').glob('flat_*.fits')):
        flat = subtract_bias(read_framedata(path), master_bias)
        flats.append(imarith(flat, np.median(flat.data) * u.adu, '/'))
    master_flat = imcombine(flats, method='median', dtype='float32')
    calibrated = []
    for path in sorted((data / 'raw').glob('science_*.fits')):
        frame = read_framedata(path)
        frame.uncertainty = np.sqrt(np.maximum(frame.data - master_bias.data, 0) + frame.meta['RDNOISE']**2)
        calibrated.append(flat_correct(subtract_bias(frame, master_bias), master_flat))
    aligned = register_framedata_list(calibrated, ref_image=0, upsample_factor=20,
                                     window=(slice(32, 224), slice(32, 224)))
    combined = imcombine(aligned, method='mean', sigma_clip=4,
                         sigma_dev_func='mad_std', dtype='float32')
    combined.meta['EXPTIME'] = 30.
    for index, frame in enumerate(calibrated):
        frame.write(output / f'calibrated_{index:02d}.fits', overwrite=True)
    for name, image in [('master_bias', master_bias), ('master_flat', master_flat),
                        ('calibrated_first', calibrated[0]), ('combined', combined)]:
        image.write(output / f'{name}.fits', overwrite=True)


def measure_dataset(data: Path) -> tuple[Table, np.ndarray, np.ndarray]:
    """Detect stars and measure local-sky-subtracted aperture fluxes."""
    from astropop.framedata import read_framedata
    from astropop.photometry import background, daofind, aperture_photometry

    image = read_framedata(data / 'reference' / 'combined.fits')
    sky, rms = background(image.data, box_size=32, filter_size=3,
                          mask=image.mask, global_bkg=False)
    sources = daofind(image.data, threshold=5, background=sky, noise=rms,
                      fwhm=4.24, mask=image.mask)
    measured = aperture_photometry(image.data, sources['x'], sources['y'],
                                   r=6, r_ann=(9, 13), bkg_error=rms, gain=1,
                                   mask=image.mask, pixel_flags=image.flags)
    return measured, sky, rms


def build_bundle(folder: Path) -> Path:
    """Write generated inputs and companions into a dedicated bundle directory."""
    folder.mkdir(parents=True, exist_ok=True)
    data = folder / 'data'
    if data.exists():
        shutil.rmtree(data)
    generate_dataset(data)
    from tutorial_notebooks import build_notebooks, WORKFLOWS, SOURCE_PAGES

    source = Path(__file__).resolve().parents[1] / 'tutorials'
    build_notebooks(source, folder / 'notebooks')
    originals = folder / 'tutorial_sources'
    originals.mkdir(exist_ok=True)
    for page, _ in WORKFLOWS:
        relative = Path(SOURCE_PAGES[page] + '.rst')
        destination = originals / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, destination)
    helpers = ['tutorial_data.py', 'workflow_data.py', 'workflow_figures.py',
               'guide_figures.py', 'tutorial_notebooks.py']
    for name in helpers:
        shutil.copyfile(Path(__file__).with_name(name), folder / name)
    (folder / 'README.txt').write_text(
        'Install astropop with its docs extra, then open notebooks/ in Jupyter.\n'
        'Run from this extracted bundle; notebook setup locates data/ automatically.\n'
        'Each numbered notebook is independent, using data/reference if earlier products\n'
        'are absent. Running in order uses your preceding products in reduced/.\n'
        'Tutorials overwrite named derived products, never the raw files.\n'
        'Data and helper sources are synthetic, seeded and work offline.\n'
        'Only the astrometry.net alternative requires an external solver/index.\n')
    with ZipFile(folder / 'ccd_tutorial.zip', 'w', compression=ZIP_DEFLATED) as archive:
        for directory in ['data', 'notebooks', 'tutorial_sources']:
            for path in sorted((folder / directory).rglob('*')):
                if path.is_file():
                    archive.write(path, path.relative_to(folder))
        for name in helpers + ['README.txt']:
            archive.write(folder / name, name)
    return folder / 'ccd_tutorial.zip'


def build_download(app: Sphinx) -> None:
    """Prepare the shared downloadable dataset inside the build directory."""
    build_bundle(Path(app.doctreedir) / 'tutorial_assets')


def setup(app: Sphinx) -> dict[str, bool]:
    """Register offline dataset generation and its download link directive."""
    from docutils import nodes
    from sphinx.addnodes import download_reference
    from sphinx.util.docutils import SphinxDirective

    class TutorialData(SphinxDirective):
        """Link the archive generated for both tutorial pages."""

        def run(self) -> list[nodes.Node]:
            """Return a standard Sphinx download link with build-relative storage."""
            archive = Path(self.env.doctreedir) / 'tutorial_assets' / 'ccd_tutorial.zip'
            self.env.note_dependency(__file__)
            target = os.path.relpath(archive, Path(self.env.doc2path(self.env.docname)).parent)
            link = download_reference('', '', nodes.Text('Download the shared tutorial data and notebooks (ZIP)'),
                                      reftarget=target)
            return [nodes.paragraph('', '', link)]

    class TutorialNotebook(SphinxDirective):
        """Link a generated notebook corresponding to the maintained RST source."""

        required_arguments = 1

        def run(self) -> list[nodes.Node]:
            """Resolve the workflow name and expose its notebook as a download."""
            from tutorial_notebooks import WORKFLOWS

            names = dict(WORKFLOWS)
            page = self.arguments[0]
            if page not in names:
                raise self.error(f'Unknown tutorial notebook: {page}')
            notebook = (Path(self.env.doctreedir) / 'tutorial_assets' / 'notebooks'
                        / f'{names[page]}.ipynb')
            self.env.note_dependency(str(Path(__file__).with_name('tutorial_notebooks.py')))
            target = os.path.relpath(notebook, Path(self.env.doc2path(self.env.docname)).parent)
            link = download_reference('', '', nodes.Text('Download this tutorial notebook'),
                                      reftarget=target)
            return [nodes.paragraph('', '', link)]

    app.connect('builder-inited', build_download)
    app.add_directive('tutorial-data', TutorialData)
    app.add_directive('tutorial-notebook', TutorialNotebook)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    generate_dataset(parser.parse_args().directory)
