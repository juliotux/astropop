"""Generate shared offline tutorial observations, reference products and figures."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import Table

if TYPE_CHECKING:
    from sphinx.application import Sphinx


def generate_dataset(directory: Path) -> None:
    """Write seeded raw FITS exposures, source truth and reduced reference data.

    Counts are in ADU with a gain of one electron per ADU. Gaussian read noise
    is added to Poisson photoelectron counts. This is a teaching camera model,
    without dark current, saturation, overscan or instrumental photometric effects.
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
                              'FILTER': 'R'})
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


def plot_calibrations() -> None:
    """Show the bias structure and normalized detector response used in reduction."""
    import matplotlib.pyplot as plt

    with TemporaryDirectory() as temporary:
        data = Path(temporary)
        generate_dataset(data)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for axis, name, title in zip(axes, ['master_bias', 'master_flat'],
                                     ['Master bias (ADU)', 'Normalized master flat']):
            pixels = fits.getdata(data / 'reference' / f'{name}.fits', memmap=False)
            image = axis.imshow(pixels, origin='lower', cmap='viridis')
            axis.set(title=title, xlabel='x (pixel)', ylabel='y (pixel)')
            fig.colorbar(image, ax=axis, shrink=0.8)
        fig.tight_layout()


def plot_reduction() -> None:
    """Compare the same sky field in raw, calibrated and combined exposures."""
    import matplotlib.pyplot as plt
    from astropy.visualization import simple_norm

    with TemporaryDirectory() as temporary:
        data = Path(temporary)
        generate_dataset(data)
        names = ['raw/science_00.fits', 'reference/calibrated_first.fits', 'reference/combined.fits']
        titles = ['Raw exposure', 'Bias and flat corrected', 'Registered, clipped mean']
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        pixels = [fits.getdata(data / name, memmap=False) for name in names]
        norm = simple_norm(pixels[2], 'asinh', percent=99.5)
        for axis, image, title in zip(axes, pixels, titles):
            axis.imshow(image, origin='lower', norm=norm, cmap='gray')
            axis.set(title=title, xlabel='x (pixel)', ylabel='y (pixel)')
        fig.tight_layout()


def plot_photometry() -> None:
    """Show the measured field with apertures and its estimated sky background."""
    import matplotlib.pyplot as plt
    from astropy.visualization import simple_norm
    from matplotlib.patches import Circle

    with TemporaryDirectory() as temporary:
        data = Path(temporary)
        generate_dataset(data)
        measured, sky, rms = measure_dataset(data)
        pixels = fits.getdata(data / 'reference' / 'combined.fits', memmap=False)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].imshow(pixels, origin='lower', cmap='gray', norm=simple_norm(pixels, 'asinh', percent=99.5))
        for row in measured:
            for radius, color in [(6, 'cyan'), (9, 'orange'), (13, 'orange')]:
                axes[0].add_patch(Circle((row['x'], row['y']), radius, fill=False, color=color, lw=0.7))
        axes[0].set_title('Apertures and sky annuli')
        image = axes[1].imshow(sky, origin='lower', cmap='viridis')
        axes[1].set_title('Estimated sky (ADU)')
        fig.colorbar(image, ax=axes[1], shrink=0.8)
        truth = Table.read(data / 'sources.ecsv')
        distance2 = ((measured['x'][:, None] - truth['x'])**2
                     + (measured['y'][:, None] - truth['y'])**2)
        injected = truth['flux'][np.argmin(distance2, axis=1)]
        axes[2].errorbar(injected, 100 * (measured['flux'] / injected - 1),
                         yerr=100 * measured['flux_error'] / injected, fmt='o')
        axes[2].axhline(0, color='gray', linestyle='--')
        axes[2].set(title='Flux recovery (simulation check)', xlabel='Injected flux (ADU)',
                    ylabel='Flux difference (%)')
        for axis in axes[:2]:
            axis.set(xlabel='x (pixel)', ylabel='y (pixel)')
        fig.tight_layout()


def build_download(app: Sphinx) -> None:
    """Prepare the shared downloadable dataset inside the build directory."""
    folder = Path(app.doctreedir) / 'tutorial_assets'
    data = folder / 'data'
    if data.exists():
        shutil.rmtree(data)
    generate_dataset(data)
    shutil.make_archive(str(folder / 'ccd_tutorial'), 'zip', root_dir=folder, base_dir='data')


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
            link = download_reference('', '', nodes.Text('Download the synthetic FITS dataset (ZIP)'),
                                      reftarget=target)
            return [nodes.paragraph('', '', link)]

    app.connect('builder-inited', build_download)
    app.add_directive('tutorial-data', TutorialData)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    generate_dataset(parser.parse_args().directory)
