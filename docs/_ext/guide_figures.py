"""Draw offline diagnostics using the small examples in the authored guides."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from astropop.polarimetry import StokesParameters
    from sphinx.application import Sphinx


def plot_polarimetry(psi: np.ndarray | None = None,
                     result: StokesParameters | None = None) -> Figure:
    """Plot the supplied retarder modulation and fitted residuals."""
    from astropy import units as u
    from astropop.polarimetry import SLSDualBeamPolarimetry

    if psi is None or result is None:
        psi = np.arange(8) * 22.5
        z = 0.03 * np.cos(np.deg2rad(4 * psi)) - 0.02 * np.sin(np.deg2rad(4 * psi))
        ordinary = 10000 * (1 + z) / 2
        extraordinary = 10000 * (1 - z) / 2
        result = SLSDualBeamPolarimetry('halfwave', k=1., zero=0.).compute(
            psi, ordinary, extraordinary, np.sqrt(ordinary), np.sqrt(extraordinary))
    stop = max(180., 90 * np.ceil(np.max(psi) / 90))
    angles = np.linspace(0, stop, 601)
    kind = 'Half-wave' if result.retarder == 'halfwave' else 'Quarter-wave'
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True,
                             layout='constrained', height_ratios=[2, 1])
    axes[0].errorbar(psi, 100 * result.zi.nominal,
                     yerr=100 * result.zi.uncertainty, fmt='o', color='C0',
                     label='Beam measurements with supplied errors')
    axes[0].plot(angles, 100 * result.model(angles * u.deg).nominal, color='C1',
                 label=f'Fitted {kind.lower()} model')
    axes[0].set(ylabel='Normalized beam difference (%)', title=f'{kind} modulation')
    axes[0].legend(fontsize='small')
    residual = result.zi.nominal - result.model(psi * u.deg).nominal
    axes[1].errorbar(psi, 100 * residual, yerr=100 * result.zi.uncertainty,
                     fmt='o', color='C0')
    axes[1].axhline(0, color='0.3', linestyle='--')
    extent = max(0.1, 120 * np.max(np.abs(residual) + result.zi.uncertainty))
    axes[1].set(xlabel='Retarder angle (deg)', ylabel='Residual (%)',
                ylim=(-extent, extent), xticks=np.arange(0, stop + 1, 45))
    for ax in axes:
        ax.grid(alpha=0.2)
    return fig


def plot_catalog_matching() -> Figure:
    """Show spherical offsets and the tutorial's one-arcsecond match radius."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from matplotlib.patches import Circle
    from astropop.catalogs import SourcesCatalog

    catalog = SourcesCatalog(ids=['A', 'B'], ra=[10., 11.], dec=[20., 21.], unit='deg')
    origins = SkyCoord([11., 10., 10.], [21., 20., 20.], unit='deg')
    targets = origins.spherical_offsets_by([0.2, -0.4, 2.] * u.arcsec,
                                           [0.1, 0.2, -0.5] * u.arcsec)
    matched = catalog.match_objects(targets.ra.deg, targets.dec.deg, '1 arcsec')
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), layout='constrained',
                             sharex=True, sharey=True)
    for ax, index, rows in zip(axes, [0, 1], [[1, 2], [0]]):
        center = catalog.get_coordinates()[index]
        east, north = center.spherical_offsets_to(targets[rows])
        ax.add_patch(Circle((0, 0), 1, fill=False, linestyle='--', color='0.4'))
        ax.plot(0, 0, '*', color='black', markersize=12, label='Catalog position')
        for row, dx, dy in zip(rows, east.arcsec, north.arcsec):
            accepted = matched.sources_id()[row] != ''
            ax.plot(dx, dy, 'o' if accepted else 'x', color='C0' if accepted else 'C3',
                     markersize=8)
            ax.annotate(f'Input {row + 1}', (dx, dy), xytext=(5, 7),
                         textcoords='offset points', fontsize='small')
        ax.set(title=f'Neighborhood of {catalog.sources_id()[index]}',
                xlabel='East offset (arcsec)', xlim=(-1.5, 2.7), ylim=(-1.4, 1.4),
                aspect='equal')
        ax.grid(alpha=0.2)
    axes[0].set_ylabel('North offset (arcsec)')
    axes[0].legend(loc='lower right', fontsize='small')
    return fig


def plot_combination() -> Figure:
    """Compare actual mean, median and clipped-mean results on a seeded bias stack."""
    from matplotlib.ticker import ScalarFormatter
    from astropop.framedata import FrameData
    from astropop.image import imcombine

    rng = np.random.default_rng(42)
    pixels = rng.normal(1000, 3, (9, 48, 48))
    pixels[0, 24, 24] += 600
    frames = [FrameData(image, unit='adu') for image in pixels]
    results = {
        'Mean': imcombine(frames, method='mean'),
        'Median': imcombine(frames, method='median'),
        'Clipped mean': imcombine(frames, method='mean', sigma_clip=4,
                                   sigma_dev_func='mad_std'),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout='constrained')
    axes[0].plot(np.arange(1, 10), pixels[:, 24, 24], 'ko', label='Input samples')
    for index, (name, image) in enumerate(results.items()):
        value = image.data[24, 24]
        axes[0].axhline(value, color=f'C{index}', linestyle=['-', '--', ':'][index],
                        label=f'{name}: {value:.1f} ADU')
        axes[1].hist((image.data - 1000).ravel(), bins=np.arange(-8, 81),
                     histtype='step', color=f'C{index}', label=name, linewidth=1.5)
    axes[0].axhline(1000, color='0.5', linestyle='-.', label='Injected level: 1000 ADU')
    axes[0].set(xlabel='Exposure number', ylabel='Pixel value (ADU)',
                title='One pixel with a transient event', xticks=[1, 3, 5, 7, 9])
    axes[0].legend(fontsize='small')
    axes[1].axvline(0, color='0.5', linestyle='-.')
    axes[1].set_xscale('symlog', linthresh=4)
    axes[1].set_xticks([-8, -4, 0, 4, 20, 60])
    axes[1].xaxis.set_major_formatter(ScalarFormatter())
    axes[1].set(xlabel='Residual from 1000 ADU (compressed tails)',
                ylabel='Number of pixels', yscale='log', ylim=(0.7, 3000),
                title='Residuals across the combined image')
    axes[1].legend(fontsize='small')
    for ax in axes:
        ax.grid(alpha=0.2)
    return fig


def plot_registration() -> Figure:
    """Compare central and edge residuals for the guide's conflicting translations."""
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    from astropop.framedata import FrameData
    from astropop.image.register import register_framedata_list

    y, x = np.indices((128, 128))
    positions = np.array([[55., 50.], [75., 70.], [10., 10.]])
    drift = np.array([[-2., -1.], [0., 0.], [2., 1.]])
    frames = []
    for dx, dy in drift:
        image = np.zeros(x.shape)
        for sx, sy in positions[:2] + [dx, dy]:
            image += 100 * np.exp(-((x - sx)**2 + (y - sy)**2) / 2)
        image += 1e6 * np.exp(-((x - 10)**2 + (y - 10)**2) / 2)
        frames.append(FrameData(image))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), layout='constrained')
    axes[0].imshow(np.maximum(frames[1].data, 1), origin='lower', cmap='magma',
                    norm=LogNorm(vmin=1, vmax=1e6))
    axes[0].add_patch(Rectangle((34.5, 34.5), 55, 55, fill=False,
                                edgecolor='cyan', linewidth=1.5))
    axes[0].set(title='Middle reference and window', xlabel='x (pixel)', ylabel='y (pixel)')
    observed = np.broadcast_to(positions, (3, 3, 2)).copy()
    observed[:, :2] += drift[:, None, :]
    for name, window, color in [('Full image', None, 'C0'),
                                ('Central window', (slice(35, 90), slice(35, 90)), 'C1')]:
        registered = register_framedata_list(frames, ref_image=1, window=window)
        shifts = np.array([[frame.meta['astropop registration_shift_x'],
                            frame.meta['astropop registration_shift_y']]
                           for frame in registered])
        residual = np.linalg.norm(observed - shifts[:, None, :] - positions, axis=2)
        for ax, values in zip(axes[1:], [residual[:, :2].max(axis=1), residual[:, 2]]):
            ax.plot(np.arange(3), values, 'o-', color=color, label=name)
    for ax, title in zip(axes[1:], ['Largest central-star residual', 'Edge-feature residual']):
        ax.set(title=title, xlabel='Frame index (reference = 1)',
                ylabel='Position residual (pixel)', xticks=[0, 1, 2], ylim=(-0.1, 2.6))
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize='small')
    return fig


def setup(app: Sphinx) -> dict[str, bool]:
    """Invalidate figure pages when their shared plotting source changes."""
    def track_source(app: Sphinx, docname: str, source: list[str]) -> None:
        """Record the source dependency before Sphinx parses a figure page."""
        if 'plot:: _ext/guide_figures.py' in source[0]:
            app.env.note_dependency(__file__)

    app.connect('source-read', track_source)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
