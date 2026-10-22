"""Draw deterministic diagnostics for the task guides using public operations."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from sphinx.application import Sphinx


def plot_arithmetic() -> Figure:
    """Show how independent bias and flat errors enter arithmetic results."""
    from astropop.framedata import FrameData
    from astropop.image import imarith

    x = np.arange(80)
    signal = 50 + 300 * np.exp(-0.5 * ((x - 42) / 6)**2)
    response = 0.8 + 0.4 * x / (len(x) - 1)
    raw = FrameData((100 + response * signal)[None, :], unit='adu', uncertainty=4.)
    bias = FrameData(np.full(raw.shape, 100.), unit='adu', uncertainty=2.)
    flat = FrameData(response[None, :], uncertainty=0.02)
    subtracted = imarith(raw, bias, '-')
    corrected = imarith(subtracted, flat, '/')
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 5.3),
                             layout='constrained')
    for frame, name, color in [(raw, 'Raw', '0.5'),
                                (subtracted, 'Bias subtracted', 'C0'),
                                (corrected, 'Flat corrected', 'C1')]:
        axes[0].plot(x, frame.data[0], color=color, label=name)
        axes[1].plot(x, frame.uncertainty[0], color=color)
    axes[0].plot(x, signal, 'k--', linewidth=1, label='Injected signal')
    axes[0].set(ylabel='Nominal value (ADU)', title='Arithmetic on one image row')
    axes[0].legend(fontsize='small')
    axes[1].set(xlabel='Column x (pixel)', ylabel='Standard uncertainty (ADU)',
                title='Supplied errors propagated as independent inputs')
    for ax in axes:
        ax.grid(alpha=0.2)
    return fig


def plot_quality_flags() -> Figure:
    """Follow the guide's two pixels through arithmetic, calibration and FITS."""
    from astropy import units as u
    from astropop.framedata import FrameData, PixelMaskFlags as F, read_framedata
    from astropop.image import imarith, imcombine
    from astropop.image.processing import subtract_bias

    raw = FrameData([[110., 120.]], unit='adu')
    raw.add_flags(F.MASKED | F.SATURATED, where=(0, 0))
    arithmetic = imarith(raw, 10 * u.adu, '-', merge_flags='or')
    cleared = subtract_bias(raw, FrameData([[10., 10.]], unit='adu'))
    restored = cleared.copy()
    restored.flags = raw.flags.copy()
    combined = imcombine([restored, FrameData([[90., 110.]], unit='adu')], method='mean')
    with TemporaryDirectory() as directory:
        filename = Path(directory) / 'quality.fits'
        restored.write(filename)
        imported = read_framedata(filename)
    stages = [(raw, 'Raw'), (arithmetic, "imarith: merge_flags='or'"),
              (cleared, 'subtract_bias: cleared flags'), (restored, 'Restore detector flags'),
              (combined, 'Combine with usable exposure'), (imported, 'FITS reload of restored frame')]
    fig, axes = plt.subplots(2, 3, figsize=(10, 4.6), layout='constrained')
    for ax, (frame, title) in zip(axes.flat, stages):
        ax.imshow(frame.data, cmap='Blues', vmin=70, vmax=150, origin='lower')
        for column in range(2):
            ax.text(column, 0, f'{frame.data[0, column]:.0f} ADU\nflags: {frame.flags[0, column]}',
                    ha='center', va='center', fontsize=10)
        masked = np.flatnonzero(frame.mask[0])
        ax.plot(masked, np.full(len(masked), 0.33), 'rx', markersize=9, markeredgewidth=2)
        ax.set(title=title, xticks=[0, 1], yticks=[], xlabel='Column x (pixel)')
    return fig



def plot_wcs() -> Figure:
    """Map a small detector asterism through a supplied TAN coordinate system."""
    from astropy.wcs import WCS

    camera = WCS(naxis=2)
    camera.wcs.crpix = [64.5, 64.5]
    camera.wcs.crval = [150., -30.]
    camera.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    angle = np.deg2rad(15)
    camera.wcs.cd = 0.34 / 3600 * np.array([[-np.cos(angle), np.sin(angle)],
                                           [np.sin(angle), np.cos(angle)]])
    x = np.array([20., 39., 65., 92., 110.])
    y = np.array([28., 96., 60., 35., 103.])
    center = camera.pixel_to_world(63.5, 63.5)
    east, north = center.spherical_offsets_to(camera.pixel_to_world(x, y))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
    grid = np.linspace(0, 127, 64)
    for fixed in np.linspace(0, 127, 5):
        for xx, yy in [(grid, np.full_like(grid, fixed)), (np.full_like(grid, fixed), grid)]:
            axes[0].plot(xx, yy, color='0.85', linewidth=1)
            dx, dy = center.spherical_offsets_to(camera.pixel_to_world(xx, yy))
            axes[1].plot(dx.arcsec, dy.arcsec, color='0.85', linewidth=1)
    for ax, xx, yy in [(axes[0], x, y), (axes[1], east.arcsec, north.arcsec)]:
        ax.plot(xx, yy, 'C0*', markersize=11)
        for name, sx, sy in zip('ABCDE', xx, yy):
            ax.annotate(name, (sx, sy), xytext=(5, 5), textcoords='offset points')
        ax.set_aspect('equal')
    axes[0].set(xlim=(-5, 132), ylim=(-5, 132), xlabel='x / column (pixel)',
                ylabel='y / row (pixel)', title='Detector coordinates; origin = 0')
    axes[1].set(xlim=(28, -28), ylim=(-28, 28), xlabel='East offset (arcsec)',
                ylabel='North offset (arcsec)', title='Sky offsets; east increases left')
    return fig


def plot_catalog_epoch() -> Figure:
    """Compare actual catalog matching with and without an observation epoch."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    from matplotlib.patches import Circle
    from astropop.catalogs import SourcesCatalog

    reference = SourcesCatalog(ids=['Moving star'], ra=[10.] * u.deg, dec=[20.] * u.deg,
                               pm_ra_cosdec=[1000.] * u.mas / u.yr,
                               pm_dec=[200.] * u.mas / u.yr, distance=[100.] * u.pc,
                               radial_velocity=[0.] * u.km / u.s,
                               obstime=Time(2015., format='jyear'))
    years = np.arange(2015, 2026)
    predicted = reference.get_coordinates(Time(years, format='jyear'))
    # Plot only positions; each propagated coordinate has a different epoch.
    track = SkyCoord(ra=predicted.ra, dec=predicted.dec)
    start = track[0]
    observed = track[-1].spherical_offsets_by(0.15 * u.arcsec, -0.1 * u.arcsec)
    east, north = start.spherical_offsets_to(track)
    dx, dy = start.spherical_offsets_to(observed)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), layout='constrained')
    axes[0].plot(east.arcsec, north.arcsec, '.-', color='0.6', label='Catalog motion model')
    for index, color, year in [(0, 'C0', 2015), (-1, 'C1', 2025)]:
        axes[0].add_patch(Circle((east.arcsec[index], north.arcsec[index]), 1,
                                 fill=False, color=color, label=f'1 arcsec at {year}'))
    axes[0].plot(dx.arcsec, dy.arcsec, 'k*', markersize=12, label='Observed in 2025')
    axes[0].set(aspect='equal', xlim=(-1.5, 12), ylim=(-2, 5),
                xlabel='East offset (arcsec)', ylabel='North offset (arcsec)',
                title='The same source changes position')
    axes[0].legend(fontsize='small', loc='lower right')
    axes[1].plot(years, observed.separation(track).arcsec, 'ko-')
    axes[1].axhline(1, color='C3', linestyle='--', label='Match radius: 1 arcsec')
    for year in [None, 2025]:
        epoch = None if year is None else Time(year, format='jyear')
        matched = reference.match_objects([observed.ra.deg], [observed.dec.deg],
                                           '1 arcsec', obstime=epoch)
        accepted = matched.sources_id()[0] != ''
        label = f'{"No epoch" if year is None else year}: {"matched" if accepted else "unmatched"}'
        axes[1].plot([], [], ' ', label=label)
    axes[1].set(xlabel='Epoch used for catalog coordinates (year)',
                ylabel='Separation from observation (arcsec)', title='Propagate before matching',
                xticks=[2015, 2020, 2025])
    axes[1].legend(fontsize='small')
    axes[1].grid(alpha=0.2)
    return fig


def plot_tutorial_paths() -> Figure:
    """Separate the stack, time-series and paired-beam tutorial dependencies."""
    fig, ax = plt.subplots(figsize=(12, 4.8), layout='constrained')
    nodes = {
        'bias': (0, 3.2, 'Master\nbias', 'C0'),
        'flat': (2, 3.2, 'Master\nflat', 'C0'),
        'stack': (4, 3.2, 'Calibrate +\nregister + stack', 'C0'),
        'measure': (6, 3.2, 'Stack\nphotometry', 'C0'),
        'wcs': (8, 3.2, 'Fit\nastrometry', 'C0'),
        'magnitude': (10, 3.2, 'Calibrate\nmagnitudes', 'C0'),
        'series': (4, 1.6, 'Calibrate separate\ntime-series images', 'C1'),
        'curve': (6, 1.6, 'Differential\nlight curve', 'C1'),
        'beams': (4, 0, 'Paired beam\nmeasurements', 'C2'),
        'stokes': (6, 0, 'Fit Stokes\nparameters', 'C2'),
    }
    for x, y, label, color in nodes.values():
        ax.text(x, y, label, ha='center', va='center', fontsize=10,
                bbox={'boxstyle': 'round,pad=0.45', 'fc': 'white', 'ec': color})
    edges = [('bias', 'flat'), ('flat', 'stack'), ('stack', 'measure'),
             ('measure', 'wcs'), ('wcs', 'magnitude'), ('flat', 'series'),
             ('series', 'curve'), ('beams', 'stokes')]
    for start, end in edges:
        ax.annotate('', xy=nodes[end][:2], xytext=nodes[start][:2],
                    arrowprops={'arrowstyle': '->', 'shrinkA': 50, 'shrinkB': 50,
                                'color': '0.4'})
    ax.annotate('', xy=nodes['stack'][:2], xytext=nodes['bias'][:2],
                arrowprops={'arrowstyle': '->', 'shrinkA': 34, 'shrinkB': 48,
                            'connectionstyle': 'arc3,rad=-0.35', 'color': '0.4'})
    ax.text(0, 1.6, 'Separate raw sequence;\nsame bias and flat masters',
            va='center', fontsize=10, color='C1')
    ax.text(0, 0, 'Separate polarimetric data;\nno input from the imaging stack',
            va='center', fontsize=10, color='C2')
    ax.set(xlim=(-1, 11), ylim=(-0.6, 4.2))
    ax.axis('off')
    return fig


def plot_pixel_coordinates() -> Figure:
    """Show zero/FITS pixel centers and the local origin of a nonsquare crop."""
    from matplotlib.patches import Rectangle

    image = np.arange(15).reshape(3, 5)
    crop = image[1:3, 2:5]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout='constrained')
    panels = [(image, 0, 'NumPy centers: data[y, x]'),
              (image, 1, 'FITS centers: CRPIX convention'),
              (crop, 0, 'Crop: data[1:3, 2:5]')]
    for ax, (pixels, origin, title) in zip(axes, panels):
        rows, cols = pixels.shape
        ax.imshow(pixels, origin='lower', cmap='Blues', vmin=0, vmax=20,
                  extent=(origin - 0.5, cols + origin - 0.5,
                          origin - 0.5, rows + origin - 0.5))
        for y, x in np.ndindex(pixels.shape):
            ax.text(x + origin, y + origin, f'({x + origin}, {y + origin})',
                    ha='center', va='center', fontsize=9)
        ax.set(xticks=np.arange(cols) + origin, yticks=np.arange(rows) + origin,
               xlabel='Column x', ylabel='Row y', title=title)
        ax.set_xticks(np.arange(cols + 1) + origin - 0.5, minor=True)
        ax.set_yticks(np.arange(rows + 1) + origin - 0.5, minor=True)
        ax.grid(which='minor', color='white', linewidth=1.5)
        ax.tick_params(which='minor', length=0)
    axes[0].add_patch(Rectangle((1.5, 0.5), 3, 2, fill=False, ec='C3', lw=2))
    axes[1].add_patch(Rectangle((2.5, 1.5), 3, 2, fill=False, ec='C3', lw=2))
    axes[2].set_xlabel('Column x\nFull-image (2, 1) becomes crop (0, 0)',
                       fontsize=9)
    return fig


def plot_psf_profiles() -> Figure:
    """Compare circular 2D profiles with equal total flux and FWHM."""
    from astropop.math.models import PSFGaussian2D, PSFMoffat2D

    fwhm, flux, power = 4., 1000., 2.5
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    width = fwhm / (2 * np.sqrt(2**(1 / power) - 1))
    radius = np.linspace(0, 5 * fwhm, 1500)
    gaussian = PSFGaussian2D(flux=flux, sigma_x=sigma, sigma_y=sigma)
    moffat = PSFMoffat2D(flux=flux, width=width, power=power)
    # Evaluate the numerical profiles directly. Sphinx resolves model type
    # annotations for display, which can alter Astropy's inferred input units.
    profiles = [model.evaluate(radius, np.zeros_like(radius), *model.parameters)
                for model in (gaussian, moffat)]
    fractions = [1 - np.exp(-radius**2 / (2 * sigma**2)),
                 1 - (1 + (radius / width)**2)**(1 - power)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout='constrained')
    for name, profile, fraction in zip(('Gaussian', 'Moffat'), profiles, fractions):
        axes[0].semilogy(radius / fwhm, profile, label=name)
        axes[1].plot(radius / fwhm, fraction, label=name)
    axes[0].set(xlabel='Radius / FWHM', ylabel='Surface brightness (flux / pixel²)',
                title='Equal integrated flux; different peaks', ylim=(0.01, 100))
    axes[1].set(xlabel='Aperture radius / FWHM', ylabel='Enclosed flux / total flux',
                title='Continuous circular-aperture integral', ylim=(0, 1.03))
    for ax in axes:
        ax.axvline(0.5, color='0.5', linestyle=':', linewidth=1)
        ax.grid(alpha=0.2)
        ax.legend()
    return fig


def setup(app: Sphinx) -> dict[str, bool]:
    """Invalidate task-guide plots when this shared helper changes."""
    def track_source(app: Sphinx, docname: str, source: list[str]) -> None:
        """Record the plotting helper as a dependency of each using page."""
        if 'plot:: _ext/task_figures.py' in source[0]:
            app.env.note_dependency(__file__)

    app.connect('source-read', track_source)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
