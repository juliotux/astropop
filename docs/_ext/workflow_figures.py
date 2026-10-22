"""Render the numerical workflows, reusing their RST source for Sphinx figures."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Iterator

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from astropy import units as u
    from astropy.table import Table
    from astropop.framedata import FrameData
    from astropop.polarimetry import StokesParameters
    from matplotlib.figure import Figure
    from sphinx.application import Sphinx

# Notebook figures use the live results of preceding cells. Sphinx wrappers
# replay those same examples in an isolated directory, without testing helpers.
NOTEBOOK_PLOTS = {
    'plot_observations': 'draw_observations(frame)',
    'plot_beam_image': 'draw_beam_image(first_image, detections, pair)',
    'plot_polarimetry': 'draw_polarimetry(psi, result)',
    'plot_quarterwave': 'draw_polarimetry(quarter_psi, quarter_result)',
    'plot_bias': 'draw_bias(master_bias, first_bias)',
    'plot_flat': 'draw_flat(corrected, master_flat)',
    'plot_science': 'draw_science(calibrated[0], combined, science_files[0])',
    'plot_photometry': 'draw_photometry(image, sky, selected, truth)',
    'plot_light_curve': 'draw_light_curve(curve, truth)',
    'plot_comparison_curves': 'draw_comparison_curves(curve, comparisons, finder["id"][1:])',
    'plot_astrometry': 'draw_astrometry(x, y, east, north, fit_rows, check_rows)',
    'plot_photometric_calibration': ('draw_photometric_calibration(standards, offsets, '
                                   'offset_error, zero_point, zero_point_error, calibrated, '
                                   'calibrated_error, calibrators, check)'),
}


@contextmanager
def example(page: str) -> Iterator[dict[str, object]]:
    """Prepare a temporary dataset and evaluate only the guide's visible examples."""
    from tutorial_data import generate_dataset
    from tutorial_notebooks import execute_workflow

    with TemporaryDirectory() as temporary:
        data = Path(temporary) / 'data'
        generate_dataset(data, include_extras=page in {
            'light_curve', 'astrometric_calibration', 'photometric_calibration', 'polarimetry', 'quarterwave'})
        yield execute_workflow(page, data)


def draw_beam_image(image: FrameData, detections: Table, pair: np.void) -> Figure:
    """Show the actual detected beam positions and photometry apertures."""
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(5.5, 4.5), layout='constrained')
    shown = ax.imshow(image.data, origin='lower', cmap='gray')
    fig.colorbar(shown, ax=ax, label='Pixel value (electron)')
    for key, name, color in [('o', 'Ordinary', 'C0'), ('e', 'Extraordinary', 'C1')]:
        row = detections[pair[key]]
        position = row['x'], row['y']
        for radius in [6, 9, 12]:
            ax.add_patch(Circle(position, radius, fill=False, color=color,
                                linestyle='-' if radius == 6 else '--'))
        ax.annotate(name, position, xytext=(0, 16), textcoords='offset points',
                    ha='center', color=color)
    ax.set(xlabel='Column x (pixel)', ylabel='Row y (pixel)')
    return fig


def plot_beam_image() -> Figure:
    """Render image-to-beam measurement from the maintained tutorial source."""
    with example('polarimetry') as ns:
        return draw_beam_image(ns['first_image'], ns['detections'], ns['pair'])


def draw_polarimetry(psi: np.ndarray, result: StokesParameters) -> Figure:
    """Draw the recovered modulation from the currently loaded beam measurements."""
    from guide_figures import plot_polarimetry

    return plot_polarimetry(psi, result)


def plot_polarimetry() -> Figure:
    """Render image-extracted beam modulation using the published calculations."""
    with example('polarimetry') as ns:
        return draw_polarimetry(ns['psi'], ns['result'])


def plot_quarterwave() -> Figure:
    """Replay the noisy quarter-wave workflow through the shared plot function."""
    with example('quarterwave') as ns:
        return draw_polarimetry(ns['quarter_psi'], ns['quarter_result'])


def draw_observations(frame: FrameData) -> Figure:
    """Show a raw field and its large-scale column profile."""
    from astropy.visualization import simple_norm

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout='constrained')
    axes[0].imshow(frame.data, origin='lower', cmap='gray',
                    norm=simple_norm(frame.data, 'asinh', percent=99.5))
    axes[0].set(title='Raw science exposure', xlabel='x (pixel)', ylabel='y (pixel)')
    axes[1].plot(np.median(frame.data, axis=0))
    axes[1].set(title='Median over image rows', xlabel='Column x (pixel)', ylabel='Signal (ADU)')
    return fig


def plot_observations() -> Figure:
    """Render observation inspection using the published calculations."""
    with example('observations') as ns:
        return draw_observations(ns['frame'])


def draw_bias(master: FrameData, raw: FrameData) -> Figure:
    """Show additive structure and the single-frame/master column profiles."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout='constrained')
    fig.get_layout_engine().set(wspace=0.20)
    image = axes[0].imshow(master.data, origin='lower', cmap='viridis')
    fig.colorbar(image, ax=axes[0], label='Bias (ADU)', shrink=0.8)
    axes[0].set(title='Master bias', xlabel='x (pixel)', ylabel='y (pixel)')
    axes[1].plot(np.median(raw.data, axis=0), alpha=0.6, label='One exposure')
    axes[1].plot(np.median(master.data, axis=0), label='Master bias')
    axes[1].set(title='Median over image rows', xlabel='Column x (pixel)', ylabel='Bias (ADU)')
    axes[1].legend()
    return fig


def plot_bias() -> Figure:
    """Render master-bias construction using the published calculations."""
    with example('master_bias') as ns:
        return draw_bias(ns['master_bias'], ns['first_bias'])


def draw_flat(corrected: FrameData, master: FrameData) -> Figure:
    """Show illumination scale before normalization and the final response map."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout='constrained')
    for ax, frame, title, unit in zip(axes, [corrected, master],
                                     ['One bias-subtracted flat', 'Normalized master flat'],
                                     ['Signal (ADU)', 'Relative response']):
        image = ax.imshow(frame.data, origin='lower', cmap='viridis')
        fig.colorbar(image, ax=ax, label=unit, shrink=0.8)
        ax.set(title=title, xlabel='x (pixel)', ylabel='y (pixel)')
    return fig


def plot_flat() -> Figure:
    """Render flat normalization using the published calculations."""
    with example('master_flat') as ns:
        return draw_flat(ns['corrected'], ns['master_flat'])


def draw_science(first: FrameData, combined: FrameData, path: Path) -> Figure:
    """Compare one exposure before/after calibration and the registered stack."""
    from astropy.io import fits
    from astropy.visualization import simple_norm

    raw = fits.getdata(path, memmap=False)
    norm = simple_norm(combined.data, 'asinh', percent=99.5)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout='constrained')
    for ax, pixels, title in zip(axes, [raw, first.data, combined.data],
                                 ['Raw exposure', 'Bias/flat corrected', 'Registered clipped mean']):
        ax.imshow(pixels, origin='lower', cmap='gray', norm=norm)
        ax.set(title=title, xlabel='x (pixel)', ylabel='y (pixel)')
    return fig


def plot_science() -> Figure:
    """Render science calibration using the published calculations."""
    with example('science_reduction') as ns:
        return draw_science(ns['calibrated'][0], ns['combined'], ns['science_files'][0])


def draw_photometry(image: FrameData, sky: np.ndarray, measured: Table, truth: Table) -> Figure:
    """Show apertures, background and recovery against unused injected source truth."""
    from astropy.visualization import simple_norm
    from matplotlib.patches import Circle

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout='constrained')
    axes[0].imshow(image.data, origin='lower', cmap='gray',
                    norm=simple_norm(image.data, 'asinh', percent=99.5))
    for row in measured:
        for radius, color in [(6, 'cyan'), (9, 'orange'), (13, 'orange')]:
            axes[0].add_patch(Circle((row['x'], row['y']), radius, fill=False, color=color, lw=0.7))
    axes[0].set_title('Apertures and sky annuli')
    sky_image = axes[1].imshow(sky, origin='lower', cmap='viridis')
    axes[1].set_title('Estimated sky')
    fig.colorbar(sky_image, ax=axes[1], label='ADU', shrink=0.8)
    d2 = (np.asarray(measured['x'])[:, None] - truth['x'])**2
    d2 += (np.asarray(measured['y'])[:, None] - truth['y'])**2
    injected = truth['flux'][np.argmin(d2, axis=1)]
    axes[2].errorbar(injected, 100 * (measured['flux'] / injected - 1),
                     yerr=100 * measured['flux_error'] / injected, fmt='o')
    axes[2].axhline(0, color='0.5', linestyle='--')
    axes[2].set(title='Simulation recovery', xlabel='Injected flux (ADU)', ylabel='Flux residual (%)')
    for ax in axes[:2]:
        ax.set(xlabel='x (pixel)', ylabel='y (pixel)')
    return fig


def plot_photometry() -> Figure:
    """Render aperture measurements using the published calculations."""
    with example('photometry') as ns:
        return draw_photometry(ns['image'], ns['sky'], ns['selected'], ns['truth'])


def draw_light_curve(curve: Table, truth: Table) -> Figure:
    """Compare shared throughput and the recovered differential variability."""
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.4), sharex=True, layout='constrained')
    axes[0].plot(curve['elapsed_min'], curve['target_relative'], 'o-', label='Target')
    axes[0].plot(curve['elapsed_min'], curve['ensemble'], 's--', label='Comparison ensemble')
    axes[0].set(ylabel='Normalized flux', title='Individual-exposure photometry')
    axes[0].legend(fontsize='small')
    axes[1].errorbar(curve['elapsed_min'], curve['relative_flux'],
                     yerr=curve['flux_error'], fmt='o', label='Differential target')
    axes[1].plot(truth['elapsed_minutes'], truth['relative_flux'], '--', label='Injected variability')
    axes[1].set(xlabel='Elapsed mid-exposure time (min, UTC)', ylabel='Differential flux')
    axes[1].legend(fontsize='small')
    for ax in axes:
        ax.grid(alpha=0.2)
    return fig


def plot_light_curve() -> Figure:
    """Render the time series using the published calculations."""
    with example('light_curve') as ns:
        return draw_light_curve(ns['curve'], ns['truth'])


def draw_comparison_curves(curve: Table, comparisons: np.ndarray,
                            identifiers: np.ndarray) -> Figure:
    """Inspect each comparison against an ensemble that excludes that star."""
    count = comparisons.shape[1]
    if count < 2:
        raise ValueError('At least two comparison stars are needed for this diagnostic.')
    others = (comparisons.sum(axis=1)[:, None] - comparisons) / (count - 1)
    ratios = comparisons / others
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.5, 5.4),
                             layout='constrained')
    for index, identifier in enumerate(identifiers):
        axes[0].plot(curve['elapsed_min'], comparisons[:, index], 'o-', label=str(identifier))
        axes[1].plot(curve['elapsed_min'], ratios[:, index], 'o-', label=str(identifier))
    axes[0].set(ylabel='Flux / own median', title='Individual comparison measurements')
    axes[1].axhline(1, color='0.4', linestyle='--', linewidth=1)
    axes[1].set(xlabel='Elapsed mid-exposure time (min, UTC)',
                ylabel='Star / other comparisons', title='Exclude each star from its reference')
    for ax in axes:
        ax.legend(fontsize='small')
        ax.grid(alpha=0.2)
    return fig


def plot_comparison_curves() -> Figure:
    """Replay the tutorial and call the same helper used by notebook cells."""
    with example('light_curve') as ns:
        return draw_comparison_curves(ns['curve'], ns['comparisons'], ns['finder']['id'][1:])


def draw_astrometry(x: np.ndarray, y: np.ndarray, east: u.Quantity, north: u.Quantity,
                    fit: np.ndarray, check: np.ndarray) -> Figure:
    """Locate fit/check stars and display their angular WCS residuals."""
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout='constrained')
    fig.get_layout_engine().set(wspace=0.25)
    axes[1].xaxis.set_major_locator(MaxNLocator(5))
    axes[1].yaxis.set_major_locator(MaxNLocator(5))
    radial = np.hypot(east.arcsec, north.arcsec)
    for rows, marker, label in [(fit, 'o', 'Fit stars'), (check, '*', 'Check stars')]:
        image = axes[0].scatter(x[rows], y[rows], c=radial[rows], cmap='viridis',
                                 vmin=0, vmax=radial.max(), marker=marker, s=85, label=label)
        axes[1].scatter(east.arcsec[rows], north.arcsec[rows], marker=marker, s=65, label=label)
    fig.colorbar(image, ax=axes[0], label='Angular residual (arcsec)', shrink=0.8)
    axes[0].set(xlabel='x (pixel)', ylabel='y (pixel)', title='References across the field',
                xlim=(0, 256), ylim=(0, 256))
    axes[1].set(xlabel='East residual (arcsec)', ylabel='North residual (arcsec)',
                title='Fitted WCS residuals')
    axes[1].axhline(0, color='0.5', linestyle='--')
    axes[1].axvline(0, color='0.5', linestyle='--')
    axes[1].legend(fontsize='small')
    return fig


def plot_astrometry() -> Figure:
    """Render offline WCS fitting using the published calculations."""
    with example('astrometric_calibration') as ns:
        return draw_astrometry(ns['x'], ns['y'], ns['east'], ns['north'],
                                ns['fit_rows'], ns['check_rows'])


def draw_photometric_calibration(standards: Table, offsets: np.ndarray,
                                 offset_error: np.ndarray, zero: float, zero_error: float,
                                 calibrated: np.ndarray, calibrated_error: np.ndarray,
                                 fit: np.ndarray, check: np.ndarray) -> Figure:
    """Display per-standard offsets, fitted zero point and a held-out check star."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout='constrained')
    for rows, marker, label in [(fit, 'o', 'Standards'), (check, '*', 'Check star')]:
        axes[0].errorbar(standards['R'][rows], offsets[rows], yerr=offset_error[rows],
                         fmt=marker, label=label)
        error = np.hypot(calibrated_error[rows], standards['R_error'][rows])
        axes[1].errorbar(standards['R'][rows], (calibrated - standards['R'])[rows],
                         yerr=error, fmt=marker, label=label)
    axes[0].axhline(zero, color='C2', label='Fitted zero point')
    axes[0].axhspan(zero - zero_error, zero + zero_error, color='C2', alpha=0.15)
    axes[0].axhline(25, color='0.4', linestyle='--', label='Injected zero point')
    axes[0].set(xlabel='Reference R (mag)', ylabel='Zero-point estimate (mag)', title='Local calibration')
    axes[1].axhline(0, color='0.4', linestyle='--')
    axes[1].set(xlabel='Reference R (mag)', ylabel='Calibrated minus reference (mag)',
                title='Calibration residuals')
    for ax in axes:
        ax.legend(fontsize='small')
        ax.grid(alpha=0.2)
    return fig


def plot_photometric_calibration() -> Figure:
    """Render zero-point fitting using the published calculations."""
    with example('photometric_calibration') as ns:
        return draw_photometric_calibration(ns['standards'], ns['offsets'], ns['offset_error'],
                                            ns['zero_point'], ns['zero_point_error'],
                                            ns['calibrated'], ns['calibrated_error'],
                                            ns['calibrators'], ns['check'])


def setup(app: Sphinx) -> dict[str, bool]:
    """Track the shared source used by the workflow figures."""
    def track_source(app: Sphinx, docname: str, source: list[str]) -> None:
        """Invalidate pages when either plotting or example-replay code changes."""
        if 'plot:: _ext/workflow_figures.py' in source[0]:
            app.env.note_dependency(__file__)
            app.env.note_dependency(str(Path(__file__).with_name('guide_figures.py')))
            app.env.note_dependency(str(Path(__file__).with_name('tutorial_notebooks.py')))
            app.env.note_dependency(str(Path(__file__).with_name('tutorial_data.py')))
            app.env.note_dependency(str(Path(__file__).with_name('workflow_data.py')))

    app.connect('source-read', track_source)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
