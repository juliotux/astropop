"""Draw offline diagnostics using the small examples in the authored guides."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from astropop.polarimetry import StokesParameters


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
