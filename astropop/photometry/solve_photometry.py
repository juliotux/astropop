# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np

__all__ = ['solve_photometry_montecarlo', 'solve_photometry_median',
           'solve_photometry_average']


def solve_photometry_median(fluxes, flux_error, references, limits=(5, 18)):
    mags = -2.5*np.log10(fluxes)

    a, b = limits
    a, b = a, b if a < b else b, a
    args = np.where(np.logical_and(references >= a, references <= b))

    diff = references - mags
    dif = np.nanmedian(diff[args])
    err = np.nanstd(diff[args])

    error = 1.086*((flux_error + np.sqrt(fluxes))/fluxes) + err
    return mags + dif, error


def solve_photometry_average(fluxes, flux_error, references, limits=(5, 18)):
    mags = -2.5*np.log10(fluxes)

    a, b = limits
    a, b = a, b if a < b else b, a
    args = np.where(np.logical_and(references >= a, references <= b))

    diff = references - mags
    dif = np.nanaverage(diff[args], weights=np.divide(1, flux_error[args]))
    err = np.nanstd(diff[args])

    error = 1.086*((flux_error + np.sqrt(fluxes))/fluxes) + err
    return mags + dif, error


def _montecarlo_loop(args):
    mags = args[0]
    references = args[1]
    n_stars = args[2]

    iter_mags = np.zeros(len(mags))
    iter_mags[:] = np.nan

    choices = np.random.choice(len(mags), n_stars)
    iter_mags = mags + np.nanmedian(references[choices] -
                                    mags[choices])
    return iter_mags


def solve_photometry_montecarlo(fluxes, flux_error, ref_mags, limits=(5, 18),
                                n_iter=100, n_stars=0.2):
    mags = -2.5*np.log10(fluxes)

    if float(n_stars).is_integer():
        n_stars = n_stars
    else:
        n_stars = max(1, int(n_stars*len(fluxes)))

    nrefs = np.array(ref_mags)
    lim = sorted(limits)
    filt = np.where(np.logical_or(nrefs < lim[0], nrefs > lim[1]))
    nrefs[filt] = np.nan

    args = (mags, nrefs, n_stars)
    iter_mags = [_montecarlo_loop(args) for i in range(n_iter)]

    result = np.nanmedian(iter_mags, axis=0)
    errors = np.nanstd(iter_mags, axis=0)
    return result, errors
