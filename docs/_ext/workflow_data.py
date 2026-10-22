"""Extend the CCD teaching observations with local standards and a time series."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS


def extend_dataset(data: Path) -> None:
    """Add standards, pointing, paired-beam images and a separate light curve."""
    from astropop.framedata import FrameData, read_framedata
    from tutorial_data import measure_dataset

    truth = Table.read(data / 'sources.ecsv')
    camera = WCS(naxis=2)
    camera.wcs.crpix = [128.5, 128.5]
    camera.wcs.crval = [150., -30.]
    camera.wcs.cdelt = [-0.34 / 3600, 0.34 / 3600]
    camera.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    coords = camera.pixel_to_world(truth['x'], truth['y'])
    standards = Table({'id': [f'S{i + 1:02d}' for i in range(9)],
                       'ra': coords.ra.deg, 'dec': coords.dec.deg,
                       'R': 25 - 2.5 * np.log10(truth['flux'] / 30),
                       'R_error': np.full(9, 0.01),
                       'standard': [False, True, True, True, True, True, True, True, False]})
    standards['ra'].unit = standards['dec'].unit = u.deg
    standards['R'].unit = standards['R_error'].unit = u.mag
    standards.write(data / 'standards.ecsv', overwrite=True)
    rough = camera.deepcopy()
    rough.wcs.crpix += [3, -2]
    fits.writeto(data / 'pointing.fits', np.zeros((1, 1)), rough.to_header(), overwrite=True)
    measured, _, _ = measure_dataset(data)
    measured.write(data / 'reference' / 'instrumental.ecsv', overwrite=True)
    matched = identify_sources(measured, standards, rough)
    fitted = fit_astrometry(measured, standards, matched)
    combined = read_framedata(data / 'reference' / 'combined.fits')
    combined.wcs = fitted
    combined.write(data / 'reference' / 'astrometric.fits', overwrite=True)

    angle = np.arange(8) * 22.5
    modulation = 0.03 * np.cos(np.deg2rad(4 * angle)) - 0.02 * np.sin(np.deg2rad(4 * angle))
    ordinary = 10000 * (1 + modulation) / 2
    extraordinary = 10000 * (1 - modulation) / 2
    beams = Table({'angle_deg': angle, 'ordinary': ordinary, 'extraordinary': extraordinary,
                   'ordinary_error': np.sqrt(ordinary),
                   'extraordinary_error': np.sqrt(extraordinary)})
    beams.meta.update({'retarder': 'halfwave', 'normalization': 1., 'zero_deg': 0.,
                       'nominal_measurements': 'noiseless; supplied Poisson error model'})
    beams.write(data / 'polarimetry.ecsv', overwrite=True)
    write_beam_images(data, beams)

    # A separate noisy sequence uses known instrument calibration constants.
    # Keep its random stream independent of the imaging/time-series examples.
    from astropop.polarimetry import quarterwave_model
    quarter_angles = np.arange(16) * 22.5
    quarter_signal = quarterwave_model(quarter_angles, 0.04, -0.03, 0.02, zero=7.5)
    beam_rng = np.random.default_rng(7)
    ordinary = beam_rng.poisson(1e5 * (1 + quarter_signal) / 2)
    extraordinary = beam_rng.poisson(1e5 * (1 - quarter_signal) / (2 * 1.1))
    quarter = Table({'angle_deg': quarter_angles, 'ordinary': ordinary,
                     'extraordinary': extraordinary,
                     'ordinary_error': np.sqrt(ordinary),
                     'extraordinary_error': np.sqrt(extraordinary)})
    quarter.meta.update({'retarder': 'quarterwave', 'normalization': 1.1,
                         'zero_deg': 7.5, 'nominal_measurements': 'independent Poisson draws'})
    quarter.write(data / 'polarimetry_quarterwave.ecsv', overwrite=True)

    series = data / 'time_series'
    (series / 'raw').mkdir(parents=True, exist_ok=True)
    finder = Table({'id': standards['id'][:4], 'role': ['target'] + ['comparison'] * 3,
                    'x': np.round(truth['x'][:4]), 'y': np.round(truth['y'][:4])})
    finder.write(series / 'finder.ecsv', overwrite=True)
    bias = fits.getdata(data / 'true_bias.fits', memmap=False)
    response = fits.getdata(data / 'true_flat.fits', memmap=False)
    y, x = np.indices(bias.shape)
    rng = np.random.default_rng(104)
    intrinsic = 1 + 0.15 * np.sin(2 * np.pi * np.arange(12) / 12)
    transparency = 0.95 + 0.04 * np.cos(2 * np.pi * np.arange(12) / 12)
    for i in range(12):
        signal = 160 + 0.3 * x + 0.2 * y
        for j, source in enumerate(truth):
            flux = source['flux'] * transparency[i] * (intrinsic[i] if j == 0 else 1)
            radius2 = (x - source['x'] - 0.4 * i)**2 + (y - source['y'] + 0.25 * i)**2
            signal += flux / (2 * np.pi * 1.8**2) * np.exp(-radius2 / (2 * 1.8**2))
        pixels = bias + rng.poisson(signal * response) + rng.normal(0, 4, bias.shape)
        time = Time('2025-01-01T00:00:00', scale='utc') + i * 60 * u.s
        frame = FrameData(pixels, unit='adu', dtype='float32',
                          meta={'EXPTIME': 30., 'GAIN': 1., 'RDNOISE': 4.,
                                'DATE-OBS': time.isot, 'IMAGETYP': 'LIGHT', 'FILTER': 'R'})
        frame.write(series / 'raw' / f'science_{i:02d}.fits', overwrite=True)
    Table({'elapsed_minutes': np.arange(12), 'relative_flux': intrinsic,
           'transparency': transparency}).write(series / 'truth.ecsv', overwrite=True)


def write_beam_images(data: Path, beams: Table) -> None:
    """Write ideal calibrated beam images with a known sky and fixed geometry.

    Nominal pixels are noiseless. The tutorial independently supplies its
    illustrative counting-error model during aperture measurement.
    """
    folder = data / 'polarimetry' / 'images'
    folder.mkdir(parents=True, exist_ok=True)
    y, x = np.indices((64, 64))
    for index, beam in enumerate(beams):
        pixels = np.full(x.shape, 100.)
        for xc, yc, flux in [(20, 20, beam['ordinary']),
                             (44, 44, beam['extraordinary'])]:
            radius2 = (x - xc)**2 + (y - yc)**2
            pixels += flux / (2 * np.pi * 1.5**2) * np.exp(-radius2 / (2 * 1.5**2))
        header = fits.Header({'BUNIT': 'electron', 'RETANGLE': float(beam['angle_deg'])})
        fits.writeto(folder / f'halfwave_{index:02d}.fits', pixels.astype('float32'),
                     header, overwrite=True)


def identify_sources(measured: Table, standards: Table, rough: WCS) -> np.ndarray:
    """Associate reference stars using approximate pointing, not injected pixel truth."""
    coordinates = SkyCoord(standards['ra'], standards['dec'])
    x, y = rough.world_to_pixel(coordinates)
    distance2 = (np.asarray(measured['x'])[:, None] - x)**2
    distance2 += (np.asarray(measured['y'])[:, None] - y)**2
    return np.argmin(distance2, axis=0)


def fit_astrometry(measured: Table, standards: Table, matched: np.ndarray) -> WCS:
    """Fit a TAN projection using seven stars, reserving two independent checks."""
    from astropy.wcs.utils import fit_wcs_from_points

    return fit_wcs_from_points((measured['x'][matched[:7]], measured['y'][matched[:7]]),
                               SkyCoord(standards['ra'][:7], standards['dec'][:7]),
                               projection='TAN')
