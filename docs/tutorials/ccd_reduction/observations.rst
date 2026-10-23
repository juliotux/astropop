Inspect and organize the observations
=====================================

Begin the reduction by identifying what each file contains. This observing
sequence has eight bias frames, six flats and six science exposures. Do not
select calibrations from their filenames alone on a real observing run.

.. tutorial-notebook:: observations

.. tutorial-data::

Extract the shared archive and start with ``data/raw``. This first stage
inspects the raw exposures and saves their selection in ``reduced``.
See :doc:`../index` for archive layout, configuration and rerun policy.
Run from the extracted bundle root when following the HTML examples::

    >>> from pathlib import Path

.. code-block:: python

    data = Path('data')

.. doctest::
    :hide:

    >>> from tempfile import TemporaryDirectory
    >>> from tutorial_data import generate_dataset

    >>> _temporary = TemporaryDirectory()
    >>> data = Path(_temporary.name).resolve() / 'data'
    >>> generate_dataset(data, include_extras=False)

Read a header before selecting files
------------------------------------

These files store their image in the primary HDU. Other instruments may put
science data in an image extension; select that extension explicitly::

    >>> import numpy as np
    >>> from astropy.io import fits
    >>> from astropy.table import Table
    >>> from astropop.framedata import read_framedata

    >>> files = sorted((data / 'raw').glob('*.fits'))
    >>> rows = []
    >>> for path in files:
    ...
    ...     # Read the observing metadata without loading the image pixels.
    ...     header = fits.getheader(path)
    ...     rows.append((path.name, header['IMAGETYP'], header['FILTER'],
    ...                  header['EXPTIME'], header['DATE-OBS']))

    >>> log = Table(rows=rows, names=['file', 'type', 'filter', 'exposure_s', 'start_utc'])

    >>> for kind in ['BIAS', 'FLAT', 'LIGHT']:
    ...     print(f'{kind}: {np.count_nonzero(log["type"] == kind)} frames')
    BIAS: 8 frames
    FLAT: 6 frames
    LIGHT: 6 frames

Keep units and instrument settings together with the log. For these simulated
observations the gain is one electron per ADU, read noise is 4 electrons,
science exposure time is 30 seconds, and all science frames use the same filter.
There is no dark current, overscan, saturation or nonlinearity in this teaching
camera. Those corrections must be planned separately for an actual detector.

.. _calibration-selection:

Choose compatible calibration groups
------------------------------------

Build groups from headers and detector knowledge before constructing masters.
The keyword names vary by instrument; missing metadata must be resolved from
the observing setup rather than treated as agreement.

.. list-table:: Calibration compatibility
    :header-rows: 1
    :widths: 25 75

    * - Setting
      - Selection rule
    * - Camera, channel and readout
      - Match detector/amplifier, readout speed, gain mode and read-noise regime.
        Bias offsets and spatial response can differ between modes.
    * - Geometry and binning
      - Match binning, subframe, overscan convention and trim region, including
        their pixel origin. Equal image dimensions alone are insufficient.
    * - Temperature and time
      - Select the detector's validated stable range; assess bias/dark changes
        with temperature and time rather than assuming a night's frames agree.
    * - Filter and optical path
      - Match flats to the science filter/channel and illumination setup. A bias
        need not share a filter, but must share the relevant detector settings.
    * - Exposure duration
      - Verify finite positive science/flat durations, the intended near-zero
        bias exposure and a common duration for a dark master before scaling.

The shared synthetic data use one camera geometry and one readout setting;
their headers do not encode a complete real-instrument configuration. The
later notebooks use ``bias_*.fits``, ``flat_*.fits`` and ``science_*.fits`` as
already selected sets so each can run independently. They do not consume the
saved observing log automatically. On observations, replace those globs with
file lists selected from the reviewed log or :doc:`../diagnostics/fits_collections`,
and keep the selection criteria with the master products.

Before applying a saved master, recheck its geometry, unit and settings against
the selected science group. The compatibility rules above also apply when
loading reference products or rerunning a notebook; file existence establishes
availability, not compatibility or freshness.

Select files through ``FitsFileGroup``
--------------------------------------

Use the header index to select exposures without loading their pixel arrays.
``IMAGETYP`` is this camera's exposure-type keyword; select the equivalent
keyword for another instrument. The additional compatibility rules above still
apply to each selected group::

    >>> from astropop.file_collection import FitsFileGroup

    >>> group = FitsFileGroup(files=[str(path) for path in files], ext=0)

    >>> # Select by exposure type; detector and filter compatibility need separate checks.
    >>> science_group = group.filtered({'IMAGETYP': 'LIGHT'})
    >>> science_files = [Path(name) for name in science_group.files]

    >>> print('Indexed exposures:', len(group), '; science exposures:', len(science_group))
    Indexed exposures: 20 ; science exposures: 6

The observing log records what was inspected; the group supplies the input
paths for image loading. Explicit file lists use string paths for the header
index's database backend. Neither interface modifies the raw observations.

Inspect the field and create file lists
---------------------------------------

Leave optional uncertainties and flags enabled before loading frames. Initial
raw frames have no uncertainty extension; loading them does not invent detector
errors. The science reduction stage estimates those errors explicitly::

    >>> frame = read_framedata(science_files[0])

    >>> print(f'Image: {frame.shape}, {frame.unit}; raw uncertainty: {frame.uncertainty}')
    Image: (256, 256), adu; raw uncertainty: None
    >>> print(f'Exposure: {frame.meta["EXPTIME"]:.0f} s; filter: {frame.meta["FILTER"]}')
    Exposure: 30 s; filter: R

    >>> output = data.parent / 'reduced'
    >>> output.mkdir(exist_ok=True)
    >>> log.write(output / 'observing_log.ecsv', overwrite=True)
    >>> for kind in ['BIAS', 'FLAT', 'LIGHT']:
    ...     names = [row['file'] for row in log if row['type'] == kind]
    ...     _ = (output / f'{kind.lower()}_files.txt').write_text('\n'.join(names) + '\n')

    >>> print('Saved: observing_log.ecsv and three file lists')
    Saved: observing_log.ecsv and three file lists

.. plot:: _ext/workflow_figures.py plot_observations
    :include-source: false
    :alt: Raw science field with stars and an edge event, beside its column median profile.
    :caption: The raw image contains both sky and detector structure. The column median shows large-scale structure that should not be interpreted as an astronomical source.

The edge event is not a star. A display stretch helps reveal the field but does
not modify pixels or fix defects. Inspect the rest of the sequence for changes
in illumination, geometry, filter, temperature and exposure before grouping it.
For a persistent searchable header database see :doc:`../diagnostics/fits_collections`.
Continue with :doc:`master_bias`.

.. doctest::
    :hide:

    >>> del group, science_group
    >>> _temporary.cleanup()
