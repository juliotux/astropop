Construct a master bias
=======================

Combine bias exposures before subtracting the detector's additive offset from
flats or science frames. The structured bias here varies across the detector;
a single scalar subtraction would not remove that structure.

.. tutorial-notebook:: master_bias

.. tutorial-data::

Use the bias exposures in ``data/raw``. This stage writes
``reduced/master_bias.fits``; see :doc:`index` for the dataset assumptions.
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

Select and combine bias exposures
---------------------------------

Follow the :ref:`calibration compatibility rules <calibration-selection>` to
verify exposure type and matching camera settings against the observing log.
This bundle's filenames provide a convenient preselected set::

    >>> import numpy as np
    >>> from astropop.framedata import read_framedata
    >>> from astropop.image import imcombine

    >>> bias_files = sorted((data / 'raw').glob('bias_*.fits'))
    >>> first_bias = read_framedata(bias_files[0])

    >>> # Combine the file inputs directly without keeping a separate loaded-image list.
    >>> master_bias = imcombine(bias_files, method='median', dtype='float32')

    >>> print(f'Bias frames: {len(bias_files)}; master: {master_bias.shape}, {master_bias.dtype}')
    Bias frames: 8; master: (256, 256), float32
    >>> print(f'Median bias level: {np.median(master_bias.data):.1f} {master_bias.unit}')
    Median bias level: 102.6 adu

The median limits the influence of isolated events. A mean with appropriate
sigma clipping is another choice; compare actual samples and the detector's
noise before selecting a rejection threshold. See
:doc:`../../modules/image/imcombine` for an explicit transient comparison.
``dtype='float32'`` bounds storage while allowing fractional combined values.

Inspect the spatial structure and save
---------------------------------------

.. plot:: _ext/workflow_figures.py plot_bias
    :include-source: false
    :alt: Master bias image and column profiles of an individual bias exposure and the master bias.
    :caption: Combining preserves the additive detector structure while smoothing the column profile's random noise. The offset remains spatially dependent.

A smooth master does not establish that the detector was stable throughout the
night. Compare bias frames from different times or temperatures when that is
relevant. The output uncertainty is the current combination scatter estimator,
not a direct measurement of the detector's read noise::

    >>> output = data.parent / 'reduced'
    >>> output.mkdir(exist_ok=True)
    >>> master_bias.write(output / 'master_bias.fits', overwrite=True)

    >>> print(f'Saved: master_bias.fits; units: {master_bias.unit}')
    Saved: master_bias.fits; units: adu

These examples overwrite their named derived products to allow rerunning a
notebook; they do not overwrite raw files. Continue with :doc:`master_flat`.

.. doctest::
    :hide:

    >>> _temporary.cleanup()
