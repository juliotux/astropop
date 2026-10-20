Installation and first steps
============================

Requirements
------------

The main dependencies of |astropop| are:

 - Python 3.11 or newer
 - NumPy 2.0 or newer
 - Astropy 6.1.4 or newer
 - SciPy 1.13 or newer
 - Photutils 3.0 or newer.

The complete runtime dependency list and optional extras are declared in ``pyproject.toml``;
pip installs the required Python packages automatically. Released versions can
have different requirements, so consult the metadata of the version installed.

Install a release in an activated virtual or conda environment::

    python -m pip install astropop

To include optional plotting support::

    python -m pip install 'astropop[all]'

For an editable checkout, install from the repository root::

    git clone https://github.com/sparc4-dev/astropop.git
    cd astropop
    python -m pip install -e '.[all,test,docs]'

The ``test`` extra installs test helpers; ``docs`` installs documentation
packages. Some scientific dependencies contain compiled extensions. Prefer
available binary wheels or install the corresponding packages in your conda
environment; building those dependencies from source may require compilers.

Astrometry.net is optional external software
--------------------------------------------

The Astrometry.net solver wrappers additionally need the ``solve-field``
executable and appropriate local index files. pip installation of Astropop does
not install that executable or the index data. Other modules, including manual
WCS construction, do not require Astrometry.net. See :doc:`reduction/astrometry`
for solver configuration. Online catalog queries additionally require access
to their external services; ordinary image reductions do not.

A first reduction
-----------------

Continue with :doc:`tutorials/ccd_reduction` for a complete offline
FITS-to-photometry example after this first calculation.

Check which installation the interpreter imports, especially when using a
notebook kernel::

    >>> import astropop
    >>> assert isinstance(astropop.__version__, str)

FrameData associates image values with units, optional standard uncertainties,
flags, and metadata. Use explicit reduction functions rather than Python
arithmetic operators on FrameData::

    >>> import numpy as np
    >>> from astropop.framedata import FrameData
    >>> from astropop.image.imarith import imarith
    >>> image = FrameData(np.full((4, 4), 100.), uncertainty=2., unit='adu')
    >>> result = imarith(image, 2, '*')
    >>> np.testing.assert_allclose(result.data, 200.)
    >>> np.testing.assert_allclose(result.uncertainty, 4.)

Use ``read_framedata('image.fits')`` for existing files. Set optional-array
configuration **before loading or creating frames** when memory savings are
required; see :ref:`optional-frame-arrays`. Continue with :doc:`user_guide` for
calibration, registration, combination, photometry, polarimetry, and catalogs.

Running checks and building the documentation
---------------------------------------------

From an editable checkout with the corresponding extras installed::

    python -m pytest tests docs --remote-data=none
    python -m sphinx -b html docs docs/_build/html

Building notebook pages requires the Pandoc executable on ``PATH`` in addition
to the Python documentation packages. The documentation build renders stored
notebook examples without executing remote queries. The ordinary test command
uses recorded catalog responses; live catalog conformance is a separate,
explicit check described in :doc:`dev/testing`.

.. currentmodule:: None

.. py:data:: astropop.__version__

    Installed Astropop version string. For an editable checkout the version is
    derived from Git metadata by setuptools-scm. Print this value and
    ``astropop.__file__`` when diagnosing an unexpected notebook installation.
