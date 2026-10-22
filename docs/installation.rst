.. include:: references.txt

Installation and first steps
============================

Requirements
------------

The main dependencies of |astropop| are:

- Python 3.11 or newer
- NumPy 2.0 or newer
- Astropy 6.1.4 or newer
- SciPy 1.13 or newer
- Photutils 3.0 or newer

The complete runtime dependency list and optional extras are declared in
``pyproject.toml``; pip installs the required Python packages automatically.
Released versions can have different requirements, so consult the metadata of
the version installed.

Create an environment and install a release
-------------------------------------------

Choose a virtual environment or a conda environment. With Python's built-in
virtual environments::

    python -m venv .venv

Activate it with ``source .venv/bin/activate`` on Linux/macOS, or
``.venv\Scripts\Activate.ps1`` in Windows PowerShell. Alternatively, create and
activate a conda environment::

    conda create -n astropop -c conda-forge python=3.14 pip
    conda activate astropop

Install Astropop using that environment's interpreter::

    python -m pip install astropop

To include optional plotting support::

    python -m pip install 'astropop[all]'

For an editable checkout, install from the repository root::

    git clone https://github.com/sparc4-dev/astropop.git
    cd astropop
    python -m pip install -e '.[all,test,docs]'

The ``test`` extra installs test helpers; ``docs`` installs documentation
packages. Building this development documentation requires Python 3.12 or
newer for automatic annotation rendering; the maintained documentation builds
use Python 3.14. Runtime use of Astropop still supports Python 3.11.
Some scientific dependencies contain compiled extensions. Prefer
available binary wheels or install the corresponding packages in your conda
environment; building those dependencies from source may require compilers.

Use the environment in Jupyter
------------------------------

Install Jupyter and register a kernel from the activated environment::

    python -m pip install jupyterlab ipykernel
    python -m ipykernel install --user --name astropop --display-name "Python (astropop)"
    jupyter lab

Select **Python (astropop)** in the notebook's kernel menu. After changing the
installation, restart the kernel and rerun the notebook from the start. Check
its interpreter and imported package path (the paths depend on your machine)::

    >>> import sys
    >>> import astropop
    >>> print('Interpreter:', sys.executable)  # doctest: +ELLIPSIS
    Interpreter: ...
    >>> print('Imported package:', astropop.__file__)  # doctest: +ELLIPSIS
    Imported package: ...

The ``latest`` documentation describes the development branch. An installed
release may have older signatures; use documentation for that release or an
editable development checkout when following examples for unreleased features.

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
notebook kernel (the version depends on your installation)::

    >>> import astropop
    >>> print('Astropop version:', astropop.__version__)  # doctest: +ELLIPSIS
    Astropop version: ...

FrameData associates image values with units, optional standard uncertainties,
flags, and metadata. Use explicit reduction functions rather than Python
arithmetic operators on FrameData::

    >>> import numpy as np
    >>> from astropop.framedata import FrameData
    >>> from astropop.image.imarith import imarith
    >>> image = FrameData(np.full((4, 4), 100.), uncertainty=2., unit='adu')
    >>> result = imarith(image, 2, '*')
    >>> print('result.data:', np.round(result.data, 6))
    result.data: [[200. 200. 200. 200.]
     [200. 200. 200. 200.]
     [200. 200. 200. 200.]
     [200. 200. 200. 200.]]
    >>> print('result.uncertainty:', np.round(result.uncertainty, 6))
    result.uncertainty: [[4. 4. 4. 4.]
     [4. 4. 4. 4.]
     [4. 4. 4. 4.]
     [4. 4. 4. 4.]]

Use ``read_framedata('image.fits')`` for existing files. Set optional-array
configuration **before loading or creating frames** when memory savings are
required; see :ref:`optional-frame-arrays`. Continue with :doc:`tutorials/index`
for complete workflows or :doc:`how_to/index` for module explanations and simple API examples.

Running checks and building the documentation
---------------------------------------------

From an editable checkout with the corresponding extras installed::

    python -m pytest tests docs --remote-data=none
    python -m sphinx -b html docs docs/_build/html

The documentation build generates notebook downloads from the tutorial pages;
it does not execute remote-query recipes. The ordinary test command
uses recorded catalog responses; live catalog conformance is a separate,
explicit check described in :doc:`dev/testing`.

.. currentmodule:: None

.. py:data:: astropop.__version__

    Installed Astropop version string. For an editable checkout the version is
    derived from Git metadata by setuptools-scm. Print this value and
    ``astropop.__file__`` when diagnosing an unexpected notebook installation.
