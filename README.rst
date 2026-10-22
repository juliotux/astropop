AstroPoP
========

|GHAction Status| |Codecov Status| |RTD Status| |CODACY|

Astropop provides Python tools for astronomical image reduction, photometry
and dual-beam polarimetry. It was developed for IAGPOL and SPARC4 at
Observatório Pico dos Dias, Brazil, and can be used in other observing workflows.

Features
^^^^^^^^

Combine these tools into a reduction sequence suited to your instrument:

- FITS image storage with units, optional uncertainties, pixel flags and metadata.
- Bias, dark, flat-field and gain corrections, trimming and cosmic-ray removal.
- Image registration and chunked mean, median and sum combination with rejection.
- Background estimation, source detection and circular-aperture photometry.
- Dual-beam pair matching and half-wave/quarter-wave Stokes-parameter fitting.
- SIMBAD, VizieR and Gaia queries and local source matching.
- FITS file grouping by headers and wrappers for the external Astrometry.net solver.

The `offline tutorials <https://astropop.readthedocs.io/en/latest/tutorials/index.html>`_
show how to connect these steps and inspect their results. Catalog matching
supplies candidate references; selecting standards and fitting a photometric
calibration remain explicit workflow steps. Astropop does not currently provide
PSF photometry or an automatic end-to-end reduction for every instrument.

Support and Community
^^^^^^^^^^^^^^^^^^^^^

We have a community of people using astropop to perform data reduction. Also, we use this community to offer support for astropop users. Join the community at `astropop-users Google-Groups <https://groups.google.com/g/astropop-users>`_

Dependencies
^^^^^^^^^^^^

Astropop requires Python 3.11 or newer. Required Python packages are declared
in ``pyproject.toml`` and installed automatically by pip. The current checkout
requires NumPy >= 2.0, Astropy >= 6.1.4, SciPy >= 1.13, and Photutils >= 3.0,
along with the other dependencies listed in that metadata. Matplotlib is an
optional plotting dependency, available through ``astropop[all]``.

Some dependencies contain compiled extensions; binary wheels or conda packages
can avoid building them locally. The Astrometry.net solver wrappers additionally
require a separately installed ``solve-field`` executable and local index files.
The other image-reduction tools do not require Astrometry.net.

Installation
^^^^^^^^^^^^

Install a release in an activated Python environment:

.. code-block:: bash

    python -m pip install astropop

For optional plotting support, use ``python -m pip install 'astropop[all]'``.

Conda environment
-----------------

To create an isolated environment with conda, then install Astropop and its
Python dependencies:

.. code-block:: bash

    conda create -n astropop -c conda-forge python=3.14 pip
    conda activate astropop
    python -m pip install 'astropop[all]'

Development checkout
--------------------

To work on the source and install the test and documentation dependencies:

.. code-block:: bash

    git clone https://github.com/sparc4-dev/astropop.git
    cd astropop
    python -m pip install -e '.[all,test,docs]'

See `installation and first steps
<https://astropop.readthedocs.io/en/latest/getting_started/installation.html>`_ for optional
external software, notebook environments and a first calculation. The package
metadata for the selected release determines its dependency requirements.

Citing
^^^^^^^^

|ADS|  |PASP|  |arXiv|  |ASCL|

An article was published in `Publications of the Astronomical Society of the Pacific, vol.131, n.996, pp.024501 <https://iopscience.iop.org/article/10.1088/1538-3873/aaecc2>`_,
which is the main reference to this work. If you do not have access to PASP, the preprint was uploaded to `arXiv:1811.01408 <https://arxiv.org/abs/1811.01408>`_.

Also, for latex citation, you can use the following BibTex:

.. code-block::

    @article{Campagnolo_2018,
    	doi = {10.1088/1538-3873/aaecc2},
	    url = {https://doi.org/10.1088%2F1538-3873%2Faaecc2},
	    year = 2018,
	    month = {dec},
	    publisher = {{IOP} Publishing},
	    volume = {131},
	    number = {996},
	    pages = {024501},
	    author = {Julio Cesar Neves Campagnolo},
	    title = {{ASTROPOP}: the {ASTROnomical} {POlarimetry} and Photometry Pipeline},
	    journal = {Publications of the Astronomical Society of the Pacific},
    }

Documentation
^^^^^^^^^^^^^

The user guides and API reference can be found at `astropop.readthedocs.io <https://astropop.readthedocs.io>`_

.. |GHAction Status| image:: https://github.com/sparc4-dev/astropop/actions/workflows/ci_workflows.yml/badge.svg
    :target: https://github.com/sparc4-dev/astropop/actions
    :alt: Astropop's Github CI Status

.. |Codecov Status| image:: https://codecov.io/gh/sparc4-dev/astropop/branch/main/graph/badge.svg?token=tzrOfWMhUb
    :target: https://codecov.io/gh/sparc4-dev/astropop
    :alt: Astropop's Codecov Coverage Status

.. |RTD Status| image:: https://readthedocs.org/projects/astropop/badge/?version=latest
    :target: https://astropop.readthedocs.io/en/latest/?badge=latest
    :alt: Astropop's Documentation Status

.. |Powered by Astropy|  image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org/
    :alt: Powered by AstroPy

.. |ADS|  image:: http://img.shields.io/badge/ADS-2019PASP..131b4501N-blue.svg?style=flat
    :target: https://ui.adsabs.harvard.edu/abs/2019PASP..131b4501N/abstract
    :alt: ADS Reference

.. |PASP| image:: http://img.shields.io/badge/PASP-pp.024501-blue.svg?style=flat
    :target: https://iopscience.iop.org/article/10.1088/1538-3873/aaecc2
    :alt: Publications of the Astronomy Society of the Pacific

.. |arXiv|  image:: http://img.shields.io/badge/arXiv-1811.01408-red.svg?style=flat
    :target: https://arxiv.org/abs/1811.01408
    :alt: arXiv preprint

.. |ASCL|  image:: https://img.shields.io/badge/ascl-1805.024-blue.svg?colorB=262255
    :target: http://ascl.net/1805.024
    :alt: ASCL register

.. |CODACY|  image:: https://app.codacy.com/project/badge/Grade/ab9d4647935d4b33aee0544b6957d7a7
    :target: https://www.codacy.com/gh/sparc4-dev/astropop/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=sparc4-dev/astropop&amp;utm_campaign=Badge_Grade
