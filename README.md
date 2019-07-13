# AstroPoP

[![Coverage Status](https://coveralls.io/repos/github/juliotux/astropop/badge.svg?branch=master)](https://coveralls.io/github/juliotux/astropop?branch=master)
[![Build Status](https://travis-ci.org/juliotux/astrojc.svg?branch=master)](https://travis-ci.org/juliotux/astrojc)
[![Documentation Status](https://readthedocs.org/projects/astropop/badge/?version=latest)](https://astropop.readthedocs.io/en/latest/?badge=latest)

The (non) famous ASTROnomical POlarimetry and Photometry pipeline. Developed for work with IAGPOL polarimeter at Observatório Pico dos Dias (Brazil), but suitable to be used in other image polarimeters around the world.

# Features:

This software is intended to provide a full pipeline to reduce raw polarimetry and photometry data taken with common CCD telescope cameras. It can do:
- Create calibrate frames;
- Calibrate images using bias/flat/dark frames;
- Gain correction and in-processing image binnig;
- Cosmic ray extraction (astroscrappy);
- Align image sets;
- Aperture and (planned) PSF photometry;
- Calcite and (planned) polaroid polarimeters;
  - Automatic pairs of stars identification;
- Automatic photometry calibration using online catalogs.

A paper with the science verification results and the full methodology is in preparation and should be submitted soon.

# Requirements:
- Python 3.5 or above;
- [SEP (Source Extractor and Photometry)](https://sep.readthedocs.io/)
- [astropy](https://www.astropy.org/)
- [scipy stack (numpy, scipy, matplotlib)](https://www.scipy.org/)
- [cython](http://cython.org/)
- [astroquery](https://github.com/astropy/astroquery)
- [Local astrometry.net installation](https://astrometry.net)
- [scikit-image](http://scikit-image.org/)

# Documentation

Documentation (not complete yet) can be found at [astropop.readthedocs.io](https://astropop.readthedocs.io)
