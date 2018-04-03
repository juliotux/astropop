# AstroPoP

The (non) famous ASTROnomical POlarimetry and Photometry pipeline. Developed for work with IAGPOL polarimeter at Observatório Pico dos Dias (Brazil), but suitable to be used in other image polarimeters around the world.

# Important

Now, I'm transporting it from my personal python package [astrojc](https://github.com/juliotux/astrojc) and soon I will get a full functional version here.

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

Future...
