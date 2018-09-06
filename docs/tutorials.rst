Some tutorials can be found here! This page will have more examples soon.

Using The Pipelines
===================

The use of curently implemented pipelines is simple. Most of them are implemented as simple command line scripts, installed automaticaly with the code. For now, pipelines 3 pipelines are available: AAVSO-BSM, ROBO40 (Obesrvatório Pico dos Dias), IMPACTON telescope and the `astropop` config-based pipeline.

AAVSO-BSM pipeline
------------------

To reduce the AAVSO Bright Star Monitor program images, you can use the `aavsobsm_process.py` script installed with the package. The arguments are quite simple::

   aavsobsm_process.py -h
   Usage: aavsobsm_process.py [options] raw_dir [raw_dir2, ...]

   Options:
      -h, --help            show this help message and exit
      -v, --verbose         Enable 'DEBUG' output in python log
      -a, --astrometry      Enable astrometry solving of stacked images with
                            astrometry.net
      -n FILE, --science-catalog=FILE
                            ASCII catalog to identify science stars. Has to be
                            astropy's table readable with columns ID, RA, DEC
      -l FILE, --save-log=FILE
                           Save log to FILE. If '%date' value, automatic name
                           based on date will be created.
      -d FOLDER, --dest=FOLDER
                           Reduced images (and created calib frames) will be
                           saved at inside FOLDER


For a pratical test:

1 - grab some raw data from AAVSOnet ftp site and save them to a folder::

   ftp://ftp.aavsonet.aavso.org/

2 - execute the script::

   aavsobsm_process.py -av -l %date -n myultrasecretcatalog.csv -d ~/AAVSOBSM-Reduced /folder/with/the/data

Creating and Reducing Polarization Datasets
===========================================

.. toctree::
   :maxdepth:2

   tutorials/astropop_polarization_mode.ipnb
