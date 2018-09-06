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
   
ASTROPOP config-based pipeline
------------------------------

For more complex instruments, like IAGPOL, a config file have to be passed to to a script, which will be read and give to the code the informations to be reduce the data. But don't worry, you can also load some default files with como parameters, also define local parameters for all products in the config!

default_config.json (This file just need to be written once!)::

  {
      "raw_dir" : "/data/{night}",
      "calib_dir" : "/data/calib/{night}",
      "product_dir" : "/data/reduced/{night}",
      "config_dir" : "/home/user/reducing_configs",
      "mem_limit" : 1e9,
      "astropop_cal" : true,
      "astropop_pol" : true,
      "calculate_mode" : "both",
      "pccdpack" : false,
      "match_pairs_tolerance" : 3,
      "photometry_type" : "aperture",
      "psf_model" : "gaussian",
      "psf_niters" : 5,
      "box_size" : 35,
      "r" : "auto",
      "r_ann" : [20, 30],
      "plate_scale" : [0.2, 0.8],
      "r_find_best" : false,
      "detect_fwhm" : 3,
      "detect_snr" : 5,
      "plot" : true,
      "solve_photometry_type" : "montecarlo",
      "montecarlo_iters" : 200,
      "montecarlo_percentage" : 0.1,
      "identify_catalog_file" : "{config_dir}/photometry_catalogs.json",
      "limit_angle" : "2 arcsec",
      "science_catalog" : "{config_dir}/objects.dat",
      "science_id_key" : "Star",
      "science_ra_key" : "RA",
      "science_dec_key" : "Dec",
      "science_format": "ascii.tab",
      "combine_method" : null,
      "combine_align_method" : "fft",
      "exclude_images"  : [],
      "return_filename" : false,
      "retarder_type" : "half",
      "remove_cosmics" : true,
      "exclude_images": [],
      "plate_scale" : 0.32
   }
   
opd_instruments.json (This file needs to be written once per instrument too!)::
   
   {
      "ra_key": "RA",
      "dec_key": "DEC",
      "jd_key": "JD",
      "rdnoise_key": "RDNOISE",
      "exposure_key": "EXPOSURE",
      "gain_key": "GAIN",
      "time_key": "DATE-OBS",
      "time_key_type": "isot",
      "airmass_key": "AIRMASS",
      "retarder_key": "LAM-POS",
      "filter_key": "FIL-POS",
      "retarder_rotation": 22.5,
      "retarder_direction": 1,
      "analyzer_key": "ANALIZER",
      "bias_check_keys": ["ACQMODE", "READMODE", "IMGRECT", "SUBRECT", "READTIME",
                          "GAIN", "EMREALGN", "VSHIFT", "PREAMP", "SERNO",
                          "CAMGAIN"],
      "flat_check_keys": ["ACQMODE", "READMODE", "IMGRECT", "SUBRECT", "READTIME",
                          "GAIN", "EMREALGN", "VSHIFT", "PREAMP", "SERNO",
                          "CAMGAIN"],
      "dark_check_keys": ["ACQMODE", "READMODE", "IMGRECT", "SUBRECT", "READTIME",
                          "GAIN", "EMREALGN", "VSHIFT", "PREAMP", "SERNO",
                          "CAMGAIN"],
      "image_north_direction": "right",
      "image_flip": null
  }
   
reduce_set.json::

   {
      "__preload__": {
         "night": "15mai21",
         "preload_config_files" : [
            "{config_dir}/opd_instruments.json"
         ],
      },
      "mbias.fits": {
         "calib_type": "bias",
         "pipeline": "calib",
         "sources_ls_pattern": "bias_*.fits.gz",
         "combine_align_method": null
      },
      "mflat_v.fits": {
         "calib_type": "flat",
         "pipeline": "calib",
         "sources_ls_pattern": "flatv_*.fits.gz",
         "filter": "V",
         "badpixmask": "mbadpix_v.fits",
         "master_bias": "mbias.fits",
         "combine_align_method": null
      },
      "HD110984_V.fits": {
         "pipeline": "polarimetry"
         "sources_ls_pattern": "HD110984V_*.fits.gz",
         "filter": "V",
         "master_bias": "mbias.fits",
         "master_flat": "mflat_v.fits",
         "identify_catalog_name": "UCAC4",
         "brightest_star_ra": "12 46 44.8",
         "brightest_star_dec": "-61 11 11.5"
      },
      "OTHER_STAR.fits":{
         ...
      },
      ...
   }

All this parameters will be better documented together with the API documentation.

All the configs inside `__preload__` environment will be set for all products in the file. Also, any document in the "preload_config_files" will also be loaded and its properties set to the reductions. Them, once you organize this config file (what is faster then you think), you just need to execute::
   
   astropop_pipeline.py -i default_config.json -v -l %date reduce_set.json
   
And all products there will be reduced.

The command line paramenters are described if you execute `astropop_pipeline.py --help`

But, why this? Because OPD do not have so standardized keywords and header information in images, what make impossible to guess what type of data is based on header (like happens to AAVSO-BSM). This is the best solution we found.

Creating and Reducing Polarization Datasets
===========================================

.. toctree::
   :maxdepth:2

   tutorials/astropop_polarization_mode.ipnb
