# Licensed under a 3-clause BSD style license - see LICENSE.rst
# Important: this is the default keys of the configuration files!
'''
#General instrument config
ra_key                  # Right Ascension header key
dec_key                 # Declination header key
gain_key                # Instrument gain header key (e-/adu)
rdnoise_key             # Read noise header key
exposure_key            # Key of the exposure time of the image
jd_key                  # Julian day header key
time_key                # Time header key will be used
time_key_type           # Type of time stored in time_key: 'jd', 'isot', 'mjd'
retarder_key            # Polarimetry retarder position key
retarder_type           # type of the retarder: 'half', 'quarter'
retarder_rotation       # rotation of each position of the retarder, in deg
                        # retarder, like 16
retarder_direction      # direction of the rotation of the retarder.
                        # can be 'cw', 'ccw' or -1, +1

#Per product config
pipeline                # pipeline name: 'photometry', 'lightcurve',
                        # 'polarimetry', 'calib'
calib_type              # 'flat', 'bias', 'dark', 'science'
result_file             # file to store the results
save_calib_path         # alternative folder to save calibrated images
sources                 # source files to process
source_ls_pattern       # filename pattern to load sources from ls
prebin                  # number of pixels to bin
filter                  # filter of the image
master_bias             # bias file to correct the image
master_flat             # flat file to correct the image
badpixmask              # bad pixel mas file. pipeline=calib will created it
plate_scale             # the plate scale of the final image
gain                    # manual set of the gain of the image,
                        # if gain_key is not set

#pipeline config
raw_dir                 # directory containing raw data
product_dir             # directory to store product data
calib_dir               # directory to store calib data
photometry_type         # aperture or psf or both
psf_model               # model name of the psf: 'gaussian' or 'moffat'
r                       # aperture radius or list of apertures (the best snr
                        # will be used)
r_in                    # internal sky subtract radius
r_out                   # external sky subtract radius
psf_niters              # number of iterations of daofind
box_size                # box size to extract and fit stars with psf
detect_fwhm             # fwhm of daofind detect
detect_snr              # minimum signal to noise ratio to detect stars
remove_cosmics          # bool: remove cosmic rays with astroscrappy
plot                    # plot results
align_images            # if calculate and apply shift in lightcurve and
                        # polarimetry
solve_photometry_type   # 'montecarlo', 'median', 'average'
match_pairs_tolerance   # maximum tolerance for matching pairs for polarimetry,
                        # in pixels
montecarlo_iters        # number of iterations of montecarlo magnitude
montecarlo_percentage   # percentage of star in each montecarlo magnitude
identify_catalog_file   # json file containing the catalog definitions
identify_catalog_name   # name of the definition dataset in json file
identify_limit_angle    # maximum distance to identify stars in arcseconds
science_catalog         # file catalog with science stars
science_id_key          # column name of the id of the star
science_ra_key          # column name of the RA of the star
science_dec_key         # column name of the DEC of the star
combine_method          # method for combine images. Can be 'median', 'average'
                        # or 'sum'
combine_sigma           # sigma of sigma_clip when combining images
combine_align_method    # if calculate and apply shift in combine: 'fft', 'wcs'
mem_limit               # maximum memory limit

# In the values, {key_name} will be formated to the key_name value
'''

class MasterReduceScript(ReduceScript):
    def __init__(self, config=None):
        super(MasterReduceScript, self).__init__(config=config)

    def run(self, name, **config):
        if 'sources' in config.keys() and 'source_ls_pattern' in config.keys():
            logger.warn('sources and sources_ls_pattern given. Using sources.')
        elif 'sources_ls_pattern' in config.keys():
            ls_pat = os.path.join(config['raw_dir'],
                                  config['sources_ls_pattern'])
            fs = glob.glob(ls_pat)
            config['sources'] = [os.path.basename(i) for i in sorted(fs)]
            if len(config['sources']) == 0:
                raise FileNotFoundError("Could not determine sources."
                                        " glob pattern: {}".format(ls_pat))

        config['sources'] = [i for i in config['sources']
                             if i not in config['exclude_images']]

        logger.debug("Product {} config:{}".format(name, str(config)))
        if 'pipeline' not in config.keys():
            raise ValueError('The config must specify what pipeline will be'
                             ' used!')
        if config['pipeline'] == 'photometry':
            p = PhotometryScript()
        elif config['pipeline'] == 'polarimetry':
            p = PolarimetryScript()
        elif config['pipeline'] == 'calib':
            p = CalibScript()
        elif config['pipeline'] == 'lightcurve':
            logger.error('lightcurve pipeline not implemented yet')
        else:
            raise ValueError('Pipeline {} not'
                             ' supported.'.format(config['pipeline']))

        p.run(name, **config)
