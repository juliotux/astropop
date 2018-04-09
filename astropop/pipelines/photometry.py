# Licensed under a 3-clause BSD style license - see LICENSE.rst

class PhotometryScript(ReduceScript):
    def __init__(self, config=None):
        super(PhotometryScript, self).__init__(config=config)

    def run(self, name, **config):
        """Run this pipeline script"""
        product_dir = config['product_dir']
        night = config['night']
        phot_prod = os.path.join(product_dir,
                                 "{}_photometry_{}".format(night, name))
        s = [os.path.join(config['raw_dir'], i) for i in config['sources']]

        if config.get('astrojc_cal', True):
            calib_kwargs = {}
            for i in ('master_bias', 'master_flat', 'dark_frame', 'badpixmask',
                      'prebin', 'gain_key', 'gain', 'rdnoise_key',
                      'combine_method', 'combine_sigma', 'exposure_key',
                      'mem_limit', 'save_calib_path', 'combine_align_method',
                      'calib_dir', 'product_dir', 'remove_cosmics',
                      'bias_check_keys', 'flat_check_keys', 'dark_check_keys'):
                if i in config.keys():
                    calib_kwargs[i] = config[i]
            ccd = calib_science(s, **calib_kwargs)
        elif 'save_calib_path' in config.keys():
            ccd = process_list(os.path.basename, s)
            ccd = [os.path.join(config['save_calib_path'], i) for i in ccd]
            ccd = combine(ccd, method=config['combine_method'])
        else:
            ccd = combine(s, method=config['combine_method'])

        photkwargs = {}
        for i in ['ra_key', 'dec_key', 'gain_key', 'rdnoise_key',
                  'filter', 'plate_scale', 'photometry_type',
                  'psf_model', 'r', 'r_in', 'r_out', 'psf_niters',
                  'box_size', 'detect_fwhm', 'detect_snr', 'remove_cosmics',
                  'align_images', 'solve_photometry_type',
                  'montecarlo_iters', 'montecarlo_percentage',
                  'identify_catalog_file', 'identify_catalog_name',
                  'identify_limit_angle', 'science_catalog', 'science_id_key',
                  'science_ra_key', 'science_dec_key']:
            if i in config.keys():
                photkwargs[i] = config[i]

        t = process_calib_photometry(ccd, **photkwargs)

        hdus = []
        for i in [i for i in t.keys() if t[i] is not None]:
            header_keys = ['solve_photometry_type', 'plate_scale', 'filter']
            if i == 'aperture':
                header_keys += ['r', 'r_in', 'r_out', 'detect_fwhm',
                                'detect_snr']
            elif i == 'psf':
                header_keys += ['psf_model', 'box_size', 'psf_niters']

            if config.get('solve_photometry_type', None) == 'montecarlo':
                header_keys += ['montecarlo_iters', 'montecarlo_percentage']

            if config.get('identify_catalog_name', None) is not None:
                header_keys += ['identify_catalog_name',
                                'identify_limit_angle']

            hdu = fits.BinTableHDU(t[i], name="{}_photometry".format(i))
            for k in header_keys:
                if k in config.keys():
                    v = config[k]
                    key = 'hierarch astrojc {}'.format(k)
                    if check_iterable(v):
                        hdu.header[key] = ','.join([str(m) for m in v])
                    else:
                        hdu.header[key] = v
            hdus.append(hdu)

            best = Table(dtype=t[i].dtype)
            for group in t[i].group_by('star_index').groups:
                b = np.argmax(group['flux']/group['flux_error'])
                best.add_row(group[b])
            best['snr'] = best['flux']/best['flux_error']

            hdu = fits.BinTableHDU(best, name="{}_best_snr".format(i))
            for k in header_keys:
                if k in config.keys():
                    v = config[k]
                    key = 'hierarch astrojc {}'.format(k)
                    if check_iterable(v):
                        hdu.header[key] = ','.join([str(m) for m in v])
                    else:
                        hdu.header[key] = v
            hdus.append(hdu)

        mkdir_p(product_dir)
        hdulist = fits.HDUList([ccd, *hdus])
        hdulist.writeto(phot_prod, overwrite=True)
