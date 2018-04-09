# Licensed under a 3-clause BSD style license - see LICENSE.rst


class CalibScript(ReduceScript):
    def __init__(self, config=None):
        super(CalibScript, self).__init__(config=config)

    def run(self, name, **config):
        for k in config.keys():
            batch_key_replace(config, k)
        s = [os.path.join(config['raw_dir'], i) for i in config['sources']]

        calib_kwargs = {}
        for i in ['calib_type', 'master_bias', 'master_flat', 'dark_frame',
                  'badpixmask', 'prebin', 'gain_key', 'rdnoise_key', 'gain',
                  'combine_method', 'combine_sigma', 'exposure_key',
                  'mem_limit', 'calib_dir',
                  'bias_check_keys', 'flat_check_keys', 'dark_check_keys']:
            if i in config.keys():
                calib_kwargs[i] = config[i]

        if 'result_file' not in config.keys():
            outname = name
        else:
            outname = config['result_file']

        mkdir_p(config['calib_dir'])
        return create_calib(s, outname, **calib_kwargs)
