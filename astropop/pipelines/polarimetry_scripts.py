import os
import shutil
import six
import time
import copy
from tempfile import mkdtemp
import numpy as np

from astropy.table import Table, Column, vstack

from .photometry_scripts import solve_photometry, process_photometry
from .astrometry_scripts import solve_astrometry, identify_stars
from ..fits_utils import check_hdu
from ..photometry import aperture_photometry
from ..astrometry import wcs_from_coords
from ..logger import logger
from ..polarimetry import pccdpack_wrapper as pccd
from ..polarimetry.calcite_polarimetry import (calculate_polarimetry,
                                               estimate_dxdy,
                                               match_pairs)
from ..py_utils import process_list, check_iterable


def run_pccdpack(image_set, retarder_type=None, retarder_key=None,
                 retarder_rotation=22.5, retarder_direction=None,
                 save_calib_path=None, r=np.arange(1, 21, 1),
                 r_in=60, r_out=70, gain_key=None, rdnoise_key=None,
                 wcs=None, **kwargs):
    files = []
    dtmp = mkdtemp(prefix='pccdpack')

    for i in range(len(image_set)):
        if isinstance(image_set[i], six.string_types):
            files.append(os.path.join(dtmp,
                                      os.path.basename(image_set[i])))
            try:
                shutil.copy(image_set[i], files[-1])
            except Exception:
                pass
        else:
            name = os.path.join(dtmp, "image{:02d}.fits".format(i))
            im = check_hdu(image_set[i])
            im.writeto(name)
            logger.debug("image {} saved to {}".format(i, name))
            files.append(name)

    script = pccd.create_script(result_dir=dtmp, image_list=files,
                                star_name='object', apertures=r, r_ann=r_in,
                                r_dann=r_out-r_in, readnoise_key=rdnoise_key,
                                retarder=retarder_type, auto_pol=True)

    print('\n\nExecute the following script:\n-----------------------------\n')
    print(script)
    time.sleep(0.5)
    print('------------------------------------\n')
    input('Press Enter when finished!')

    out_table = pccd.read_out(os.path.join(dtmp, 'object.out'),
                              os.path.join(dtmp, 'object.ord'))
    dat_table = Table.read(os.path.join(dtmp, 'dat.001'),
                           format='ascii.no_header')
    log_table = pccd.read_log(os.path.join(dtmp, 'object.log'),
                              return_table=True)

    x, y = out_table['x0'], out_table['x0']
    data = check_hdu(files[0])
    ft = aperture_photometry(data.data, x=x, y=y, r=5)

    if wcs is None:
        try:
            if kwargs.get('astrometry_calib', True):
                astkwargs = {}
                for i in ['ra_key', 'dec_key', 'plate_scale']:
                    if i in kwargs.keys():
                        astkwargs[i] = kwargs[i]
                wcs = solve_astrometry(data.header, ft, data.data.shape,
                                       **astkwargs)
        except Exception as e:
            for i in ['brightest_star_ra', 'brightest_star_dec', 'plate_scale',
                      'image_north_direction']:
                if i not in kwargs.keys():
                    raise e
            logger.info('Guessing wcs from brightest star coordinates.')
            bright = ft.sort('flux')[-1]
            wcs = wcs_from_coords(bright['xo'][0], bright['yo'][0],
                                  kwargs['brightest_star_ra'],
                                  kwargs['brightest_star_dec'],
                                  kwargs['plate_scale'],
                                  kwargs['image_north_direction'],
                                  kwargs['image_flip'])

    if wcs is not None:
        idkwargs = {}
        for i in ['identify_catalog', 'filter',
                  'limit_angle', 'science_catalog']:
            if i in kwargs.keys():
                idkwargs[i] = kwargs[i]
        ids = identify_stars(x, y, wcs, **idkwargs)
        out_table['cat_id'] = ids['cat_id']
        out_table['sci_id'] = ids['sci_id']
        out_table['ra'] = ids['ra']
        out_table['dec'] = ids['dec']

    shutil.rmtree(dtmp)

    return out_table, dat_table, log_table, wcs


def find_pairs(x, y, match_pairs_tolerance=2, delta_x=None, delta_y=None):
    if delta_x is not None and delta_y is not None:
        dx, dy = delta_x, delta_y
    else:
        dx, dy = estimate_dxdy(x, y)
    pairs = match_pairs(x, y, dx, dy, tolerance=match_pairs_tolerance)

    o_idx = pairs['o']
    e_idx = pairs['e']

    tmp = Table()
    tmp['xo'] = x[o_idx]
    tmp['yo'] = y[o_idx]
    tmp['xe'] = x[e_idx]
    tmp['ye'] = y[e_idx]

    return tmp, pairs


def _do_polarimetry(phot_table, psi, retarder_type, pairs, positions=None):
    """Calculate the polarimetry of a given photometry table.

    phot_tables is a list of tables containing ['flux', 'flux_error']
    keys.
    """
    ph = phot_table

    if 'flux' not in ph[0].colnames or 'flux_error' not in ph[0].colnames:
        raise ValueError('Table for polarimetry must contain "flux" and'
                         ' "flux_error" keys.')

    tmp = Table()

    def _process(idx):
        f = 'flux'
        fe = 'flux_error'
        o = np.array([ph[j][f][pairs[idx]['o']] for j in range(len(ph))])
        e = np.array([ph[j][f][pairs[idx]['e']] for j in range(len(ph))])
        oe = np.array([ph[j][fe][pairs[idx]['o']] for j in range(len(ph))])
        ee = np.array([ph[j][fe][pairs[idx]['e']] for j in range(len(ph))])
        res = calculate_polarimetry(o, e, psi, retarder=retarder_type,
                                    o_err=oe, e_err=ee, positions=positions,
                                    filter_negative=True)
        for k in res.keys():
            dt = 'f4'
            if k not in tmp.colnames:
                shape = (1) if k != 'z' else (len(psi))
                tmp.add_column(Column(name=k, dtype=dt, shape=shape,
                                      length=len(pairs)))
                if k not in ['sigma_theor', 'reduced_chi2']:
                    tmp.add_column(Column(name='{}_error'.format(k),
                                          dtype=dt, shape=shape,
                                          length=len(pairs)))
            if k in ['sigma_theor', 'reduced_chi2']:
                tmp[k][idx] = res[k]
            elif k == 'z':
                tmp[k][idx] = res[k]['value']
                tmp['{}_error'.format(k)][idx] = res[k]['sigma']
            else:
                tmp[k][idx] = res[k]['value']
                tmp['{}_error'.format(k)][idx] = res[k]['sigma']

    for i in range(len(pairs)):
        _process(idx=i)

    return tmp


def process_polarimetry(image_set, align_images=True, retarder_type=None,
                        retarder_key=None, match_pairs_tolerance=1.0,
                        retarder_rotation=22.5, retarder_direction=None,
                        wcs=None, **kwargs):
    """Process the photometry and polarimetry of a set of images.

    kwargs are the arguments for the following functions:
    process_photometry, _solve_photometry
    """
    s = process_list(check_hdu, image_set)
    result = {'aperture': None, 'psf': None}

    sources = aperture_photometry(s[0].data, r=5,
                                  detect_fwhm=kwargs['detect_fwhm'],
                                  detect_snr=kwargs['detect_snr'])

    logger.info('Identified {} sources'.format(len(sources)))

    _tolerance = match_pairs_tolerance
    res_tmp, pairs = find_pairs(sources['x'], sources['y'],
                                match_pairs_tolerance=_tolerance)
    if len(pairs) == 0:
        if kwargs.get('delta_x') and kwargs.get('delta_y'):
            dx = kwargs.get('delta_x')
            dy = kwargs.get('delta_y')
            logger.info('Not pairs found in automatic routine. Using given '
                        'delta_x={} delta_y={}'.format(dx, dy))
            res_tmp, pairs = find_pairs(sources['x'], sources['y'],
                                        match_pairs_tolerance=_tolerance,
                                        delta_x=dx, delta_y=dy)
    if len(pairs) > 0:
        logger.info('Matched {} pairs of sources'.format(len(pairs)))
    else:
        raise RuntimeError('No pairs of stars found on this set.')

    if wcs is None:
        try:
            if kwargs.get('astrometry_calib', True):
                wcs = solve_astrometry(sources[pairs['o']], s[0].header,
                                       s[0].data.shape,
                                       ra_key=kwargs['ra_key'],
                                       dec_key=kwargs['dec_key'],
                                       plate_scale=kwargs['plate_scale'])
        except Exception as e:
            for i in ['brightest_star_ra', 'brightest_star_dec', 'plate_scale',
                      'image_north_direction']:
                if i not in kwargs.keys():
                    raise e
            # bright = sources[pairs['o']].sort('flux')[-1]
            bright = sources[pairs['o']]
            bright.sort('flux')
            wcs = wcs_from_coords(bright[-1]['x'], bright[-1]['y'],
                                  kwargs['brightest_star_ra'],
                                  kwargs['brightest_star_dec'],
                                  kwargs['plate_scale'],
                                  kwargs['image_north_direction'],
                                  kwargs['image_flip'])

    if wcs is not None:
        idkwargs = {}
        for i in ['identify_catalog_file', 'identify_catalog_name', 'filter',
                  'identify_limit_angle', 'science_catalog',
                  'science_id_key', 'science_ra_key', 'science_dec_key']:
            if i in kwargs.keys():
                idkwargs[i] = kwargs[i]
        ids = identify_stars(Table([res_tmp['xo'], res_tmp['yo']],
                                   names=('x', 'y')), wcs,
                             **idkwargs)
        if 'sci_id' in ids.colnames:
            if not np.array(ids['sci_id'] != '').any():
                logger.warn('No science stars found')

    ids['x0'] = res_tmp['xo']
    ids['y0'] = res_tmp['yo']
    ids['x1'] = res_tmp['xe']
    ids['y1'] = res_tmp['ye']

    solvekwargs = {}
    for i in ['montecarlo_iters', 'montecarlo_percentage',
              'solve_photometry_type']:
        if i in kwargs.keys():
            solvekwargs[i] = kwargs.get(i)

    try:
        ret = [int(i.header[retarder_key]) for i in s]
    except ValueError:
        # Some positions may be in hexa
        ret = [int(i.header[retarder_key], 16) for i in s]

    if retarder_direction == 'cw':
        retarder_direction = -1
    if retarder_direction == 'ccw':
        retarder_direction = 1

    psi = np.array(ret)*retarder_rotation*retarder_direction

    solve_to_result = {'aperture': False, 'psf': False}
    solves = {}
    if not check_iterable(kwargs['r']):
        kwargs['r'] = [kwargs['r']]
    apkwargs = {}
    phot_type = kwargs['photometry_type']
    for i in ['detect_fwhm', 'detect_snr', 'box_size',
              'r_in', 'r_out', 'r_find_best', 'psf_model', 'psf_niters']:
        if i in kwargs.keys():
            apkwargs[i] = kwargs.get(i)
    for i in ['aperture', 'psf']:
        if i == phot_type or phot_type == 'both':
            do = True
        else:
            do = False

        if i == 'psf':
            rl = ['psf']
        else:
            rl = kwargs['r']

        if do:
            for ri in rl:
                logger.info('Processing polarimetry for aper:{}'.format(ri))
                napkwargs = copy.copy(apkwargs)
                napkwargs['photometry_type'] = i
                ph = process_list(process_photometry, s, x=sources['x'],
                                  y=sources['y'],
                                  r=ri, **napkwargs)
                ph = [Table([j['flux'], j['flux_error']]) for j in ph]
                solve_to_result[i] = True
                ap = _do_polarimetry(ph, psi,
                                     retarder_type=retarder_type,
                                     pairs=pairs,
                                     positions=ret)
                if wcs is not None:
                    tmp = solve_photometry(ap, cat_mag=ids['cat_mag'],
                                           **solvekwargs)
                    ap['mag'] = tmp['mag']
                    ap['mag_err'] = tmp['mag_err']

                solves[ri] = ap

    for ri in solves.keys():
        t = solves[ri]
        if ri == 'psf':
            ri = np.nan
            i = 'psf'
        else:
            i = 'aperture'

        nt = Table()
        nt.add_column(Column(data=np.arange(len(t)),
                             name='star_index', dtype='i4'))
        nt.add_column(Column(data=np.array([ri]*len(t)),
                             name='aperture', dtype='f4'))
        if ids is not None:
            for c in ids.itercols():
                nt.add_column(c)
        for c in t.itercols():
            nt.add_column(c)

        if result[i] is None:
            result[i] = nt
        else:
            result[i] = vstack([result[i], nt])

    return result, wcs, ret
