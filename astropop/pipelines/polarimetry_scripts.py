import os
import shutil
import six
import time
from tempfile import mkdtemp
import numpy as np

from astropy.table import Table, Column, vstack, hstack
from astropy.stats import sigma_clip

from .photometry_scripts import solve_photometry
from .astrometry_scripts import solve_astrometry, identify_stars
from ..fits_utils import check_hdu
from ..photometry import (aperture_photometry, starfind, background,
                          process_photometry)
from ..astrometry import wcs_from_coords
from ..logger import logger
from ..polarimetry import pccdpack_wrapper as pccd
from ..polarimetry.calcite_polarimetry import (calculate_polarimetry,
                                               estimate_dxdy,
                                               match_pairs,
                                               estimate_normalize)
from ..image_processing.register import hdu_shift_images
from ..image_processing.imarith import imcombine
from ..py_utils import process_list, check_iterable


def run_pccdpack(image_set, retarder_type=None, retarder_key=None,
                 retarder_rotation=22.5, retarder_direction=None,
                 save_calib_path=None, r=np.arange(1, 21, 1),
                 r_ann=(50, 60), gain_key=None, rdnoise_key=None,
                 wcs=None, **kwargs):
    files = []
    dtmp = mkdtemp(prefix='pccdpack')
    if r == 'auto':
        d = check_hdu(image_set[0]).data
        bkg, rms = background(d, box_size=32, filter_size=3,
                              global_bkg=False)

        s = starfind(d, 5, bkg, rms, 4)
        p = aperture_photometry(d, s['x'], s['y'], r='auto', r_ann=None)
        fwhm = p.meta['fwhm']
        r = 0.8*fwhm
        r_in = int(round(4*r, 0))
        r_out = int(max(r_in+10, round(6*r, 0)))  # Ensure a dannulus geq 10
        r_ann = (r_in, r_out)
        logger.debug("using auto radius for polarimetry: r={}, r_ann={}"
                     .format(r, r_ann))
    r_in, r_out = sorted(r_ann)

    for i in range(len(image_set)):
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

    # fix iraf-python convention
    x, y = out_table['x0']-1, out_table['y0']-1
    data = check_hdu(files[0])
    ft = aperture_photometry(data.data, x=x, y=y, r='auto')
    ft = hstack([ft, out_table])

    if wcs is None and kwargs.get('astrometry_calib', True):
        try:
            astkwargs = {}
            for i in ['ra_key', 'dec_key', 'plate_scale']:
                if i in kwargs.keys():
                    astkwargs[i] = kwargs[i]
            wcs = solve_astrometry(ft, data.header, data.data.shape,
                                   **astkwargs)
        except Exception as e:
            for i in ['brightest_star_ra', 'brightest_star_dec', 'plate_scale',
                      'image_north_direction']:
                if i not in kwargs.keys():
                    raise e
            logger.info('Guessing wcs from brightest star coordinates.')
            bright = ft.copy()
            bright.sort('flux')
            wcs = wcs_from_coords(bright['x'][-1], bright['y'][-1],
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

    logger.info("Finding pairs with dx,dy = ({}, {})".format(dx, dy))
    pairs = match_pairs(x, y, dx, dy, tolerance=match_pairs_tolerance)

    o_idx = pairs['o']
    e_idx = pairs['e']

    tmp = Table()
    tmp['xo'] = x[o_idx]
    tmp['yo'] = y[o_idx]
    tmp['xe'] = x[e_idx]
    tmp['ye'] = y[e_idx]

    return tmp, pairs


def _do_polarimetry(phot_table, psi, retarder_type, pairs, positions=None,
                    calculate_mode='sum'):
    """Calculate the polarimetry of a given photometry table.

    phot_tables is a list of tables containing ['flux', 'flux_error']
    keys.
    """
    ph = phot_table

    if 'flux' not in ph[0].colnames or 'flux_error' not in ph[0].colnames:
        raise ValueError('Table for polarimetry must contain "flux" and'
                         ' "flux_error" keys.')

    tmp = Table()

    def _get_fluxes(idx):
        f = 'flux'
        fe = 'flux_error'
        io = pairs[idx]['o']
        ie = pairs[idx]['e']
        o = np.array([ph[j][f][io] for j in range(len(ph))])
        e = np.array([ph[j][f][ie] for j in range(len(ph))])
        oe = np.array([ph[j][fe][io] for j in range(len(ph))])
        ee = np.array([ph[j][fe][ie] for j in range(len(ph))])
        return o, e, oe, ee

    # process a global normalization based on bright stars (mainly), and sigma
    # clipped.
    # ks = np.zeros(len(pairs))
    # tf = np.zeros(len(pairs))
    # ks.fill(np.nan)
    # for i in range(len(pairs)):
    #     o, e, oe, ee = _get_fluxes(i)
    #     if retarder_type == 'half':
    #         npos = 4
    #     elif retarder_type == 'quarter':
    #         npos = 8
    #     else:
    #         npos = 4
    #     ks[i] = estimate_normalize(o, e, positions, npos)
    #     tf[i] = (np.sum(o + e))/np.sum(oe + ee)
    #     if not np.isfinite(ks[i]):
    #         tf[i] = 0
    # sclip = sigma_clip(ks)
    # mask = ~sclip.mask & (tf[i] != 0)
    # k = np.average(ks[np.where(mask)], weights=tf[np.where(mask)])

    def _process(idx):
        o, e, oe, ee = _get_fluxes(idx)
        res = calculate_polarimetry(o, e, psi, retarder=retarder_type,
                                    o_err=oe, e_err=ee, positions=positions,
                                    filter_negative=True, mode=calculate_mode)
        for k in res.keys():
            dt = 'f4'
            if k not in tmp.colnames:
                shape = (1) if k != 'z' else (len(psi))
                tmp.add_column(Column(name=k, dtype=dt, shape=shape,
                                      length=len(pairs)))
                if k not in ['sigma_theor', 'reduced_chi2', 'k']:
                    tmp.add_column(Column(name='{}_error'.format(k),
                                          dtype=dt, shape=shape,
                                          length=len(pairs)))
            if k in ['sigma_theor', 'reduced_chi2', 'k']:
                tmp[k][idx] = res[k]
            elif k == 'z':
                tmp[k][idx] = res[k]['value']
                tmp['{}_error'.format(k)][idx] = res[k]['sigma']
            else:
                tmp[k][idx] = res[k]['value']
                tmp['{}_error'.format(k)][idx] = res[k]['sigma']

    for i in range(len(pairs)):
        _process(i)

    return tmp


def process_polarimetry(image_set, align_images=True, retarder_type=None,
                        retarder_key=None, match_pairs_tolerance=1.0,
                        retarder_rotation=22.5, retarder_direction=None,
                        wcs=None, calculate_mode='sum', detect_snr=5.0,
                        photometry_type='aperture', **kwargs):
    """Process the photometry and polarimetry of a set of images.

    kwargs are the arguments for the following functions:
    process_photometry, _solve_photometry

    In old_mode, the retarder positions are assumed to be in consecutive
    filenames. Except, the `retarder_key` will give informations about the
    retarder position.
    """
    s = process_list(check_hdu, image_set)

    if align_images:
        s = hdu_shift_images(s)

    if 'rdnoise_key' in kwargs:
        err = s[0].header.get(kwargs['rdnoise_key'], None)

    # identify sources in the summed image to avoid miss a beam
    ss = imcombine(s, method='average')
    bkg, rms = background(ss.data, box_size=32, filter_size=3,
                          global_bkg=False)
    # bigger limits to handle worst data
    # as we are using the summed image, the detect_snr needs to go up by sqrt(n)
    sources = starfind(ss.data, detect_snr*np.sqrt(len(s)), bkg, rms, fwhm=5,
                       round_limit=(-2.0, 2.0), sharp_limit=(0.1, 1.0))
    fwhm = sources.meta['fwhm']
    if kwargs.get('r', 'auto') == 'auto':
        r = 0.8*fwhm
        kwargs['r'] = r
        r_in = int(round(4*r, 0))
        r_out = int(max(r_in+10, round(6*r, 0)))  # Ensure a dannulus geq 10
        r_ann = (r_in, r_out)
        kwargs['r_ann'] = r_ann
        logger.debug("using auto radius for polarimetry: r={}, r_ann={}"
                     .format(r, r_ann))
    logger.info('Identified {} sources'.format(len(sources)))
    sources = aperture_photometry(s[0].data, sources['x'], sources['y'],
                                  r='auto', r_ann=None)

    _tolerance = match_pairs_tolerance
    res_tmp, pairs = find_pairs(sources['x'], sources['y'],
                                match_pairs_tolerance=_tolerance)
    if len(pairs) == 0:
        if 'delta_x' in kwargs.keys() and 'delta_y' in kwargs.keys():
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

    if wcs is None and kwargs.get('astrometry_calib', True):
        try:
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
            bright = sources[pairs['o']]
            bright.sort('flux')
            wcs = wcs_from_coords(bright[-1]['x'], bright[-1]['y'],
                                  kwargs['brightest_star_ra'],
                                  kwargs['brightest_star_dec'],
                                  kwargs['plate_scale'],
                                  kwargs['image_north_direction'],
                                  kwargs['image_flip'])

    if wcs is not None:
        idkwargs = {'identify_catalog': kwargs.get('identify_catalog'),
                    'filter': kwargs.get('filter'),
                    'limit_angle': kwargs.get('limit_angle'),
                    'science_catalog': kwargs.get('science_catalog')}
        ids = identify_stars(res_tmp['xo'], res_tmp['yo'], wcs, **idkwargs)
        ids = Table(ids)
        if 'sci_id' in ids.colnames:
            if not np.array(ids['sci_id'] != '').any():
                logger.warn('No science stars found')
    else:
        ids = Table()

    ids['x0'] = res_tmp['xo']
    ids['y0'] = res_tmp['yo']
    ids['x1'] = res_tmp['xe']
    ids['y1'] = res_tmp['ye']

    t = Table()
    t['star_index'] = np.arange(len(pairs))
    ids = hstack([t, ids])
    del t

    solvekwargs = {'montecarlo_iters': kwargs.get('montecarlo_iters', 200),
                   'montecarlo_percentage': kwargs.get('montecarlo_percentage',
                                                       0.1),
                   'solve_photometry_type': kwargs.get('solve_photometry_type',
                                                       'montecarlo')}

    ret = None
    if retarder_key is not None:
        try:
            ret = [int(i.header[retarder_key]) for i in s]
        except ValueError:
            # Some positions may be in hexa
            ret = [int(i.header[retarder_key], 16) for i in s]
        except KeyError:
            pass

    if ret is None:
        # If retarder_key not present, assume images are ordered, just
        # like pccdpack
        logger.warn('retarder_key not found. Assuming images are ordered.')
        ret = np.arange(len(s))

    if retarder_direction == 'cw':
        retarder_direction = -1
    if retarder_direction == 'ccw':
        retarder_direction = 1

    psi = np.array(ret)*retarder_rotation*retarder_direction

    apkwargs = {'box_size': kwargs.get('box_size', 25),
                'psf_model': kwargs.get('psf_model', 'moffat'),
                'psf_niters': kwargs.get('psf_niters', 5)}
    if 'r_in' in kwargs and 'r_out' in kwargs:
        apkwargs['r_ann'] = (kwargs['r_in'], kwargs['r_out'])
    r = kwargs.get('r')
    if not check_iterable(r):
        if r is None or r=='auto':
            apkwargs['r'] = max(round(0.6371*fwhm, 0), 1)
        else:
            apkwargs['r'] = r
    else:
        apkwargs['r'] = r

    if 'rdnoise_key' in kwargs.keys():
        rdnoise = kwargs.get('rdnoise_key')
        if rdnoise in s[0].header.keys():
            apkwargs['readnoise'] = float(s[0].header[rdnoise])

    phot = process_list(process_photometry, s, x=sources['x'],
                        y=sources['y'], photometry_type=photometry_type,
                        **apkwargs)
    r_used = list(set(phot[0]['aperture']))

    rtable = Table()
    for i in r_used:
        ft = [p[p['aperture'] == i] for p in phot]
        pol = _do_polarimetry(ft, psi, retarder_type=retarder_type,
                              pairs=pairs, positions=ret,
                              calculate_mode=calculate_mode)
        if wcs is not None:
            cat_unit = kwargs['identify_catalog'].flux_unit
            tmp = solve_photometry(pol, cat_mag=ids['cat_mag'],
                                   cat_scale=cat_unit,
                                   **solvekwargs)
            pol['mag'] = tmp['mag']
            pol['mag_err'] = tmp['mag_err']
        pol = hstack([ids, Table([ft[0]['aperture']]), pol])
        rtable = vstack([rtable, pol])

    # remove masked rows
    rtable = rtable[~rtable['star_index'].mask]
    return rtable, wcs, ret
