# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import shutil
import copy
import six
import time
import numpy as np
from scipy.spatial import cKDTree
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.table import Table, Column, vstack
from tempfile import mkdtemp

from ..math.polarimetry_models import HalfWaveModel, QuarterWaveModel
from .image_processing import check_hdu
from .photometry import (aperture_photometry, process_photometry,
                         solve_photometry)
from .astrometry import identify_stars, solve_astrometry, wcs_from_coords
from . import pccdpack_wrapper as pccd
from ..py_utils import (process_list, check_iterable)
from ..logging import log as logger


def estimate_dxdy(x, y, steps=[100, 30, 5, 3], bins=30):
    def _find_max(d):
        dx = 0
        for lim in (np.max(d), *steps):
            lo, hi = (dx-lim, dx+lim)
            lo, hi = (lo, hi) if (lo < hi) else (hi, lo)
            histx = np.histogram(d, bins=bins, range=[lo, hi])
            mx = np.argmax(histx[0])
            dx = (histx[1][mx]+histx[1][mx+1])/2
        return dx

    dya = []
    dxa = []

    for i in range(len(x)):
        for j in range(len(x)):
            if y[i] < y[j]:
                dya.append(y[i] - y[j])
                dxa.append(x[i] - x[j])

    return (_find_max(dxa), _find_max(dya))


def match_pairs(x, y, dx, dy, tolerance=1.0):
    """Match the pairs of ordinary/extraordinary points (x, y)."""
    dt = np.dtype([('o', int), ('e', int)])
    results = np.zeros(len(x), dtype=dt)
    npairs = 0

    p = list(zip(x, y))
    kd = cKDTree(p)

    for i in range(len(p)):
        px = p[i][0]-dx
        py = p[i][1]-dy
        d, j = kd.query((px, py), k=1, eps=tolerance,
                        distance_upper_bound=tolerance, n_jobs=-1)
        if d <= tolerance:
            results[npairs]['o'] = i
            results[npairs]['e'] = j
            npairs = npairs+1
            kd = cKDTree(p)

    return results[:npairs]


def estimate_normalize(o, e, positions, n_consecutive):
    """Estimate the normalization of a given set of data.
    """
    data_o = [[]]*n_consecutive
    data_e = [[]]*n_consecutive

    # First, we separate the data in the positions, relative to consecutive
    for i, oi, ei in zip(positions, o, e):
        index = int(i/n_consecutive)
        data_o[index].append(oi)
        data_e[index].append(ei)

    # check if all positions have a value
    for i in data_o:
        if i == []:
            logger.warn('Could not calculate polarimetry normalization. '
                        'Not all needed positions are available. Using k=1.')
            return 1

    # Now we use as each consecutive value the mean of the values in each index
    for i in range(n_consecutive):
        data_o[i] = np.nanmean(data_o[i])
        data_e[i] = np.nanmean(data_e[i])

    # Now, assuming the k will multiply e
    k = np.sum(data_o)/np.sum(data_e)
    logger.debug('Polarimetry normalization estimated as k={}'.format(k))
    return k


def compute_theta(q, u):
    '''Giving q and u, compute theta'''
    # theta = np.degrees(0.5*np.arctan(u/q)) % 180
    # result['theta'] = {'value': theta, 'sigma': 28.65*p_err/p}
    # Do in the pccdpack way
    theta = np.degrees(np.arctan(u/q))
    if q < 0:
        theta = theta + 180
    if u < 0 and q > 0:
        theta = theta + 360
    theta = theta/2
    theta = theta % 180
    return theta


def _calculate_polarimetry_parameters(z, psi, retarder='half', z_err=None):
    """Calculate the polarimetry directly using z.
    psi in degrees
    """
    result = {}
    if z_err is None:
        result['z'] = {'value': z,
                       'sigma': np.array([np.nan]*len(z))}
    else:
        result['z'] = {'value': z, 'sigma': z_err}

    if retarder == 'half':
        model = HalfWaveModel()
    elif retarder == 'quarter':
        model = QuarterWaveModel()
    else:
        raise ValueError('retarder {} not supported.'.format(retarder))

    psi = np.radians(psi)

    fitter = LevMarLSQFitter()
    if z_err is None:
        m_fitted = fitter(model, psi, z)
    else:
        m_fitted = fitter(model, psi, z, weights=1/z_err)
    info = fitter.fit_info
    # The errors of parameters are assumed to be the sqrt of the diagonal of
    # the covariance matrix
    for i, j, k in zip(m_fitted.param_names, m_fitted.parameters,
                       np.sqrt(np.diag(info['param_cov']))):
        result[i] = {'value': j, 'sigma': k}

    if z_err is not None:
        result['sigma_theor'] = np.sqrt(np.sum(np.square(z_err))/len(z))
    else:
        result['sigma_theor'] = np.nan

    q, u = result['q']['value'], result['u']['value']
    q_err, u_err = result['q']['sigma'], result['u']['sigma']

    p = np.sqrt(q**2 + u**2)
    p_err = np.sqrt(((q/p)**2)*(q_err**2) + ((u/p)**2)*(u_err**2))
    result['p'] = {'value': p, 'sigma': p_err}

    theta = compute_theta(q, u)
    result['theta'] = {'value': theta, 'sigma': 28.65*p_err/p}

    return result


def calculate_polarimetry(o, e, psi, retarder='half', o_err=None, e_err=None,
                          normalize=True, positions=None, min_snr=None,
                          filter_negative=True):
    """Calculate the polarimetry."""

    if retarder == 'half':
        ncons = 4
    elif retarder == 'quarter':
        ncons = 8
    else:
        raise ValueError('retarder {} not supported.'.format(retarder))

    o = np.array(o)
    e = np.array(e)

    # clean problematic sources (bad sky subtraction, low snr)
    if filter_negative and (np.array(o <= 0).any() or np.array(e <= 0).any()):
        o_neg = np.where(o < 0)
        e_neg = np.where(e < 0)
        o[o_neg] = np.nan
        e[e_neg] = np.nan

    if normalize and positions is not None:
        k = estimate_normalize(o, e, positions, ncons)
        z = (o-(e*k))/(o+(e*k))
    else:
        z = (o-e)/(o+e)

    # To fit pccdpack, we had to invert z
    z = -z

    if o_err is None or e_err is None:
        z_erro = None
    else:
        # Assuming individual z errors from propagation
        o_err = np.array(o_err)
        e_err = np.array(e_err)
        oi = 2*o/((o+e)**2)
        ei = -2*e/((o+e)**2)
        z_erro = np.sqrt((oi**2)*(o_err**2) + ((ei**2)*(e_err**2)))

    flux = np.sum(o)+np.sum(e)
    flux_err = np.sqrt(np.sum(o_err)**2 + np.sum(e_err)**2)

    def _return_empty():
        if retarder == 'half':
            keys = ['q', 'u']
        elif retarder == 'quarter':
            keys = ['q', 'u', 'v']
        dic = {}
        for i in keys + ['p', 'theta']:
            dic[i] = {'value': np.nan, 'sigma': np.nan}
        dic['z'] = {'value': z, 'sigma': z_erro}
        dic['sigma_theor'] = np.nan
        dic['flux'] = {'value': flux,
                       'sigma': flux_err}

        return dic

    if min_snr is not None and o_err is not None and e_err is not None:
        snr = flux/flux_err
        if snr < min_snr:
            logger.debug('Star with SNR={} eliminated.'.format(snr))
            return _return_empty()

    try:
        result = _calculate_polarimetry_parameters(z, psi, retarder=retarder,
                                                   z_err=z_erro)
    except Exception as e:
        return _return_empty()
    result['flux'] = {'value': flux,
                      'sigma': flux_err}

    return result


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
                                r_dann=r_out-r_in,
                                readnoise_key=rdnoise_key,
                                retarder=retarder_type,
                                auto_pol=True)
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
        for i in ['identify_catalog_file', 'identify_catalog_name', 'filter',
                  'identify_limit_angle', 'science_catalog',
                  'science_id_key', 'science_ra_key', 'science_dec_key']:
            if i in kwargs.keys():
                idkwargs[i] = kwargs[i]
        ids = identify_stars(Table([x, y], names=('x', 'y')), wcs, **idkwargs)
        out_table['cat_id'] = ids['cat_id']
        out_table['sci_id'] = ids['sci_id']
        out_table['ra'] = ids['ra']
        out_table['dec'] = ids['dec']

    shutil.rmtree(dtmp)

    return out_table, dat_table, log_table, wcs


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
