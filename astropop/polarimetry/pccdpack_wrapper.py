# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Read PCCDPACK result files. Generate scripts to execute in iraf for
pccdpack."""

import os
import re
from collections import OrderedDict
import numpy as np
try:
    import fortranformat as ff
except ModuleNotFoundError:
    raise('You need to have the fortranformat package to use pccdpack wrapper')

from astropy.io import ascii as asci
from astropy.io import fits
from astropy.table import Table, Column

from ..logger import logger


def read_qphot_mag(filename):
    """Read the .mag.1 file created by qphot and return it to a Table."""
    # TODO: perform reads with fortranformat
    f = open(filename, 'r').readlines()
    fields = []
    units = []
    formats = []

    header_created = False
    row = []

    cont = False
    group = False

    def _read_data(l):
        cont = False
        group = False
        if l[-1] == '\\':
            cont = True
        if l[-2] == '*':
            group = True

        data = []
        for d in l[:-2].split():
            if d == 'INDEF':
                d = np.nan
            else:
                try:
                    d = float(d)
                except ValueError:
                    pass
            data.append(d)

        return data, cont, group

    t = Table()
    for i in f:
        i = i.strip('\n')
        if i[:2] == '#K':
            mname = i[2:14].strip()
            mvalue = i[16:39].strip()
            munit = i[40:51].strip()
            mformat = i.split('%')[1].strip()[-1]
            if mformat == 's':
                pass
            elif mformat == 'd':
                try:
                    mvalue = int(mvalue)
                except Exception:
                    pass
            elif mformat == 'g':
                try:
                    mvalue = float(mvalue)
                except Exception:
                    pass
            elif mformat == 'b':
                if mvalue == 'no':
                    mvalue = False
                elif mvalue == 'yes':
                    mvalue = True
                else:
                    logger.warn('A strange bool key value found: {} {}'
                                .format(mname, mvalue))
            t.meta[mname] = (mvalue, munit)
        elif i[:2] == '#N':
            for n in i[2:-1].split():
                fields.append(n)
        elif i[:2] == '#U':
            for n in i[2:-1].split():
                units.append(n)
        elif i[:2] == '#F':
            for n in i[2:-1].split():
                formats.append(n)
        elif i[0] != '#':
            if not header_created:
                for n, u, f in zip(fields, units, formats):
                    if f[-1] == 's':
                        dtype = 'S32'
                    elif f[-1] == 'd':
                        dtype = 'int16'
                    elif f[-1] == 'g':
                        dtype = 'float64'
                    elif f[-1] == 'b':
                        dtype = 'bool'
                    else:
                        dtype = 'S32'
                    try:
                        t.add_column(Column(name=n, unit=u, dtype=dtype))
                    except Exception as e:
                        print(n)
                        raise e
                header_created = True

            data, cont, group = _read_data(i)

            for k in data:
                row.append(k)

            if not cont:
                # FIXME: simply ignore the mismatches
                try:
                    t.add_row(row)
                except Exception as e:
                    pass
                row = []

            if group:
                row = row[:-len(data)]

    for i in t.colnames:
        t.rename_column(i, str(i).lower())
    return t


def read_log(log_file, return_table=False):
    """Read the .log file from pccdpack and return a dict with the results"""

    f = open(log_file, 'r')
    f = f.readlines()

    result = OrderedDict()
    # {star_n:{aperture:{'params':{},z:[]},'positions':[]}
    star = None

    wave_n_pos = 16
    wave_pos = None
    wavetype = None

    read_star = False
    read_apertures = False
    _aperture = 0

    params_index = None
    z_index = None

    if return_table:
        t = None

    i = 0
    while i < len(f):
        lin = f[i].strip('\n').strip(' ')
        if re.match('No. of waveplate positions :\s+\d+', lin):
            wave_n_pos = int(re.findall('\d+', lin)[0])
        if re.match('Waveplate pos. : [\s+\d+]+ =\s+\d+', lin):
            wave_pos = [int(v) for v in re.findall('\d+', lin)[:-1]]
        if re.match('No. of apertures observed:\s+\d+', lin):
            n_apertures = int(re.findall('\d+', lin)[0])
        if re.match('Waveplate type :', lin):
            wavetype = lin.split(':')[1].strip()

        if re.match('STAR # \s+\d+\s+ .*', lin):
            star = int(re.findall('\d+', lin)[0])
            result[star] = OrderedDict()
            result[star]['positions'] = wave_pos
            read_star = True

        if re.match('APERTURE = \s+\d+.\d+.', lin):
            aperture = float(re.findall('\d+.\d+', lin)[0])
            result[star][aperture] = {'z': [], 'params': OrderedDict()}
            _aperture += 1
            read_apertures = True
            params_index = i+2
            z_index = i+5

        if read_star and read_apertures:
            plin = f[params_index].strip('\n').strip(' ')
            if wavetype == 'half':
                form = '1x, 4(f10.6), 2x, f8.2, 2x, f10.6'
                npar = 6
            elif wavetype == 'quarter':
                form = 'f10.6,f10.6,2f10.6,2f10.6,f6.1,f10.6,f11.7'
                npar = 8
            else:
                raise NotImplementedError('Not implementd for {}'
                                          .format(wavetype))
            try:
                params = ff.FortranRecordReader(form).read(plin)
            except ValueError:
                params = [np.nan]*npar
            plin = f[params_index-1].strip('\n').strip(' ')
            pnames = plin.split()
            del plin
            for n, v in zip(pnames, params):
                result[star][aperture]['params'][n] = v

            z = []
            while len(z) < wave_n_pos:
                zread = ff.FortranRecordReader('(1x,4(f10.6))')
                try:
                    zlin = zread.read(f[z_index])
                except Exception:
                    zlin = [np.nan]*4
                for k in zlin:
                    try:
                        z.append(float(k))
                    except:
                        z.append(np.nan)
                z_index += 1
                del zlin
            z = np.array(z)
            result[star][aperture]['z'] = z

            if _aperture == n_apertures:
                read_star = False
            read_apertures = False
            i = z_index
        i += 1

    if return_table:
        for s in result.keys():
            for a in result[s].keys():
                if a != 'positions':
                    z = result[s][a]['z']
                    params = result[s][a]['params']
                    if t is None:
                        p = ['f8']*len(params.keys())
                        t = Table(names=['STAR', 'APERTURE', 'Z',
                                         *params.keys()],
                                  dtype=['i4', 'f8', '{}f8'.format(wave_n_pos),
                                         *p])
                    try:
                        t.add_row([s, a, z, *params.values()])
                    except Exception as e:
                        print(['STAR', 'APERTURE', 'Z', *pnames],
                              [star, aperture, z, *params])
                        raise e

        t.meta['pccdpack wavetype'] = wavetype
        t.meta['pccdpack wave_npos'] = wave_n_pos
        t.meta['pccdpack wave_pos'] = ','.join(str(i) for i in wave_pos)
        return t

    for i in t.colnames:
        t.rename_column(i, str(i).lower())
    return result


def read_out(file_out, file_ord=None):
    """Read the out file, with optionally the ord file for x,y coords."""
    def _read_line(l, n):
        if n == 8:
            form = '1x, 4(f10.6), 2x, f8.2, 2x, f10.6'
            lenght = 63
        elif n == 10:
            form = 'f10.6,f10.6,2f10.6,2f10.6,f6.1,f10.6,f11.7'
            lenght = 87

        try:
            lin = ff.FortranRecordReader(form).read(l[:lenght])
        except ValueError:
            lin = [np.nan] * int(n-2)

        ap, nstar = l[lenght:].split()
        lin.append(float(ap))
        lin.append(int(nstar))
        return lin

    fout = open(file_out, 'r').readlines()
    t = Table(names=fout[0].strip('\n').split())
    for i in range(1, len(fout)):
        try:
            t.add_row(_read_line(fout[i].strip('\n'), len(t.colnames)))
        except ValueError:
            pass
    fout = t
    if file_ord is not None:
        ford = asci.read(file_ord)
        fout['X0'] = [ford['XCENTER'][2*i] for i in range(len(fout))]
        fout['Y0'] = [ford['YCENTER'][2*i] for i in range(len(fout))]
        fout['X1'] = [ford['XCENTER'][2*i+1] for i in range(len(fout))]
        fout['Y1'] = [ford['YCENTER'][2*i+1] for i in range(len(fout))]

    for i in fout.colnames:
        fout.rename_column(i, str(i).lower())
    return fout


def create_script(result_dir, image_list, star_name,
                  apertures=np.arange(1, 21, 1),
                  r_ann=60.0, r_dann=10.0, n_retarder_positions=16,
                  gain_key='GAIN', readnoise_key='RDNOISE',
                  auto_pol=False, normalize=True, retarder='half',
                  move=True):
    """Creates a script to easy execute pccdpack for a set of images."""
    wd = result_dir
    try:
        os.makedirs(wd)
    except Exception as e:
        print(e)

    # FIXME: for a bug in pccdpack, the images have to be in the data_dir
    # FIXME: when not, the pccdpack cannot read the generated .mag.1 files
    imlist = [os.path.basename(i) for i in image_list]
    d = os.path.dirname(image_list[0])
    # f = NamedTemporaryFile(prefix='pccdpack', suffix='.txt')
    # with open(f.name, 'w') as tmp:
    #     [tmp.write(i + '\n') for i in imlist]

    kwargs = {'object': star_name,
              # 'lista': '@' + f.name,
              'lista': ','.join(imlist),
              'imagem': imlist[0],
              'pospars': np.min([len(image_list), n_retarder_positions]),
              'nume_la': len(image_list),
              'nab': len(apertures),
              'abertur': ','.join([str(i) for i in apertures]),
              'anel': r_ann,
              'danel': r_dann,
              'autoabe': 'no',
              'desloca': 'no',
              'confirm': 'no',
              'verify': 'no'}

    ref = fits.open(image_list[0])[0]
    kwargs['gan'] = float(ref.header[gain_key])
    kwargs['readnoi'] = float(ref.header[readnoise_key])
    kwargs['tamanho'] = int(np.max(ref.data.shape))
    if retarder == 'quarter':
        kwargs['circula'] == 'yes'

    if auto_pol:
        comm = 'auto_pol'
    else:
        comm = 'padrao_pol'

    command = comm + ' '
    command += ' '.join(['{}={}'.format(i, j) for (i, j) in kwargs.items()])

    script = "cd {}\n{}\n".format(d, command)

    if wd != d and move:
        script += ("mv *.pdf *.ord *.mag.1 *.coo *.log *.shift *.out "
                   "*.old *.eps *.par list_mag *.dat *.dao dat.* *.nfo {}\n"
                   .format(wd))

    return script
