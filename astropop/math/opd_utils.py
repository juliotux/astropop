# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
OPD utils
---------

Specific tasks to deal with Pico dos Dias Observatory data standards.
"""


import re
import six
import numpy as np

from astropy.time import Time

from ..py_utils import check_iterable


__all__ = ['opd2jd', 'solve_decimal', 'read_opd_header_number']


dict_meses = {v: k+1 for (k, v) in enumerate(['jan', 'fev', 'mar', 'abr',
                                              'mai', 'jun', 'jul', 'ago',
                                              'set', 'out', 'nov', 'dez'])}


def opd2jd(opd_dates):
    '''
    Convert the OPD std date designation '99ago01' to Julian Date.
    '''
    def _match_opddate(date):
        if not re.match(r'\d\d\S\S\S\d\d', date):
            return False
        else:
            if date[2:5] in dict_meses.keys():
                return True
            else:
                return False

    matched = True
    if check_iterable(opd_dates):
        for i in opd_dates:
            if not _match_opddate(i):
                matched = False
    elif isinstance(opd_dates, six.string_types):
        if not _match_opddate(opd_dates):
            matched = False
        opd_dates = [opd_dates]
    else:
        matched = False
    if not matched:
        raise ValueError(f'Invalid OPD date to convert: {opd_dates}')

    res_arr = []
    opd_dates = np.array(opd_dates)
    for i in opd_dates:
        yy = int(i[0:2])
        mmm = dict_meses[i[2:5]]
        dd = int(i[5:])

        # Workaround for only 2 digits year
        if yy < 50:
            yy = 2000 + yy
        else:
            yy = 1900 + yy

        res_arr.append("%i-%i-%i" % (yy, mmm, dd))
    return Time(res_arr).jd


def solve_decimal(value):
    """Resolve problems with ',' decimal separator"""
    decmark = re.compile(r'(?<=\d),(?=\d)')
    return decmark.sub('.', str(value))


def read_opd_header_number(value):
    """Reads the numbers in headers from OPD images, that uses ',' as dec."""
    if isinstance(value, (six.integer_types, float)):
        return value

    if ',' in value:
        v = solve_decimal(value)
    else:
        v = value

    if v:
        try:
            if '.' in v:
                v = float(v)
            else:
                v = int(v)
        except ValueError:
            raise ValueError(f'Could not read the number: {value}')

    return v
