"""Module to manage and classify fits files."""

import os
import fnmatch
import glob
from venv import create
import numpy as np

from astropy.table import Table
from astropy.io import fits

from .fits_utils import _fits_extensions, \
                        _fits_extensions_with_compress
from .framedata import check_framedata
from .py_utils import check_iterable
from .logger import logger

__all__ = ['list_fits_files', 'FitsFileGroup', 'create_table_summary']


# TODO: different backends
# - table: in memory only. faster but not persistent
# - sql: disk file. slower but persistent


def create_table_summary(headers, n):
    """Create a table summary of headers.

    Parameters
    ----------
    headers: iterator
        Iterator for a list of header files.
    n: int
        Number of headers to iterate.
    """
    summary_dict = {}
    for i, head in enumerate(headers):
        logger.debug('Reading file %i from %i', i, n)
        keys = head.keys()
        for k in keys:
            k_lower = k.lower()
            if k_lower not in summary_dict.keys():
                summary_dict[k_lower] = [None]*n
            summary_dict[k_lower][i] = head.get(k)

    return Table(summary_dict)


def list_fits_files(location, fits_extensions=None,
                    glob_include=None, glob_exclude=None):
    """List all fist files in a directory, if compressed or not."""
    if fits_extensions is None:
        fits_extensions = _fits_extensions

    if not check_iterable(fits_extensions):
        fits_extensions = [fits_extensions]

    f = []
    for i in fits_extensions:
        files = glob.glob(os.path.join(location, '*' + i))
        # filter only glob include
        if glob_include is not None:
            if not check_iterable(glob_include):
                glob_include = [glob_include]
            for inc in glob_include:
                files = [i for i in files if fnmatch.fnmatch(i, inc)]
        f.extend(files)

    # Filter excluded files
    if glob_exclude is not None:
        if not check_iterable(glob_exclude):
            glob_exclude = [glob_exclude]
        for exc in glob_exclude:
            f = [i for i in f if not fnmatch.fnmatch(i, exc)]

    files = sorted(f)
    files = [os.path.join(location, i) for i in files]
    return sorted(files)


def gen_mask(table, keywords):
    """Generate a mask to be applyed in the filtering."""
    if len(table) == 0:
        return []

    t = Table(table)

    mask = np.ones(len(t), dtype=bool)
    for k, v in keywords.items():
        if not check_iterable(v):
            v = [v]
        k = k.lower()
        if k not in t.colnames:
            t[k] = [None]*len(t)
        nmask = [t[k][i] in v for i in range(len(t))]
        mask &= np.array(nmask)

    return mask


class FitsFileGroup():
    """Easy handle groups of fits files."""

    def __init__(self, location=None, files=None, ext=0,
                 compression=False, **kwargs):
        if kwargs.get('__uninitialized', False):
            # Skip init if not initialize. Manual initialization needed.
            return

        self._ext = ext
        self._extensions = kwargs.get('fits_ext',
                                      _fits_extensions_with_compress
                                      if compression else _fits_extensions)

        self._include = kwargs.get('glob_include')
        self._exclude = kwargs.get('glob_exclude')
        self._keywords = kwargs.get('keywords')

        if location is None and files is None:
            raise ValueError("You must specify a 'location'"
                             "or a list of 'files'")
        if files is None and location is not None:
            files = list_fits_files(location, self._extensions,
                                    self._include, self._exclude)

        self._files = files
        self._location = location

        self._summary = create_table_summary(self.headers(), len(self))

    def __len__(self):
        return len(self.files)

    @property
    def files(self):
        return self._files.copy()

    @property
    def location(self):
        return self._location

    @property
    def keywords(self):
        return self._keywords

    @property
    def summary(self):
        return Table(self._summary)

    def __copy__(self, files=None, summary=None):
        nfg = FitsFileGroup(__uninitialized=True)
        for k, v in self.__dict__.items():
            if k == '_summary':
                nfg._summary = summary or self._summary
            elif k == '_files':
                nfg._files = files if files is not None else self._files
            else:
                nfg.__dict__[k] = v
        return nfg

    def __getitem__(self, item):
        if isinstance(item, str):
            # string will be interpreted as collumn name
            if item.lower() not in self._summary.colnames:
                raise KeyError(f'Column {item} not found.')
            return self._summary.columns[item.lower()]

        # returning FitsFileGroups
        if isinstance(item, (int, np.integer)):
            # single index will be interpreted as a single file group
            return self.__copy__(files=[self._files[item]],
                                 summary=self._summary[item])
        if (isinstance(item, slice)):
            files = self._files[item]
            summ = self._summary[item]
            return self.__copy__(files=files, summary=summ)
        if isinstance(item, (np.ndarray, list, tuple)):
            item = np.array(item)
            if len(item) == 0:
                return self.__copy__(files=[], summary=self._summary[item])
            files = list(np.take(self._files, item))
            summ = self._summary[item]
            return self.__copy__(files=files, summary=summ)

        raise KeyError(f'{item}')

    def filtered(self, keywords=None):
        """Create a new FileGroup with only filtered files."""
        where = np.where(gen_mask(self._summary, keywords))[0]
        return self[where]

    def values(self, keyword, unique=False):
        """Return the values of a keyword in the summary.

        If unique, only unique values returned.
        """
        if keyword not in self.summary.colnames:
            if unique:
                n = 1
            else:
                n = len(self.summary)
            return [None]*n
        if unique:
            return list(set(self.summary[keyword].tolist()))
        return self.summary[keyword].tolist()

    def add_column(self, name, values, mask=None):
        """Add a new column to the summary."""
        if not check_iterable(values):
            values = [values]*len(self.summary)
        elif len(values) != len(self.summary):
            values = [values]*len(self.summary)

        self.summary[name] = values
        self.summary[name].mask = mask

    def _intern_yelder(self, files=None, ext=None, ret_type=None,
                       **kwargs):
        """Iter over files."""
        ext = ext if ext is not None else self._ext
        files = files if files is not None else self._files
        for i in files:
            if ret_type == 'header':
                yield fits.open(i, **kwargs)[ext].header
            if ret_type == 'data':
                yield fits.open(i, **kwargs)[ext].data
            if ret_type == 'hdu':
                yield fits.open(i, **kwargs)[ext]
            if ret_type == 'framedata':
                yield check_framedata(i, hdu=ext, **kwargs)

    def hdus(self, ext=None, **kwargs):
        """Read the files and iterate over their HDUs."""
        return self._intern_yelder(ext=ext, ret_type='hdu', **kwargs)

    def headers(self, ext=None, **kwargs):
        """Read the files and iterate over their headers."""
        return self._intern_yelder(ext=ext, ret_type='header', **kwargs)

    def data(self, ext=None, **kwargs):
        """Read the files and iterate over their data."""
        return self._intern_yelder(ext=ext, ret_type='data', **kwargs)

    def framedata(self, ext=None, **kwargs):
        """Read the files and iterate over their data."""
        return self._intern_yelder(ext=ext, ret_type='framedata', **kwargs)
