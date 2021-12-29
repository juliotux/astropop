"""Module to manage and classify fits files."""

import os
import fnmatch
import glob
import numpy as np

from astropy.table import Table

from .fits_utils import fits_yielder, _fits_extensions, \
                        _fits_extensions_with_compress
from .py_utils import check_iterable
from .logger import logger

__all__ = ['list_fits_files', 'FitsFileGroup']


def list_fits_files(location, fits_extensions=None,
                    glob_include=None, glob_exclude=None):
    """List all fist files in a directory, if compressed or not."""
    if fits_extensions is None:
        fits_extensions = _fits_extensions

    f = []
    for i in fits_extensions:
        files = glob.glob(os.path.join(location, '*' + i))
        # filter only glob include
        if glob_include is not None:
            if not check_iterable(glob_include):
                glob_include = []
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
    return files


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

        self._create_summary()

    def _create_summary(self):
        summary_dict = {}
        size = len(self)
        for i, head in enumerate(self.headers()):
            logger.debug('Reading file %i from %i', i, size)
            keys = self._keywords or head.keys()
            for k in keys:
                k_lower = k.lower()
                if k_lower not in summary_dict.keys():
                    summary_dict[k_lower] = [None]*size
                summary_dict[k_lower][i] = head.get(k)

        self._summary = Table(summary_dict)

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

    def _intern_yielder(self, return_type, save_to=None, append_to_name=None,
                        overwrite=False):
        return fits_yielder(return_type, self.files, self._ext,
                            append_to_name=append_to_name,
                            save_to=save_to, overwrite=overwrite)

    def hdus(self, **kwargs):
        return self._intern_yielder(return_type='hdu', **kwargs)

    def headers(self, **kwargs):
        return self._intern_yielder(return_type='header', **kwargs)

    def data(self, **kwargs):
        return self._intern_yielder(return_type='data', **kwargs)
