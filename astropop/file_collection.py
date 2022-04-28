"""Module to manage and classify fits files."""

import os
import fnmatch
import glob
from pathlib import Path
import numpy as np

from astropy.table import Table
from astropy.io import fits

from ._db import SQLDatabase, _ID_KEY, sql
from .fits_utils import _fits_extensions, \
                        _fits_extensions_with_compress
from .framedata import check_framedata
from .py_utils import check_iterable
from .logger import logger

__all__ = ['list_fits_files', 'FitsFileGroup', 'create_table_summary']


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
        files = list(glob.iglob(os.path.join(location, '**/*' + i),
                                recursive=True))
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


_headers = 'headers'
_metadata = 'astropop_metadata'
_files_col = '__file'


class FitsFileGroup():
    """Easy handle groups of fits files."""

    def __init__(self, location=None, files=None, ext=0,
                 compression=False, database=':memory:', **kwargs):
        self._ext = ext
        self._extensions = kwargs.get('fits_ext')
        self._include = kwargs.get('glob_include')
        self._exclude = kwargs.get('glob_exclude')

        self._db = SQLDatabase(database)
        if database == ':memory:':
            self._db_dir = None
        else:
            self._db_dir = Path(database).resolve().parent

        self._read_db(files, location, compression, kwargs.get('update', 0))

    def _list_files(self, files, location, compression):
        extensions = self._extensions
        if extensions is None:
            if compression:
                extensions = _fits_extensions_with_compress
            else:
                extensions = _fits_extensions

        if files is not None and location is not None:
            raise ValueError('You can only specify either files or location.')
        if files is None and location is not None:
            files = list_fits_files(location, extensions,
                                    self._include, self._exclude)
        if files is None:
            files = []
        return files

    def _read_db(self, files, location, compression, update=False):
        """Read the database and generate the summary if needed."""
        initialized = _metadata in self._db.table_names
        if location is not None:
            location = str(location)

        if not initialized:
            self._db.add_table(_metadata)
            self._db.add_row(_metadata, {'GLOB_INCLUDE': self._include,
                                         'GLOB_EXCLUDE': self._exclude,
                                         'LOCATION': location,
                                         'COMPRESSION': compression,
                                         'FITS_EXT': self._extensions,
                                         'EXT': self._ext},
                             add_columns=True)

        self._include = self._db[_metadata, 'glob_include'][0]
        self._exclude = self._db[_metadata, 'glob_exclude'][0]
        self._extensions = self._db[_metadata, 'fits_ext'][0]
        self._ext = self._db[_metadata, 'ext'][0]
        self._location = self._db[_metadata, 'location'][0]
        self._compression = self._db[_metadata, 'compression'][0]

        if update or not initialized:
            self.update(files, location, compression)

    @property
    def files(self):
        """List files in the group."""
        files = self._db[_headers, _files_col].values
        if self._db_dir is not None:
            return [os.path.join(self._db_dir, f) for f in files]
        return files

    @property
    def summary(self):
        """Get a table with summary of the fits files."""
        return self._db[_headers].as_table()

    def __copy__(self, indexes=None):
        """Copy the current instance to a new object."""
        db = self._db.copy()
        if indexes is not None:
            db.drop_table(_headers)
            db.add_table(_headers, columns=self._db[_headers].column_names)
            for i in indexes:
                db.add_row(_headers, self._db[_headers][i].as_dict())

        nfg = object.__new__(FitsFileGroup)
        nfg._db = db
        nfg._db_dir = None
        nfg._read_db(None, None, None, False)
        nfg._location = None
        return nfg

    def __len__(self):
        """Get the number of files in the group."""
        return len(self._db[_headers])

    def filtered(self, keywords):
        """Create a new FitsFileGroup with only filtered files."""
        try:
            indexes = self._db.select(_headers, columns=[_ID_KEY],
                                      where=keywords)
        except sql.OperationalError:
            indexes = []
        if len(indexes) == 0:
            return self.__copy__(indexes=[])
        indexes = np.array(indexes).ravel() - 1
        return self.__copy__(indexes)

    def update(self, files=None, location=None, compression=False):
        """Update the database with the current files."""
        if _headers in self._db.table_names:
            self._db.drop_table(_headers)

        self._db.add_table(_headers)
        location = location or self._location
        compression = compression or self._compression
        files = self._list_files(files, location, compression)
        for i, f in enumerate(files):
            logger.debug('reading file %i from %i', i, len(files))
            self.add_file(f)

    def values(self, keyword, unique=False):
        """Return the values of a keyword in the summary.

        If unique, only unique values returned.
        """
        vals = self._db[_headers, keyword].values()
        if unique:
            vals = list(set(vals))
        return vals

    def add_column(self, name, values=None):
        """Add a new column to the summary."""
        self._db.add_column(_headers, name, data=values)

    def add_file(self, file):
        """Add a new file to the group."""
        header = fits.open(file)[self._ext].header
        logger.debug('reading file %s', file)
        if self._db_dir is not None:
            file = os.path.relpath(file, self._db_dir)
        hdr = {_files_col: file}
        hdr.update(dict(header))
        hdr.pop('COMMENT', None)
        hdr.pop('HISTORY', None)
        self._db.add_row(_headers,  hdr, add_columns=True)

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._db[_headers, item].values

        # returning FitsFileGroups
        if isinstance(item, (int, np.integer)):
            # single index will be interpreted as a single file group
            return self.__copy__(indexes=[item])
        if (isinstance(item, slice)):
            item = list(range(*item.indices(len(self))))
        if isinstance(item, (np.ndarray, list)):
            item = np.array(item)
            return self.__copy__(indexes=item)

        raise KeyError(f'{item}')

    def _intern_yelder(self, ext=None, ret_type=None, **kwargs):
        """Iterate over files."""
        ext = ext if ext is not None else self._ext
        for i in self.files:
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
