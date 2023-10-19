"""Module to manage and classify fits files."""

import os
import fnmatch
import glob
from pathlib import Path
import numpy as np
import sqlite3 as sql

from astropy.io import fits
from astropy.table import Column
from dbastable import SQLDatabase, _ID_KEY, SQLTable

from .fits_utils import _fits_extensions, \
                        _fits_extensions_with_compress
from .framedata import check_framedata
from .logger import logger

__all__ = ['FitsFileGroup', 'list_fits_files']


def list_fits_files(location, fits_extensions=None,
                    glob_include=None, glob_exclude=None):
    """List all fist files in a directory, if compressed or not.

    Parameters
    ----------
    location : str
        Main directory to look for the files. Files will be listed recursively.
    fits_extensions : str or list, optional
        FITS file name extension to be used. Default is None, wich means
        that the default extensions will be used, like '.fits' and '.fit'.
    glob_include : str, optional
        Glob pattern to include files. Default is None.
    glob_exclude : str, optional
        Glob pattern to exclude files. Default is None.

    Returns
    -------
    list
        List of files found.
    """
    if fits_extensions is None:
        fits_extensions = _fits_extensions

    if np.isscalar(fits_extensions):
        fits_extensions = [fits_extensions]

    f = []
    for i in fits_extensions:
        files = glob.glob(os.path.join(location, '**/*'+i), recursive=True)
        # filter only glob include
        if glob_include is not None:
            if np.isscalar(glob_include):
                glob_include = [glob_include]
            for inc in glob_include:
                files = [i for i in files if fnmatch.fnmatch(i, inc)]
        f.extend(files)

    # Filter excluded files
    if glob_exclude is not None:
        if np.isscalar(glob_exclude):
            glob_exclude = [glob_exclude]
        for exc in glob_exclude:
            f = [i for i in f if not fnmatch.fnmatch(i, exc)]

    files = sorted(f)
    # files = [os.path.join(location, i) for i in files]
    return files


_headers = 'headers'
_metadata = 'astropop_metadata'
_files_col = '__file'


class FitsFileGroup():
    """Easy handle groups of fits files.

    Parameters
    ----------
    location : str, optional
        Location of the fits files. If not specified, the files must be
        specified in the files parameter.
    files : list, optional
        List of files to be included in the group. If not specified, the
        location parameter must be specified.
    ext : int, optional
        FITS extension to be used. Default is 0.
    compression : bool, optional
        If True, add compression file name extensions to the list of
        extensions to be searched. Default is False.
    database : str, optional
        SQLite database to be used to store the information. Default is
        ':memory:', which means that the database will be stored in memory.
    glob_include : str, optional
        Glob pattern to include files. Default is None.
    glob_exclude : str, optional
        Glob pattern to exclude files. Default is None.
    fits_ext : str, optional
        FITS file name extension to be used. Default is None, wich means
        that the default extensions will be used.
    show_progress_bar : bool, optional
        If True, show a progress bar while reading the files. Default is
        False. Requires ``tqdm`` module. May not be compatible with every mod.
    """

    def __init__(self, location=None, files=None, ext=0,
                 compression=False, database=':memory:', **kwargs):
        self._ext = ext
        self._extensions = kwargs.pop('fits_ext', None)
        self._include = kwargs.pop('glob_include', None)
        self._exclude = kwargs.pop('glob_exclude', None)
        self._progress = kwargs.pop('show_progress_bar', False)

        for i in kwargs.keys():
            raise ValueError('Unknown parameter: {}'.format(i))

        # As the headers may contain not allowed keywords, let's enable Base32
        # column names
        self._db = SQLDatabase(database, allow_b32_colnames=True)
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
            self._db.add_rows(_metadata, {'GLOB_INCLUDE': self._include,
                                          'GLOB_EXCLUDE': self._exclude,
                                          'LOCATION': location,
                                          'COMPRESSION': compression,
                                          'EXT': self._ext},
                              add_columns=True)
            self._db.add_column(_metadata, 'FITS_EXT', self._extensions)

        self._include = self._db[_metadata, 'glob_include'][0]
        self._exclude = self._db[_metadata, 'glob_exclude'][0]
        self._extensions = self._db[_metadata, 'fits_ext'].values
        if self._extensions == [None]:
            self._extensions = None
        self._ext = self._db[_metadata, 'ext'][0]
        self._location = self._db[_metadata, 'location'][0]
        self._compression = self._db[_metadata, 'compression'][0]
        self._table = SQLTable(self._db, _headers)

        if update or not initialized:
            self.update(files, location, compression)

    @property
    def files(self):
        """List files in the group."""
        return [self.full_path(i) for i in self.values(_files_col)]

    @property
    def summary(self):
        """Get a readonly table with summary of the fits files."""
        return self._table.as_table()

    @property
    def keys(self):
        """List the keywords of the headers table."""
        return self._table.column_names

    def __copy__(self, indexes=None):
        """Copy the current instance to a new object."""
        if indexes is None:
            db = self._db.copy()
        else:
            db = self._db.copy(indexes={_headers: indexes})

        nfg = object.__new__(FitsFileGroup)
        nfg._db = db
        nfg._db_dir = self._db_dir  # need to keep the db_dir for the iterators
        nfg._read_db(None, None, None, False)
        return nfg

    def __len__(self):
        """Get the number of files in the group."""
        return len(self._table)

    def filtered(self, keywords):
        """Create a new FitsFileGroup with only filtered files.

        Parameters
        ----------
        keywords : dict
            Dictionary with the keywords to be used to filter the files.
            The keys are the column names and the values are the values
            to be used to filter the files.

        Returns
        -------
        FitsFileGroup
            A new FitsFileGroup with only the files that match the
            keywords.
        """
        try:
            indexes = self._table.select(columns=[_ID_KEY],
                                         where=keywords)
        except sql.OperationalError:
            indexes = []
        if len(indexes) == 0:
            return self.__copy__(indexes=[])
        indexes = np.array(indexes).ravel() - 1
        return self.__copy__(indexes)

    def group_by(self, keywords):
        """Create FitsFileGroups grouped by keywords.

        Parameters
        ----------
        keywords : list
            List of column names to be used to group the files.

        Yields
        ------
        FitsFileGroup
            A new FitsFileGroup with only the files that match the
            keywords.
        """
        summary = self.summary
        id_key = 'id'
        while id_key in summary.colnames:
            id_key += '_'
        summary.add_column(Column(np.arange(len(summary))),
                           name=id_key)

        grouped = summary.group_by(keywords)
        for g in grouped.groups:
            yield self.__copy__(indexes=list(g[id_key]))

    def update(self, files=None, location=None, compression=False):
        """Update the database with the current files."""
        if _headers in self._db.table_names:
            self._db.drop_table(_headers)

        self._db.add_table(_headers)
        location = location or self._location
        compression = compression or self._compression
        files = self._list_files(files, location, compression)
        if self._progress:
            from tqdm.contrib import tenumerate as enum
        else:
            enum = enumerate
        for i, f in enum(files):
            try:
                self.add_file(f)
            except Exception as e:
                # just log error instead of raise an exception
                logger.error('Error reading file %s: %s', f, e)

    def values(self, keyword, unique=False):
        """Return the values of a keyword in the summary.

        Parameters
        ----------
        keyword : str
            Name of the keyword to be used to filter the files.
        unique : bool, optional
            If unique, only unique values returned.

        Returns
        -------
        list
            List of values for the keyword.
        """
        vals = self._table[keyword].values
        if unique:
            vals = list(set(vals))
        return vals

    def add_column(self, name, values=None):
        """Add a new column to the summary.

        Parameters
        ----------
        name : str
            Name of the column.
        values : list, optional
            List of values for the column. If None, the column is
            initialized with null values.
        """
        self._table.add_column(name, data=values)

    def add_file(self, file):
        """Add a new file to the group.

        Parameters
        ----------
        file : str
            File name with absolute path or relative to the filegroup
            location.
        """
        header = fits.open(file)[self._ext].header
        logger.debug('reading file %s', file)
        file = self.relative_path(file)
        hdr = {_files_col: file}
        hdr.update(dict(header))

        # get rid of comments, history and empty keys
        hdr.pop('COMMENT', None)
        hdr.pop('HISTORY', None)
        hdr.pop('', None)

        self._table.add_rows(hdr, add_columns=True)

    def remove_file(self, file):
        """Remove a file from the group.

        Parameters
        ----------
        file : str or int
            If string, the file name with absolute path or relative to the
            filegroup location. If int, the index of the file.
        """
        if isinstance(file, int):
            index = file
        else:
            file = self.relative_path(file)
            if file not in self._table[_files_col]:
                raise ValueError(f'{file} file not in group')
            index = self._table.index_of({_files_col: file})
        self._table.delete_row(index)

    def full_path(self, file):
        """Get the full path of a file in the group.

        Parameters
        ----------
        file : str or int
            If string, the file name. If int, the index of the file.

        Returns
        -------
        path : str
            Full path of the file.
        """
        if isinstance(file, int):
            file = self._table[_files_col][file]
        if self._db_dir is not None:
            return os.path.abspath(os.path.join(self._db_dir, file))
        return file

    def relative_path(self, file):
        """Get the relative path of a file.

        Parameters
        ----------
        file : str
            Full path of the file.

        Returns
        -------
        path : str
            Relative path of the file.
        """
        if self._db_dir is not None:
            return os.path.relpath(file, self._db_dir)
        return file

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._table[item].values

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

    def __setitem__(self, item, value):
        """Set the value of a keyword in the summary."""
        self._table[item] = value

    def _intern_yelder(self, ext=None, ret_type=None, **kwargs):
        """Iterate over files."""
        ext = ext if ext is not None else self._ext
        for i in self.files:
            if self._db_dir is not None:
                i = os.path.join(self._db_dir, i)

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
