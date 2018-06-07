"""Module to manage and classify fits files."""

import os
import fnmatch
import numpy as np
from collections import OrderedDict

from astropy.table import Table, vstack, Row, Column
from astropy.io import fits

from .fits_utils import fits_yielder, headers_to_table
from .py_utils import check_iterable


def list_fits_files(directory, fits_extensions=['.fts', '.fits', '.fit', '.fz'],
                    compression_extensions=None,
                    exclude=None):
    """List all fist files in a directory, if compressed or not."""
    all_files = os.listdir(directory)

    for i in ['.gz', '.bz2', '.Z', '.zip']:
        fits_extensions.extend([e + i for e in fits_extensions])

    f = []
    # Filter by filename
    for e in fits_extensions:
        f.extend(fnmatch.filter(all_files, '*' + e))
    # Filter excluded files
    if exclude is not None:
        if not check_iterable(exclude):
            exclude = [exclude]
        for exc in exclude:
            f = [i for i in f if not fnmatch.fnmatch(i, exc)]

    files = sorted(f)
    files = [os.path.join(directory, i) for i in files]
    return files


def gen_mask(table, **kwargs):
    """Generate a mask to be applyed in the filtering."""
    if len(table) == 0:
        return []

    t = table

    mask = np.ones(len(t), dtype=bool)
    for k, v in kwargs.items():
        if not check_iterable(v):
            v = [v]
        k = k.lower()
        nmask = [t[k][i] in v for i in range(len(t))]
        mask &= np.array(nmask)

    return mask


def row_to_header(row):
    """Transform a table row (or dict) in a header."""
    if isinstance(row, Row):
        rdict = OrderedDict((i, row[i]) for i in row.colnames
                            if str(row[i]) != '--')
    else:
        rdict = OrderedDict(row)
    return fits.Header(rdict)


class FileGroup():
    """Easy handle groups of fits files."""

    def __init__(self, files, ext, summary):
        """Easy handle groups of fits files."""
        self.files = np.array(files)
        if len(self.files) > 0:
            self.summary = Table(summary)
        else:
            self.summary = Table()

        if len(self.files) != len(self.summary):
            raise ValueError('Files and summary do not have same sizes.')
        self.ext = ext

    def __len__(self):
        return len(self.files)

    def values(self, keyword, unique=False):
        """Return the values of a keyword in the summary.

        If unique, only unique values returned.
        """
        if unique:
            return list(set(self.summary[keyword].tolist()))
        else:
            return self.summary[keyword].tolist()

    def add_column(self, name, values, mask=None):
        """Add a new column to the summary."""
        if not check_iterable(values):
            values = [values]*len(self.summary)
        elif len(values) != len(self.summary):
            values = [values]*len(self.summary)

        self.summary[name] = values
        self.summary[name].mask = mask

    def add_file(self, file):
        """Add a file to the current group."""
        h = fits.getheader(file, ext=self.ext)
        if len(self.summary) == 0:
            self.summary = headers_to_table([h], lower_keywords=True)
        else:
            nh = OrderedDict()
            for i in h.keys():
                nh[i.lower()] = h[i]
            h = nh

            r = OrderedDict()
            for i in self.summary.colnames:
                if i in h.keys():
                    r[i] = h[i]
                else:
                    r[i] = None
            vals = r.values()
            mask = [i is None for i in vals]
            self.summary.add_row(vals, mask=mask)

            for k in h.keys():
                if k not in self.summary.colnames:
                    col = [None]*len(self.summary)
                    col[-1] = h[k]
                    self.summary.add_column(Column(col, name=k))
                    self.summary[k].mask = [i is None for i in col]

        self.files = np.array(list(self.files) + [file])

    def _intern_yielder(self, return_type, save_to=None, append_to_name=None,
                       overwrite=False):
        return fits_yielder(return_type, self.files, self.ext,
                            append_to_name=append_to_name,
                            save_to=save_to, overwrite=overwrite)

    def hdus(self, **kwargs):
        return self._intern_yielder(return_type='hdu', **kwargs)

    def headers(self, **kwargs):
        return self._intern_yielder(return_type='header', **kwargs)

    def data(self, **kwargs):
        return self._intern_yielder(return_type='data', **kwargs)

class FileManager():
    """Handle and organize fits files in a simple way."""

    def __init__(self, ext=0, fits_extensions=['.fits', '.fts', '.fit', '.fz'],
                 compression=True, summary=None):
        """Handle and organize fits files in a simple way."""
        self.ext = ext
        self.extensions = fits_extensions
        if compression:
            self.compression = ['.gz', '.bz2', '.Z', '.zip']
        else:
            self.compression = []

    def group_by(self, filegroup, keywords):
        """Group the files by a list of keywords in multiple FileManagers."""
        keywords = [k.lower() for k in keywords]
        groups = filegroup.summary.group_by(keywords)
        keys = groups.groups.keys

        for i in range(len(keys)):
            fk = dict((k, keys[i][k]) for k in keys.colnames)
            yield self.filtered(filegroup, **fk)

    def filtered(self, filegroup, **kwargs):
        """Filter the current FileManager with fits keywords.

        Arguments:
            **kwargs :
                keywords to filter from fits header. Keywords with spaces,
                dashes or other things not allowed in python arguments
                can be put in a dict and be unpacked in the function.
                Ex:
                filtered(observer='Nobody', **{'space key' : 1.0})

        Return:
            A new FileManager instance with only filtered files.
        """
        where = np.where(gen_mask(filegroup.summary, **kwargs))[0]
        files = np.array([filegroup.files[i] for i in where])
        if len(files) > 0:
            summ = Table(filegroup.summary[where])
        else:
            summ = None
        nfg = FileGroup(files=files, summary=summ, ext=self.ext)
        return nfg

    def create_filegroup(self, path=None, files=None, ext=None, exclude=None):
        """Create a file group from a directory or specified files."""
        if path is None and files is None:
            raise ValueError("You must specify a 'path' or a list of 'files'")
        elif files is None:
            files = list_fits_files(path, self.extensions, self.compression,
                                    exclude)

        ext = ext or self.ext

        headers = fits_yielder('header', files, ext)
        summary = headers_to_table(headers, lower_keywords=True)

        return FileGroup(files, ext, summary)
