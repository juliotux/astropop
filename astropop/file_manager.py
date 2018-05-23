"""Module to manage and classify fits files."""

import os
import copy
import fnmatch
import numpy as np
from functools import partial

from astropy.table import Table, hstack

from .fits_utils import fits_yielder, headers_to_table
from .py_utils import check_iterable, process_list


class FileManager():
    """Handle and organize fits files in a simple way."""

    def __init__(self, path=None, files=None, ext=0, exclude=None,
                 fits_extensions=['.fits', '.fts', '.fit', '.fz'],
                 compression=True, summary=None):
        """Handle and organize fits files in a simple way."""
        if path is not None and files is not None:
            raise ValueError("files and path set. Just one allowed.")

        self.path = path
        self.ext = ext
        self.exclude = exclude
        self.extensions = fits_extensions
        if compression:
            for i in ['.gz', '.bz2', '.Z', '.zip']:
                self.extensions.extend([e + i for e in fits_extensions])

        self.path = path
        self._files = []
        if path is not None and files is None:
            # Only load all files if files not set manually
            self._read_files()
        elif self._files is not None:
            self._files = files

        # Store all files with absolute path
        self._files = [os.path.abspath(i) for i in self._files]

        if summary is not None:
            self._summary = summary
        else:
            self._update_summary()

    @property
    def summary(self):
        """Return the Table summary of fits headers."""
        c = Table()
        c['file'] = process_list(os.path.basename, self._files)
        return hstack([c, self._summary])

    @property
    def files(self):
        """Return the file list with full path."""
        return copy.copy(self._files)

    def values(self, keyword, unique=False):
        """Return the values of a keyword in the summary.
        If unique, only unique values returned.
        """
        if unique:
            return list(set(self.summary[keyword].tolist()))
        else:
            return self.summary[keyword].tolist()

    def filtered(self, **kwargs):
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
        filt_files, filt_summ = self._filter_keywords(**kwargs)
        nfm = FileManager(files=filt_files, summary=filt_summ, ext=self.ext)

        return nfm

    def filtered_glob(self, pattern):
        """Filter the current FileManager with a 'ls' or 'glob' pattern."""
        matches = [fnmatch.fnmatch(i, pattern) for i in self._files]
        where = np.where(matches)[0]

        return FileManager(files=[self._files[i] for i in where],
                           summary=self._summary[where],
                           ext=self.ext)

    def _filter_keywords(self, **kwargs):
        t = self.summary

        mask = np.ones(len(t), dtype=bool)
        for k, v in kwargs.items():
            k = k.lower()
            nmask = t[k] == v
            mask &= np.array(nmask)

        where = np.where(mask)[0]

        return [self._files[i] for i in where], self.summary[where]

    def _filter_fnames(self, files):
        f = []

        # Filter by filename
        for e in self.extensions:
            f.extend(fnmatch.filter(files, '*' + e))

        # Filter excluded files
        if self.exclude is not None:
            if not check_iterable(self.exclude):
                self.exclude = [self.exclude]
            for exc in self.exclude:
                f = [i for i in f if not fnmatch.fnmatch(i, exc)]

        return sorted(f)

    def _read_files(self):
        all_files = os.listdir(self.path)
        files = self._filter_fnames(all_files)
        files = [os.path.join(self.path, i) for i in files]
        self._files = files

    def _update_summary(self):
        headers = fits_yielder('header', self._files, self.ext)
        self._summary = headers_to_table(headers, lower_keywords=True)

    def _intern_yielder(self, return_type, save_to=None, append_to_name=None,
                       overwrite=False):
        return fits_yielder(return_type, self._files, self.ext,
                            append_to_name=append_to_name,
                            save_to=save_to, overwrite=overwrite)

    hdus = partial(_intern_yielder, 'hdu')
    header = partial(_intern_yielder, 'header')
    data = partial(_intern_yielder, 'data')
