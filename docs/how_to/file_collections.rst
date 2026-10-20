Index and iterate over local FITS files
=======================================

This pytest example creates three tiny multi-extension FITS files in a temporary
directory. It checks actual table columns and returned values; the older
:doc:`../tools/files` also contains illustrative verbatim transcripts, which
are not execution results from the current build.

Create image extensions and index their headers
-----------------------------------------------

The primary HDU is empty; the image and ``FILTER`` header live in extension 1.
The file group stores a header database without modifying the FITS files.
Resolve the temporary directory so input files and the database use the same
canonical path, including when the temporary directory is reached by a symlink::

    >>> import numpy as np
    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> from astropy.io import fits
    >>> from astropop.file_collection import FitsFileGroup
    >>> temporary = TemporaryDirectory()
    >>> folder = Path(temporary.name).resolve()
    >>> for index, band in enumerate(['R', 'G', 'R']):
    ...     hdu = fits.ImageHDU(np.full((4, 4), index), name='SCI')
    ...     hdu.header['FILTER'] = band
    ...     fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(folder / f'image_{index}.fits')
    >>> database = str(folder / 'headers.db')
    >>> group = FitsFileGroup(location=str(folder), ext=1, database=database)
    >>> assert len(group) == 3
    >>> assert 'filter' in [name.lower() for name in group.summary.colnames]
    >>> red = group.filtered({'FILTER': 'R'})
    >>> assert len(red) == 2
    >>> filter_column = next(name for name in group.summary.colnames if name.lower() == 'filter')
    >>> assert sorted(len(subgroup) for subgroup in group.group_by([filter_column])) == [1, 2]

``group_by`` uses exact summary-column names (lowercase ``filter`` here),
unlike case-insensitive header keyword filtering.

The persistent database stores paths relative to its own directory; public
``files`` returns resolved paths. Reopen the database to reuse its stored index,
then call ``update`` explicitly after files or headers change::

    >>> assert group.relative_path(str(folder / 'image_0.fits')) == 'image_0.fits'
    >>> assert Path(group.full_path(0)).is_absolute()
    >>> reopened = FitsFileGroup(database=database)
    >>> assert len(reopened) == 3
    >>> reopened.update(location=str(folder))
    >>> assert len(reopened) == 3
    >>> assert fits.getheader(folder / 'image_0.fits', 1)['FILTER'] == 'R'

Update rebuilds the index and logs unreadable files. It is not a transaction
that guarantees every discovered file was accepted. Inspect the resulting file
count and logs. Custom database columns do not edit headers on disk and must be
reconsidered when rebuilding the header table.

Iterate without retaining all frames
-------------------------------------

Choose the data extension explicitly when iterating. Keep only the scalar
measurements or output paths needed downstream, not every FrameData::

    >>> means = []
    >>> for frame in reopened.framedata(ext=1):
    ...     assert frame.shape == (4, 4)
    ...     means.append(float(np.mean(frame.data)))
    >>> assert sorted(means) == [0., 1., 2.]

Release the groups and collect reference cycles before deleting the temporary
directory. This finalizes dbastable's SQLite connections, which can otherwise
keep the database file open on Windows::

    >>> import gc
    >>> del frame, group, reopened, red
    >>> _ = gc.collect()
    >>> temporary.cleanup()

Do not wrap the iterator in ``list`` when the purpose is to bound live image
storage. Database ownership and FITS file ownership are distinct: filtering,
indexing and adding custom columns do not calibrate or rewrite the files.
