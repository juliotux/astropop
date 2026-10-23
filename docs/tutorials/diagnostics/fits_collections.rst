Index and iterate over local FITS files
=======================================

Create three tiny multi-extension FITS files in a temporary directory, then
inspect their indexed headers and image values. Use
:doc:`../../modules/files/index` for constructor options, index ownership, alternate
iterators and additional simple examples.

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

    >>> print('len(group):', len(group))
    len(group): 3
    >>> print('Filter columns:', [name for name in group.summary.colnames if name.lower() == 'filter'])
    Filter columns: ['filter']

    >>> red = group.filtered({'FILTER': 'R'})

    >>> print('len(red):', len(red))
    len(red): 2

    >>> # Grouping needs the stored column name, unlike case-insensitive header filtering.
    >>> filter_column = next(name for name in group.summary.colnames if name.lower() == 'filter')

    >>> print('Frames per filter:', sorted(len(subgroup) for subgroup in group.group_by([filter_column])))
    Frames per filter: [1, 2]

``group_by`` uses exact summary-column names (lowercase ``filter`` here),
unlike case-insensitive header keyword filtering.

Select files by path before indexing
------------------------------------

``location`` scans a directory, while ``files`` selects an explicit list;
supply only one. ``glob_include`` and ``glob_exclude`` match the discovered
path, including its directory. Build a pattern using the same directory as the
scan, rather than matching only a basename::

    >>> from astropop.file_collection import list_fits_files

    >>> pattern = str(folder / 'image_0*')
    >>> included = list_fits_files(folder, glob_include=pattern)

    >>> print('Included:', [Path(name).name for name in included])
    Included: ['image_0.fits']

    >>> excluded = list_fits_files(folder, glob_exclude=pattern)

    >>> print('Remaining:', [Path(name).name for name in excluded])
    Remaining: ['image_1.fits', 'image_2.fits']

Use the same pattern in ``FitsFileGroup(location=folder, ext=1,
glob_include=pattern)`` to index the selection. See :doc:`../../modules/files/index`
for explicit basename selection across nested directories.

Reuse the header database
-------------------------

The persistent database stores paths relative to its own directory; public
``files`` returns resolved paths. Reopen the database to reuse its stored index,
then call ``update`` explicitly after files or headers change::

    >>> print("group.relative_path(str(folder / 'image_0.fits')):", group.relative_path(str(folder / 'image_0.fits')))
    group.relative_path(str(folder / 'image_0.fits')): image_0.fits
    >>> print('Path(group.full_path(0)).is_absolute():', Path(group.full_path(0)).is_absolute())
    Path(group.full_path(0)).is_absolute(): True

    >>> reopened = FitsFileGroup(database=database)

    >>> print('len(reopened):', len(reopened))
    len(reopened): 3

    >>> # Refresh explicitly; reopening a database reuses its saved header index.
    >>> reopened.update(location=str(folder))

    >>> print('len(reopened):', len(reopened))
    len(reopened): 3
    >>> print("fits.getheader(folder / 'image_0.fits', 1)['FILTER']:", fits.getheader(folder / 'image_0.fits', 1)['FILTER'])
    fits.getheader(folder / 'image_0.fits', 1)['FILTER']: R

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
    ...     print('frame.shape:', frame.shape)
    ...     means.append(float(np.mean(frame.data)))
    frame.shape: (4, 4)
    frame.shape: (4, 4)
    frame.shape: (4, 4)

    >>> print('sorted(means):', sorted(means))
    sorted(means): [0.0, 1.0, 2.0]

Release the groups before deleting the temporary directory so their SQLite
connections no longer hold the database files open on Windows::

    >>> del frame, group, reopened, red
    >>> temporary.cleanup()

Do not wrap the iterator in ``list`` when the purpose is to bound live image
storage. Database ownership and FITS file ownership are distinct: filtering,
indexing and adding custom columns do not calibrate or rewrite the files.
