# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import os
import pytest
import numpy as np
import warnings

from astropy.io import fits
from astropop.file_collection import FitsFileGroup, list_fits_files
from astropop.framedata import FrameData
from astropop.testing import *


@pytest.fixture(scope='session')
def tmpdir(tmp_path_factory):
    fn = tmp_path_factory.mktemp('filegroups')
    files = {}
    for i in ('fits', 'fz', 'fits.gz', 'fts', 'fit'):
        folder = fn / i.replace('.', '_')
        folder.mkdir()
        files[i] = create_test_files(folder, extension=i)

    # Also create the images on the custom HDU

    tmpdir = fn / 'custom_hdu'
    tmpdir.mkdir()
    flist = []
    for i in range(10):
        fname = tmpdir / f'bias_{i+1}.fits'
        if fname.is_file():
            continue
        hdr = fits.Header({'obstype': 'bias',
                           'exptime': 0.0001,
                           'observer': 'Galileo Galileo',
                           'object': 'bias',
                           'filter': ''})
        hdul = fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(np.ones((8, 8), dtype=np.int16), hdr,
                          name='image')
            ])
        hdul.writeto(fname)
        flist.append(str(fname))
    files['custom_hdu'] = flist

    files['custom'] = []
    folder = fn / 'custom'
    folder.mkdir()
    for i in ('myfits', 'otherfits'):
        f = create_test_files(folder, extension=i)
        files['custom'].extend(f)
        files[i] = f

    return fn, files


def create_test_files(tmpdir, extension='fits'):
    """Create dummy test files for testing."""
    files_list = []
    # create 10 bias files
    for i in range(10):
        iname = f'bias_{i}.{extension}'
        fname = tmpdir / iname
        if fname.is_file():
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',
                                  category=fits.verify.VerifyWarning)
            hdr = fits.Header({'obstype': 'bias',
                               'exptime': 0.0001,
                               'observer': 'Galileo Galileo',
                               'object': 'bias',
                               'filter': '',
                               'space key': 1,
                               'image': iname})
            hdu = fits.PrimaryHDU(np.ones((8, 8), dtype=np.int16), hdr)
            hdu.writeto(fname)
        files_list.append(str(fname))

    # create 10 flat V files
    for i in range(10):
        iname = f'flat_{i}_v.{extension}'
        fname = tmpdir / iname
        if fname.is_file():
            continue
        hdr = fits.Header({'obstype': 'flat',
                           'exptime': 10.0,
                           'observer': 'Galileo Galileo',
                           'object': 'flat',
                           'filter': 'V',
                           'space key': 1,
                           'image': iname})
        hdu = fits.PrimaryHDU(np.ones((8, 8), dtype=np.int16), hdr)
        hdu.writeto(fname)
        files_list.append(str(fname))

    # create 10 object V files
    for i in range(10):
        iname = f'object_{i}_v.{extension}'
        fname = tmpdir / iname
        if fname.is_file():
            continue
        hdr = fits.Header({'obstype': 'science',
                           'exptime': 1.0,
                           'observer': 'Galileo Galileo',
                           'object': 'Moon',
                           'filter': 'V',
                           'space key': 1,
                           'image': iname})
        hdu = fits.PrimaryHDU(np.ones((8, 8), dtype=np.int16), hdr)
        hdu.writeto(fname)
        files_list.append(str(fname))

    return files_list


class Test_FitsFileGroup():
    def test_fg_creation_empty(self):
        fg = FitsFileGroup()
        assert_is_instance(fg, FitsFileGroup)
        assert_equal(len(fg), 0)

    def test_fg_create_filegroup(self, tmpdir):
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(location=tmpdir/'fits_gz', compression=True)
        assert_is_instance(fg, FitsFileGroup)
        assert_equal(len(fg), 30)
        assert_equal(sorted(fg.files), sorted(flist['fits.gz']))

    def test_fg_create_filegroup_without_compression(self, tmpdir):
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(location=tmpdir/'fits', compression=False)
        assert_is_instance(fg, FitsFileGroup)
        assert_equal(len(fg), 30)
        assert_equal(sorted(fg.files), sorted(flist['fits']))

        # Default is False
        fg = FitsFileGroup(location=tmpdir/'fits')
        assert_is_instance(fg, FitsFileGroup)
        assert_equal(len(fg), 30)
        assert_equal(sorted(fg.files), sorted(flist['fits']))

    def test_fg_creation_files(self, tmpdir):
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(files=flist['fits'])
        assert_is_instance(fg, FitsFileGroup)
        assert_equal(len(fg), 30)
        assert_equal(sorted(fg.files), sorted(flist['fits']))

    def test_fg_creation_files_and_location_error(self, tmpdir):
        tmpdir, flist = tmpdir
        with pytest.raises(ValueError,
                           match='You can only specify either files or '
                           'location.'):
            fg = FitsFileGroup(location=tmpdir/'fits', files=flist['fits'])

    def test_fg_creation_no_std_extension(self, tmpdir):
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(location=tmpdir/'fts')
        assert_is_instance(fg, FitsFileGroup)
        assert_equal(len(fg), 30)
        assert_equal(sorted(fg.files), sorted(flist['fts']))

        fg = FitsFileGroup(location=tmpdir/'fit')
        assert_is_instance(fg, FitsFileGroup)
        assert_equal(len(fg), 30)
        assert_equal(sorted(fg.files), sorted(flist['fit']))

        fg = FitsFileGroup(location=tmpdir/'fz')
        assert_is_instance(fg, FitsFileGroup)
        assert_equal(len(fg), 30)
        assert_equal(sorted(fg.files), sorted(flist['fz']))

    def test_fg_creation_glob_include_exclude(self, tmpdir):
        tmpdir, flist = tmpdir
        # only bias
        fg = FitsFileGroup(location=tmpdir/'fits', glob_include='*bias*')
        assert_is_instance(fg, FitsFileGroup)
        assert_equal(len(fg), 10)
        assert_equal(sorted(fg.files), sorted(flist['fits'][:10]))

        # everything except bias
        fg = FitsFileGroup(location=tmpdir/'fits', glob_exclude='*bias*')
        assert_is_instance(fg, FitsFileGroup)
        assert_equal(len(fg), 20)
        assert_equal(sorted(fg.files), sorted(flist['fits'][10:]))

    @pytest.mark.parametrize('hdu', [1, 'image'])
    def test_fg_creation_custom_hdu(self, tmpdir, hdu):
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(location=str(tmpdir/'custom_hdu'), ext=hdu)
        assert_is_instance(fg, FitsFileGroup)
        assert_equal(len(fg), 10)
        assert_equal(sorted(fg.files), sorted(flist['custom_hdu']))
        for k in ('object', 'exptime', 'observer', 'filter'):
            assert_in(k, fg.summary.colnames)
        for i in fg.summary:
            assert_equal(i['object'], 'bias')

    def test_fg_filtered_single_key(self, tmpdir):
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(location=tmpdir/'fits', compression=False)
        bias_files = flist['fits'][:10]
        flat_files = flist['fits'][10:20]
        sci_files = flist['fits'][20:]

        # object keyword
        fg_b = fg.filtered({'object': 'bias'})
        assert_equal(len(fg_b), 10)
        assert_equal(sorted(fg_b.files),
                     sorted(bias_files))

        # filter keyword
        fg_v = fg.filtered({'filter': 'V'})
        assert_equal(len(fg_v), 20)
        assert_equal(sorted(fg_v.files),
                     sorted(flat_files + sci_files))

        # Hierarch key with space
        fg_s = fg.filtered({'space key': 1})
        assert_equal(len(fg_s), 30)
        assert_equal(sorted(fg_s.files),
                     sorted(bias_files + flat_files + sci_files))

    def test_fg_filtered_multiple_key(self, tmpdir):
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(location=tmpdir/'fits', compression=False)
        nfg = fg.filtered({'object': 'Moon',
                           'exptime': 1.0,
                           'image': 'object_4_v.fits'})
        assert_equal(len(nfg), 1)
        assert_equal(nfg.files, [flist['fits'][24]])

    def test_fg_filtered_empty(self, tmpdir):
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(location=tmpdir/'fits', compression=False)

        # non existing key
        with pytest.raises(KeyError):
            nfg = fg.filtered({'NON-EXISTING': 1})

        # existing but not matched
        nfg = fg.filtered({'object': 'Sun'})
        assert_is_instance(nfg, FitsFileGroup)
        assert_equal(len(nfg), 0)
        assert_equal(nfg.files, [])

    def test_fg_getitem_column(self, tmpdir):
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(location=tmpdir/'fits', compression=False)
        obj_column = fg['object']
        assert_equal(sorted(obj_column),
                     sorted(['bias']*10+['flat']*10+['Moon']*10))

    def test_fg_getitem_single_file(self, tmpdir):
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(location=tmpdir/'fits', compression=False)
        row = fg[1]
        assert_is_instance(row, FitsFileGroup)
        assert_equal(len(row), 1)
        assert_equal(row.files, [flist['fits'][1]])

    def test_fg_getitem_slice(self, tmpdir):
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(location=tmpdir/'fits', compression=False)
        row = fg[2:5]
        assert_is_instance(row, FitsFileGroup)
        assert_equal(len(row), 3)
        assert_equal(row.files, flist['fits'][2:5])

    def test_fg_getitem_array_or_tuple(self, tmpdir):
        tmpdir, flist = tmpdir
        flist = flist['fits']
        files = [flist[2], flist[4]]
        fg = FitsFileGroup(location=tmpdir/'fits', compression=False)

        row = fg[[2, 4]]
        assert_is_instance(row, FitsFileGroup)
        assert_equal(len(row), 2)
        assert_equal(row.files, files)

        row = fg[np.array([2, 4])]
        assert_is_instance(row, FitsFileGroup)
        assert_equal(len(row), 2)
        assert_equal(row.files, files)

    def test_fg_getitem_empty(self, tmpdir):
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(location=tmpdir/'fits', compression=False)
        row = fg[[]]
        assert_is_instance(row, FitsFileGroup)
        assert_equal(len(row), 0)
        assert_equal(row.files, [])

    def test_fg_getitem_keyerror(self, tmpdir):
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(location=tmpdir/'fits', compression=False)
        with pytest.raises(KeyError):
            fg['NonExistingKey']

    def test_fg_getitem_keyerror_type(self, tmpdir):
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(location=tmpdir/'fits', compression=False)
        with pytest.raises(KeyError):
            fg[1.0]

    def test_fg_setitem_str(self, tmpdir):
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(location=tmpdir/'fits', compression=False)
        fg.add_column('new_column')
        fg['new_column'] = ['test']*len(fg)
        assert_equal(sorted(fg['new_column']), ['test']*len(fg))

    def test_fg_setitem_tuple(self, tmpdir):
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(location=tmpdir/'fits', compression=False)
        fg.add_column('new_column', values=['test']*len(fg))
        fg['new_column', -1] = 'test1'
        expect = ['test']*len(fg)
        expect[-1] = 'test1'
        assert_equal(sorted(fg['new_column']), expect)

    def test_fg_group_by(self, tmpdir):
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(location=tmpdir/'fits', compression=False)
        for i in fg.group_by('object'):
            assert_is_instance(i, FitsFileGroup)
            assert_equal(len(i.values('object', unique=True)), 1)
            assert_in(i.values('object', unique=True)[0],
                      ['bias', 'flat', 'Moon'])
            assert_equal(len(i), 10)

    def test_fg_group_by_used_id(self, tmpdir):
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(location=tmpdir/'fits', compression=False)
        fg.add_column('id', values=np.arange(len(fg)))
        for i in fg.group_by('object'):
            assert_is_instance(i, FitsFileGroup)
            assert_equal(len(i.values('object', unique=True)), 1)
            assert_in(i.values('object', unique=True)[0],
                      ['bias', 'flat', 'Moon'])
            assert_equal(len(i), 10)

    def test_fg_remove_file_int(self, tmpdir):
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(location=tmpdir/'fits', compression=False)

        fg.remove_file(0)
        assert_equal(len(fg), 29)
        assert_equal(fg.files, flist['fits'][1:])

        fg.remove_file(-1)
        assert_equal(len(fg), 28)
        assert_equal(fg.files, flist['fits'][1:-1])

    def test_fg_remove_file_str(self, tmpdir):
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(location=tmpdir/'fits', compression=False)

        fg.remove_file(flist['fits'][0])
        assert_equal(len(fg), 29)
        assert_equal(fg.files, flist['fits'][1:])

        fg.remove_file(flist['fits'][-1])
        assert_equal(len(fg), 28)
        assert_equal(fg.files, flist['fits'][1:-1])

        with pytest.raises(ValueError, match='file not in group'):
            fg.remove_file('NonExistingFile')


class Test_FitsFileGroup_Properties():
    def test_fg_keys(self, tmpdir):
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(location=tmpdir/'fits', compression=False)
        assert_equal(fg.keys,
                     ['__file', 'simple', 'bitpix', 'naxis', 'naxis1',
                      'naxis2', 'obstype', 'exptime', 'observer', 'object',
                      'filter', 'space key', 'image'])


class Test_ListFitsFiles():
    def test_list_custom_extension(self, tmpdir):
        tmpdir, flist = tmpdir
        found_files = list_fits_files(tmpdir/'custom',
                                      fits_extensions='.myfits')
        assert_equal(sorted(found_files), sorted(flist['myfits']))

        found_files = list_fits_files(tmpdir/'custom',
                                      fits_extensions=['.myfits',
                                                       '.otherfits'])
        assert_equal(sorted(found_files), sorted(flist['custom']))

    @pytest.mark.parametrize('ext', ['fits', 'fz', 'fit', 'fts'])
    def test_list_no_extension(self, tmpdir, ext):
        tmpdir, flist = tmpdir
        found_files = list_fits_files(tmpdir/f'{ext}')
        assert_equal(sorted(found_files), sorted(flist[ext]))

    def test_list_glob_include(self, tmpdir):
        tmpdir, flist = tmpdir
        found_files = list_fits_files(tmpdir/'fits', glob_include='*bias*')
        # must find only bias
        assert_equal(sorted(found_files), sorted(flist['fits'][:10]))

    def test_list_glob_exclude(self, tmpdir):
        tmpdir, flist = tmpdir
        found_files = list_fits_files(tmpdir/'fits', glob_exclude='*bias*')
        # find everything except bias
        assert_equal(sorted(found_files), sorted(flist['fits'][10:]))


class Test_FitsFileGroup_Paths():
    def test_fg_full_path_file(self, tmpdir):
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(location=tmpdir/'fits', compression=False)
        for i in range(len(fg)):
            assert_equal(fg.full_path(i), str(flist['fits'][i]))
        for f in fg.files:
            assert_equal(fg.full_path(f), str(f))

    def test_fg_full_path_file_db(self, tmpdir):
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(location=tmpdir/'fits', compression=False,
                           database=tmpdir/'fits'/'database.db')

        for i in range(len(fg)):
            assert_equal(fg.full_path(i), str(flist['fits'][i]))
        for f in fg.files:
            assert_equal(fg.full_path(f), str(f))

        # but the files must be stored relativelly in database
        for i in range(len(fg)):
            assert_equal(fg['__file'][i],
                         os.path.relpath(flist['fits'][i], tmpdir/'fits'))

    def test_fg_relative_path_file(self, tmpdir):
        # no database means full paths
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(location=tmpdir/'fits', compression=False)
        for i in range(len(fg)):
            assert_equal(fg.relative_path(flist['fits'][i]), flist['fits'][i])

        for f in fg.files:
            assert_equal(fg.relative_path(f), f)

    def test_fg_relative_path_file_db(self, tmpdir):
        # database means relative paths
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(location=tmpdir/'fits', compression=False,
                           database=tmpdir/'fits'/'database.db')
        for i in range(len(fg)):
            assert_equal(fg.relative_path(flist['fits'][i]),
                         os.path.relpath(flist['fits'][i], tmpdir/'fits'))

        for f in fg.files:
            assert_equal(fg.relative_path(f),
                         os.path.relpath(f, tmpdir/'fits'))


class Test_FitsFileGroup_Yielders:
    def test_fg_yield_files(self, tmpdir):
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(location=tmpdir/'fits', compression=False)
        for i, f in enumerate(fg.hdus()):
            h = fits.open(flist['fits'][i])[0]
            assert_is_instance(f, fits.PrimaryHDU)
            assert_equal(f.data, h.data)
            assert_equal(f.header, h.header)

    def test_fg_yield_headers(self, tmpdir):
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(location=tmpdir/'fits', compression=False)
        for i, h in enumerate(fg.headers()):
            assert_is_instance(h, fits.Header)
            assert_equal(h, fits.getheader(flist['fits'][i]))

    def test_fg_yield_data(self, tmpdir):
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(location=tmpdir/'fits', compression=False)
        for i, d in enumerate(fg.data()):
            assert_is_instance(d, np.ndarray)
            assert_equal(d, fits.getdata(flist['fits'][i]))

    def test_fg_yield_framedata(self, tmpdir):
        tmpdir, flist = tmpdir
        fg = FitsFileGroup(location=tmpdir/'fits', compression=False)
        for i, d in enumerate(fg.framedata()):
            assert_is_instance(d, FrameData)
            frame = FrameData(fits.getdata(flist['fits'][i]),
                              header=fits.getheader(flist['fits'][i]))
            assert_equal(d.data, frame.data)
            assert_equal(d.header, frame.header)
