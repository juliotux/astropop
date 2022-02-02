import pytest
import os
import numpy as np

from astropy.io import fits
from astropy.table import Column
from astropop.file_collection import FitsFileGroup, list_fits_files
from astropop.testing import assert_is_instance, assert_equal, \
                             assert_in


def create_test_files(tmpdir, compression=False, extension='fits'):
    """Create dummy test files for testing."""
    files_list = []
    # create 10 bias files
    for i in range(10):
        iname = f'bias_{i}.{extension}'
        fname = tmpdir / iname
        if compression:
            fname += '.gz'
        hdr = fits.Header({'obstype': 'bias',
                           'exptime': 0.0001,
                           'observer': 'Galileo Galileo',
                           'object': 'bias',
                           'filter': '',
                           'space key': 1,
                           'image': iname})
        hdu = fits.PrimaryHDU(np.ones((256, 256), dtype=np.int16), hdr)
        hdu.writeto(fname)
        files_list.append(fname.strpath)

    # create 10 flat V files
    for i in range(10):
        iname = f'flat_{i}_v.{extension}'
        fname = tmpdir / iname
        if compression:
            fname += '.gz'
        hdr = fits.Header({'obstype': 'flat',
                           'exptime': 10.0,
                           'observer': 'Galileo Galileo',
                           'object': 'flat',
                           'filter': 'V',
                           'space key': 1,
                           'image': iname})
        hdu = fits.PrimaryHDU(np.ones((256, 256), dtype=np.int16), hdr)
        hdu.writeto(fname)
        files_list.append(fname.strpath)

    # create 10 object V files
    for i in range(10):
        iname = f'object_{i}_v.{extension}'
        fname = tmpdir / iname
        if compression:
            fname += '.gz'
        hdr = fits.Header({'obstype': 'science',
                           'exptime': 1.0,
                           'observer': 'Galileo Galileo',
                           'object': 'Moon',
                           'filter': 'V',
                           'space key': 1,
                           'image': iname})
        hdu = fits.PrimaryHDU(np.ones((256, 256), dtype=np.int16), hdr)
        hdu.writeto(fname)
        files_list.append(fname.strpath)

    return files_list


def delete_files_list(flist):
    for file in flist:
        os.remove(file)


class Test_FitsFileGroup():
    def test_fg_creation_empty(self):
        with pytest.raises(ValueError):
            FitsFileGroup()
        # test the hidden option to create uninitialized group.
        fg = FitsFileGroup(__uninitialized=True)
        assert_is_instance(fg, FitsFileGroup)

    def test_fg_create_filegroup(self, tmpdir):
        flist = create_test_files(tmpdir, compression=True)
        fg = FitsFileGroup(location=tmpdir, compression=True)
        assert_is_instance(fg, FitsFileGroup)
        assert_equal(len(fg), 30)
        assert_equal(sorted(fg.files), sorted(flist))
        delete_files_list(flist)

    def test_fg_create_filegroup_without_compression(self, tmpdir):
        flist = create_test_files(tmpdir, compression=False)
        fg = FitsFileGroup(location=tmpdir, compression=False)
        assert_is_instance(fg, FitsFileGroup)
        assert_equal(len(fg), 30)
        assert_equal(sorted(fg.files), sorted(flist))

        #Default is False
        fg = FitsFileGroup(location=tmpdir)
        assert_is_instance(fg, FitsFileGroup)
        assert_equal(len(fg), 30)
        assert_equal(sorted(fg.files), sorted(flist))
        delete_files_list(flist)

    def test_fg_creation_files(self, tmpdir):
        flist = create_test_files(tmpdir, compression=False)
        fg = FitsFileGroup(files=flist)
        assert_is_instance(fg, FitsFileGroup)
        assert_equal(len(fg), 30)
        assert_equal(sorted(fg.files), sorted(flist))
        delete_files_list(flist)

    def test_fg_creation_custom_extension(self, tmpdir):
        flist = create_test_files(tmpdir, extension='fts')
        fg = FitsFileGroup(location=tmpdir)
        assert_is_instance(fg, FitsFileGroup)
        assert_equal(len(fg), 30)
        assert_equal(sorted(fg.files), sorted(flist))
        delete_files_list(flist)

        flist = create_test_files(tmpdir, extension='fit')
        fg = FitsFileGroup(location=tmpdir)
        assert_is_instance(fg, FitsFileGroup)
        assert_equal(len(fg), 30)
        assert_equal(sorted(fg.files), sorted(flist))
        delete_files_list(flist)

        flist = create_test_files(tmpdir, extension='fz')
        fg = FitsFileGroup(location=tmpdir)
        assert_is_instance(fg, FitsFileGroup)
        assert_equal(len(fg), 30)
        assert_equal(sorted(fg.files), sorted(flist))
        delete_files_list(flist)

    def test_fg_creation_glob_include_exclude(self, tmpdir):
        flist = create_test_files(tmpdir, compression=False)

        # only bias
        fg = FitsFileGroup(location=tmpdir, glob_include='*bias*')
        assert_is_instance(fg, FitsFileGroup)
        assert_equal(len(fg), 10)
        assert_equal(sorted(fg.files), sorted(flist[:10]))

        # everything except bias
        fg = FitsFileGroup(location=tmpdir, glob_exclude='*bias*')
        assert_is_instance(fg, FitsFileGroup)
        assert_equal(len(fg), 20)
        assert_equal(sorted(fg.files), sorted(flist[10:]))
        delete_files_list(flist)

    @pytest.mark.parametrize('hdu', [1, 'image'])
    def test_fg_creation_custom_hdu(self, tmpdir, hdu):
        flist = []
        for i in range(10):
            fname = tmpdir / f'bias_{i+1}.fits'
            hdr = fits.Header({'obstype': 'bias',
                               'exptime': 0.0001,
                               'observer': 'Galileo Galileo',
                               'object': 'bias',
                               'filter': ''})
            hdul = fits.HDUList([
                fits.PrimaryHDU(),
                fits.ImageHDU(np.ones((256, 256), dtype=np.int16), hdr,
                              name='image')
                ])
            hdul.writeto(fname)
            flist.append(fname.strpath)

        fg = FitsFileGroup(location=tmpdir, ext=hdu)
        assert_is_instance(fg, FitsFileGroup)
        assert_equal(len(fg), 10)
        assert_equal(sorted(fg.files), sorted(flist))
        for k in ('object', 'exptime', 'observer', 'filter'):
            assert_in(k, fg.summary.colnames)
        for i in fg.summary:
            assert_equal(i['object'], 'bias')
        delete_files_list(flist)

    def test_fg_filtered_single_key(self, tmpdir):
        flist = create_test_files(tmpdir, compression=False)
        fg = FitsFileGroup(location=tmpdir, compression=False)
        bias_files = flist[:10]
        flat_files = flist[10:20]
        sci_files = flist[20:]

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

        delete_files_list(flist)

    def test_fg_filtered_multiple_key(self, tmpdir):
        flist = create_test_files(tmpdir, compression=False)
        fg = FitsFileGroup(location=tmpdir, compression=False)
        nfg = fg.filtered({'object': 'Moon',
                           'exptime': 1.0,
                           'image': 'object_4_v.fits'})
        assert_equal(len(nfg), 1)
        assert_equal(nfg.files, [flist[24]])

        delete_files_list(flist)

    def test_fg_filtered_empty(self, tmpdir):
        flist = create_test_files(tmpdir)
        fg = FitsFileGroup(location=tmpdir, compression=False)

        # non existing key
        nfg = fg.filtered({'NON-EXISTING': 1})
        assert_is_instance(nfg, FitsFileGroup)
        assert_equal(len(nfg), 0)
        assert_equal(nfg.files, [])

        # existing but not matched
        nfg = fg.filtered({'object': 'Sun'})
        assert_is_instance(nfg, FitsFileGroup)
        assert_equal(len(nfg), 0)
        assert_equal(nfg.files, [])
        delete_files_list(flist)

    def test_fg_getitem_column(self, tmpdir):
        flist = create_test_files(tmpdir)
        fg = FitsFileGroup(location=tmpdir, compression=False)
        obj_column = fg['object']
        assert_equal(sorted(obj_column),
                     sorted(['bias']*10+['flat']*10+['Moon']*10))
        assert_is_instance(obj_column, Column)
        delete_files_list(flist)

    def test_fg_getitem_single_file(self, tmpdir):
        flist = create_test_files(tmpdir)
        fg = FitsFileGroup(location=tmpdir, compression=False)
        row = fg[1]
        assert_is_instance(row, FitsFileGroup)
        assert_equal(len(row), 1)
        assert_equal(row.files, [flist[1]])
        delete_files_list(flist)

    def test_fg_getitem_slice(self, tmpdir):
        flist = create_test_files(tmpdir)
        fg = FitsFileGroup(location=tmpdir, compression=False)
        row = fg[2:5]
        assert_is_instance(row, FitsFileGroup)
        assert_equal(len(row), 3)
        assert_equal(row.files, flist[2:5])
        delete_files_list(flist)

    def test_fg_getitem_array_or_tuple(self, tmpdir):
        flist = create_test_files(tmpdir)
        files = [flist[2], flist[4]]
        fg = FitsFileGroup(location=tmpdir, compression=False)
        row = fg[2, 4]
        assert_is_instance(row, FitsFileGroup)
        assert_equal(len(row), 2)
        assert_equal(row.files, files)

        row = fg[[2, 4]]
        assert_is_instance(row, FitsFileGroup)
        assert_equal(len(row), 2)
        assert_equal(row.files, files)

        row = fg[(2, 4)]
        assert_is_instance(row, FitsFileGroup)
        assert_equal(len(row), 2)
        assert_equal(row.files, files)

        row = fg[np.array([2, 4])]
        assert_is_instance(row, FitsFileGroup)
        assert_equal(len(row), 2)
        assert_equal(row.files, files)
        delete_files_list(flist)

    def test_fg_getitem_empty(self, tmpdir):
        flist = create_test_files(tmpdir)
        fg = FitsFileGroup(location=tmpdir, compression=False)
        row = fg[[]]
        assert_is_instance(row, FitsFileGroup)
        assert_equal(len(row), 0)
        assert_equal(row.files, [])
        delete_files_list(flist)

    def test_fg_getitem_keyerror(self, tmpdir):
        flist = create_test_files(tmpdir)
        fg = FitsFileGroup(location=tmpdir, compression=False)
        with pytest.raises(KeyError):
            fg['NonExistingKey']
        delete_files_list(flist)


class Test_ListFitsFiles():

    def test_list_custom_extension(self, tmpdir):
        flist1 = create_test_files(tmpdir, extension='myfits')
        found_files = list_fits_files(tmpdir, fits_extensions='myfits')
        assert_equal(sorted(found_files), sorted(flist1))

        flist2 = create_test_files(tmpdir, extension='otherfits')
        found_files = list_fits_files(tmpdir,
                                      fits_extensions=['myfits', 'otherfits'])
        assert_equal(sorted(found_files), sorted(flist1+flist2))
        delete_files_list(flist1)
        delete_files_list(flist2)

    def test_list_no_extension(self, tmpdir):
        flist = create_test_files(tmpdir)
        found_files = list_fits_files(tmpdir)
        assert_equal(sorted(found_files), sorted(flist))
        delete_files_list(flist)

        flist = create_test_files(tmpdir, extension='fz')
        found_files = list_fits_files(tmpdir)
        assert_equal(sorted(found_files), sorted(flist))
        delete_files_list(flist)

        flist = create_test_files(tmpdir, extension='fit')
        found_files = list_fits_files(tmpdir)
        assert_equal(sorted(found_files), sorted(flist))
        delete_files_list(flist)

        flist = create_test_files(tmpdir, extension='fts')
        found_files = list_fits_files(tmpdir)
        assert_equal(sorted(found_files), sorted(flist))
        delete_files_list(flist)

    def test_list_glob_include(self, tmpdir):
        flist = create_test_files(tmpdir)
        found_files = list_fits_files(tmpdir, glob_include='*bias*')
        # must find only bias
        assert_equal(sorted(found_files), sorted(flist[:10]))
        delete_files_list(flist)

    def test_list_glob_exclude(self, tmpdir):
        flist = create_test_files(tmpdir)
        found_files = list_fits_files(tmpdir, glob_exclude='*bias*')
        # find everything except bias
        assert_equal(sorted(found_files), sorted(flist[10:]))
        delete_files_list(flist)
