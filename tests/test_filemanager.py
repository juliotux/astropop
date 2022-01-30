import pytest
import urllib.request
import os
import numpy as np

from astropy.io import fits
from astropy.table import Column
from astropop.file_collection import FitsFileGroup
from astropop.testing import assert_is_instance, assert_equal


try:
    # Create temp files and directory
    # tmpdir = tempfile.TemporaryDirectory().name
    # Hardcoded to do not redownload every single time
    tmpdir = '/tmp/astropop-data-dir'
    os.makedirs(tmpdir, exist_ok=True)
    # Download some fits files
    with urllib.request.urlopen('https://raw.githubusercontent.com/sparc4-dev/astropop-data/main/raw_images/opd_ixon_bc_hd5980/filelist.txt') as f:
        for line in f.readlines():
            url = line.decode('UTF-8')
            filename = os.path.join(tmpdir, url.split('/')[-1].split('?')[0])
            if not os.path.exists(filename):
                urllib.request.urlretrieve(url, filename)
            fgz = fits.open(filename)
            fgz.writeto(filename + '.gz', overwrite=True)
    with open(os.path.join(tmpdir, 'test.txt'), 'w') as f:
        f.write('text file for test')
except:
    pytest.mark.skip

class Test_FitsFileGroup():
    tmp = tmpdir

    def test_fg_creation_empty(self):
        with pytest.raises(ValueError):
            FitsFileGroup()
        # test the hidden option to create uninitialized group.
        fg = FitsFileGroup(__uninitialized=True)
        assert_is_instance(fg, FitsFileGroup)

    def test_fg_create_filegroup_with_compression(self):
        fg = FitsFileGroup(location=self.tmp, compression=True)
        assert_is_instance(fg, FitsFileGroup)
        # 30 with compression
        assert_equal(len(fg), 30)

    def test_fg_create_filegroup_without_compression(self):
        fg = FitsFileGroup(location=self.tmp, compression=False)
        assert_is_instance(fg, FitsFileGroup)
        # 15 without compression
        assert_equal(len(fg), 15)

        # Default is False
        fg = FitsFileGroup(location=self.tmp)
        assert_is_instance(fg, FitsFileGroup)
        # 15 without compression
        assert_equal(len(fg), 15)

    @pytest.mark.skip
    def test_fg_creation_files(self):
        pass

    @pytest.mark.skip
    def test_fg_creation_custom_extension(self):
        pass

    @pytest.mark.skip
    def test_fg_creation_glob_include_exclude(self):
        pass

    @pytest.mark.skip
    def test_fg_creation_custom_hdu(self):
        pass

    def test_fg_filtered_single_key(self):
        fg = FitsFileGroup(location=self.tmp, compression=False)
        bias_files = [f'/tmp/astropop-data-dir/BIAS_2x2_{i}.fits' for i in range(5)]
        flat_files = [f'/tmp/astropop-data-dir/FLAT_2x2_V_{i}.fits' for i in range(5)]
        sci_files = [f'/tmp/astropop-data-dir/HD5980_V_{i}.fits' for i in range(5)]

        # object keyword
        fg_b = fg.filtered({'object': 'BIAS'})
        assert_equal(len(fg_b), 5)
        assert_equal(sorted(fg_b.files),
                     sorted(bias_files))

        # filter keyword
        fg_v = fg.filtered({'FILTER': 'V'})
        assert_equal(len(fg_v), 10)
        assert_equal(sorted(fg_v.files),
                     sorted(flat_files + sci_files))

        # Hierarch key with space
        fg_s = fg.filtered({'space key': '1'})
        assert_equal(len(fg_s), 15)
        assert_equal(sorted(fg_s.files),
                     sorted(bias_files + flat_files + sci_files))

    def test_fg_filtered_multiple_key(self):
        fg = FitsFileGroup(location=self.tmp, compression=False)
        nfg = fg.filtered({'OBJECT': 'HD5980',
                           'EXPTIME': '30,00000',
                           'IMAGE': 'HD5980_V_0103'})
        assert_equal(len(nfg), 1)
        assert_equal(nfg.files,
                     ['/tmp/astropop-data-dir/HD5980_V_3.fits'])

    def test_fg_filtered_empty(self):
        fg = FitsFileGroup(location=self.tmp, compression=False)

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

    def test_fg_getitem_column(self):
        fg = FitsFileGroup(location=self.tmp, compression=False)
        obj_column = fg['object']
        assert_equal(sorted(obj_column),
                     sorted(['BIAS']*5+['FLAT']*5+['HD5980']*5))
        assert_is_instance(obj_column, Column)

    def test_fg_getitem_single_file(self):
        fg = FitsFileGroup(location=self.tmp, compression=False)
        row = fg[1]
        assert_is_instance(row, FitsFileGroup)
        assert_equal(len(row), 1)
        assert_equal(row.files, ['/tmp/astropop-data-dir/BIAS_2x2_1.fits'])

    def test_fg_getitem_slice(self):
        fg = FitsFileGroup(location=self.tmp, compression=False)
        row = fg[2:5]
        assert_is_instance(row, FitsFileGroup)
        assert_equal(len(row), 3)
        assert_equal(row.files, ['/tmp/astropop-data-dir/BIAS_2x2_2.fits',
                                 '/tmp/astropop-data-dir/BIAS_2x2_3.fits',
                                 '/tmp/astropop-data-dir/BIAS_2x2_4.fits'])

    def test_fg_getitem_array_or_tuple(self):
        fg = FitsFileGroup(location=self.tmp, compression=False)
        row = fg[2, 4]
        assert_is_instance(row, FitsFileGroup)
        assert_equal(len(row), 2)
        assert_equal(row.files, ['/tmp/astropop-data-dir/BIAS_2x2_2.fits',
                                 '/tmp/astropop-data-dir/BIAS_2x2_4.fits'])

        row = fg[[2, 4]]
        assert_is_instance(row, FitsFileGroup)
        assert_equal(len(row), 2)
        assert_equal(row.files, ['/tmp/astropop-data-dir/BIAS_2x2_2.fits',
                                 '/tmp/astropop-data-dir/BIAS_2x2_4.fits'])

        row = fg[(2, 4)]
        assert_is_instance(row, FitsFileGroup)
        assert_equal(len(row), 2)
        assert_equal(row.files, ['/tmp/astropop-data-dir/BIAS_2x2_2.fits',
                                 '/tmp/astropop-data-dir/BIAS_2x2_4.fits'])

        row = fg[np.array([2, 4])]
        assert_is_instance(row, FitsFileGroup)
        assert_equal(len(row), 2)
        assert_equal(row.files, ['/tmp/astropop-data-dir/BIAS_2x2_2.fits',
                                 '/tmp/astropop-data-dir/BIAS_2x2_4.fits'])

    def test_fg_getitem_empty(self):
        fg = FitsFileGroup(location=self.tmp, compression=False)
        row = fg[[]]
        assert_is_instance(row, FitsFileGroup)
        assert_equal(len(row), 0)
        assert_equal(row.files, [])

    pytest.mark.skip
    def test_fg_getitem_keyerror(self):
        pass


@pytest.mark.skip
class Test_ListFitsFiles():
    def test_list_no_compress(self):
        pass

    def test_list_compress(self):
        pass

    def test_list_custom_extension(self):
        pass

    def test_list_no_extension(self):
        pass

    def test_list_glob_include(self):
        pass

    def test_list_glob_exclude(self):
        pass
