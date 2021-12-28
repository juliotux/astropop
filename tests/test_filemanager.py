import pytest
import tempfile
import urllib.request
import os

from astropy.io import fits
from astropop.file_manager import FileManager, FileGroup
from astropop.testing import assert_is_instance, assert_equal


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


class Test_FileManager():
    tmp = tmpdir

    def test_fm_creation(self):
        fm = FileManager()
        assert_is_instance(fm, FileManager)

    def test_fm_create_filegroup_with_compression(self):
        fm = FileManager()
        fg = fm.create_filegroup(path=self.tmp)

        assert_is_instance(fg, FileGroup)
        # 30 with compression
        assert_equal(len(fg), 30)

    def test_fm_create_filegroup_without_compression(self):
        fm = FileManager(compression=False)
        fg = fm.create_filegroup(path=self.tmp)

        assert_is_instance(fg, FileGroup)
        # 15 with compression
        assert_equal(len(fg), 15)

    def test_fm_filtered(self):
        fm = FileManager(compression=False)
        fg = fm.create_filegroup(path=self.tmp)
        bias_files = [f'/tmp/astropop-data-dir/BIAS_2x2_{i}.fits' for i in range(5)]
        flat_files = [f'/tmp/astropop-data-dir/FLAT_2x2_V_{i}.fits' for i in range(5)]
        sci_files = [f'/tmp/astropop-data-dir/HD5980_V_{i}.fits' for i in range(5)]

        # object keyword
        fg_b = fm.filtered(fg, {'object': 'BIAS'})
        assert_equal(len(fg_b), 5)
        assert_equal(sorted(fg_b.files),
                     sorted(bias_files))

        # filter keyword
        fg_v = fm.filtered(fg, {'FILTER': 'V'})
        assert_equal(len(fg_v), 10)
        assert_equal(sorted(fg_v.files),
                     sorted(flat_files + sci_files))

        # Hierarch key with space
        fg_s = fm.filtered(fg, {'space key': '1'})
        # assert_equal(len(fg_s), 15)
        assert_equal(sorted(fg_s.files),
                     sorted(bias_files + flat_files + sci_files))
