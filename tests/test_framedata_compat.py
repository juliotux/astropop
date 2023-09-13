# Licensed under a 3-clause BSD style license - see LICENSE.rst
# flake8: noqa: F403, F405

import pytest
import numpy as np
from astropop.framedata._compat import extract_header_wcs, _extract_ccddata, \
                                      _extract_fits, _normalize_and_strip_dict
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.nddata import StdDevUncertainty, InverseVariance, \
                           VarianceUncertainty, UnknownUncertainty, \
                           CCDData

from astropop.logger import logger, log_to_list
from astropop.testing import *


_comon_wcs_keys = ('CTYPE', 'CRVAL', 'CRPIX', 'CD1_', 'CD2_', 'PC1_', 'PC2_')


_base_header = """
SIMPLE  =                    T / Fits standard
BITPIX  =                  -32 / FOUR-BYTE SINGLE PRECISION FLOATING POINT
NAXIS   =                    2 / STANDARD FITS FORMAT
NAXIS1  =                  256 / STANDARD FITS FORMAT
NAXIS2  =                  256 / STANDARD FITS FORMAT
DATE-OBS= '1996-10-14T10:14:36.123' / Date and time of start of obs. in UTC.
"""

_wcs_no_sip = """
CTYPE1  = 'RA---TAN'       / Tangent projection
CTYPE2  = 'DEC--TAN'       / Tangent projection
CRVAL1  =     202.482322805429 / [deg] RA at CRPIX1,CRPIX2
CRVAL2  =     47.1751189300101 / [deg] DEC at CRPIX1,CRPIX2
CRPIX1  =                 128. / Reference pixel along axis 1
CRPIX2  =                 128. / Reference pixel along axis 2
CD1_1   = 0.000249756880272355 / Corrected CD matrix
CD1_2   = 0.000230177809743655 / Corrected CD matrix
CD2_1   = 0.000230428519265417 / Corrected CD matrix element
CD2_2   = -0.000249965770576587 / Corrected CD matrix element
"""

_wcs_sip = """
CTYPE1  = 'RA---TAN-SIP'       / RA---TAN with distortion in pixel
CTYPE2  = 'DEC--TAN-SIP'       / DEC--TAN with distortion in pixel
CRVAL1  =     202.482322805429 / [deg] RA at CRPIX1,CRPIX2
CRVAL2  =     47.1751189300101 / [deg] DEC at CRPIX1,CRPIX2
CRPIX1  =                 128. / Reference pixel along axis 1
CRPIX2  =                 128. / Reference pixel along axis 2
CD1_1   = 0.000249756880272355 / Corrected CD matrix element
CD1_2   = 0.000230177809743655 / Corrected CD matrix element
CD2_1   = 0.000230428519265417 / Corrected CD matrix element
CD2_2   = -0.000249965770576587 / Corrected CD matrix element
A_ORDER =                    3 / polynomial order, axis 1, detector to sky
A_0_2   =           2.9656E-06 / distortion coefficient
A_0_3   =           3.7746E-09 / distortion coefficient
A_1_1   =           2.1886E-05 / distortion coefficient
A_1_2   =          -1.6847E-07 / distortion coefficient
A_2_0   =          -2.3863E-05 / distortion coefficient
A_2_1   =           -8.561E-09 / distortion coefficient
A_3_0   =          -1.4172E-07 / distortion coefficient
A_DMAX  =                1.394 / [pixel] maximum correction
B_ORDER =                    3 / polynomial order, axis 2, detector to sky
B_0_2   =             2.31E-05 / distortion coefficient
B_0_3   =          -1.6168E-07 / distortion coefficient
B_1_1   =          -2.4386E-05 / distortion coefficient
B_1_2   =          -5.7813E-09 / distortion coefficient
B_2_0   =           2.1197E-06 / distortion coefficient
B_2_1   =          -1.6583E-07 / distortion coefficient
B_3_0   =          -2.0249E-08 / distortion coefficient
B_DMAX  =                1.501 / [pixel] maximum correction
AP_ORDER=                    3 / polynomial order, axis 1, sky to detector
AP_0_1  =          -6.4275E-07 / distortion coefficient
AP_0_2  =          -2.9425E-06 / distortion coefficient
AP_0_3  =           -3.582E-09 / distortion coefficient
AP_1_0  =          -1.4897E-05 / distortion coefficient
AP_1_1  =           -2.225E-05 / distortion coefficient
AP_1_2  =           1.7195E-07 / distortion coefficient
AP_2_0  =           2.4146E-05 / distortion coefficient
AP_2_1  =            6.709E-09 / distortion coefficient
AP_3_0  =           1.4492E-07 / distortion coefficient
BP_ORDER=                    3 / polynomial order, axis 2, sky to detector
BP_0_1  =          -1.6588E-05 / distortion coefficient
BP_0_2  =          -2.3424E-05 / distortion coefficient
BP_0_3  =            1.651E-07 / distortion coefficient
BP_1_0  =          -2.6783E-06 / distortion coefficient
BP_1_1  =           2.4753E-05 / distortion coefficient
BP_1_2  =           3.8917E-09 / distortion coefficient
BP_2_0  =           -2.151E-06 / distortion coefficient
BP_2_1  =              1.7E-07 / distortion coefficient
BP_3_0  =           2.0482E-08 / distortion coefficient
"""

_invalid_wcs = """
CRVAL1  =         164.98110962 / Physical value of the reference pixel X
CRVAL2  =          44.34089279 / Physical value of the reference pixel Y
CRPIX1  =                -34.0 / Reference pixel in X (pixel)
CRPIX2  =               2041.0 / Reference pixel in Y (pixel)
CDELT1  =           0.10380000 / X Scale projected on detector (#/pix)
CDELT2  =           0.10380000 / Y Scale projected on detector (#/pix)
CTYPE1  = 'RA---TAN'           / Pixel coordinate system
CTYPE2  = 'WAVELENGTH'         / Pixel coordinate system
CUNIT1  = 'degree  '           / Units used in both CRVAL1 and CDELT1
CUNIT2  = 'nm      '           / Units used in both CRVAL2 and CDELT2
CD1_1   =           0.20760000 / Pixel Coordinate translation matrix
CD1_2   =           0.00000000 / Pixel Coordinate translation matrix
CD2_1   =           0.00000000 / Pixel Coordinate translation matrix
CD2_2   =           0.10380000 / Pixel Coordinate translation matrix
C2YPE1  = 'RA---TAN'           / Pixel coordinate system
C2YPE2  = 'DEC--TAN'           / Pixel coordinate system
C2NIT1  = 'degree  '           / Units used in both C2VAL1 and C2ELT1
C2NIT2  = 'degree  '           / Units used in both C2VAL2 and C2ELT2
RADECSYS= 'FK5     '           / The equatorial coordinate system
"""

_hist_comm_blank = """
HISTORY First line of history
HISTORY Second line of history
HISTORY Third line of history
COMMENT This is a first comment
COMMENT This is a second comment
COMMENT This is a third comment
"""


class Test_NormalizeAndStripHeader():
    def test_double_keyword_warning(self):
        header = _base_header
        header += "TEST    =           0.10000000 / Test multiple keywords\n"
        header += "Test    =           0.20000000 / Test multiple keywords\n"
        h = fits.Header.fromstring(header, sep='\n')
        with pytest.warns(UserWarning) as record:
            h = _normalize_and_strip_dict(h)

    def test_enforce_fits(self):
        with pytest.raises(TypeError,
                           match='meta must be a fits.Header instance.'):
            _normalize_and_strip_dict({'test': 1})

    def test_strip_hist_comm_keys(self):
        h = fits.Header.fromstring(_base_header, sep='\n')
        h['COMMENT'] = 'This is a comment'
        h['HISTORY'] = 'This is a history'

        hdr, hist, comm = _normalize_and_strip_dict(h)
        assert_not_in('COMMENT', hdr.keys())
        assert_not_in('HISTORY', hdr.keys())
        assert_equal(hist, ['This is a history'])
        assert_equal(comm, ['This is a comment'])

    def test_strip_proctected_keys(self):
        h = fits.Header.fromstring(_base_header, sep='\n')

        hdr, hist, comm = _normalize_and_strip_dict(h)
        assert_not_in('SIMPLE', hdr.keys())
        assert_not_in('BITPIX', hdr.keys())
        assert_equal(hist, [])
        assert_equal(comm, [])

        assert_in('DATE-OBS', hdr.keys())
        assert_equal(hdr['DATE-OBS'], '1996-10-14T10:14:36.123')
        assert_equal(hdr.comments['DATE-OBS'],
                     'Date and time of start of obs. in UTC.')

    def test_strip_blank_keys(self):
        h = fits.Header.fromstring(_base_header, sep='\n')
        h[''] = 'This is a blank key'

        hdr, hist, comm = _normalize_and_strip_dict(h)
        assert_not_in('', hdr.keys())
        assert_equal(hist, [])
        assert_equal(comm, [])

    def test_key_without_key_comments(self):
        h = fits.Header.fromstring(_base_header, sep='\n')
        h['TEST'] = 'This is a test key'

        hdr, hist, comm = _normalize_and_strip_dict(h)
        assert_equal(hdr['TEST'], 'This is a test key')
        assert_equal(hdr.comments['TEST'], '')

    def test_keep_key_comments(self):
        h = fits.Header.fromstring(_base_header, sep='\n')
        h['TEST'] = ('This is a test key', 'This is a comment')

        hdr, hist, comm = _normalize_and_strip_dict(h)
        assert_equal(hdr['TEST'], 'This is a test key')
        assert_equal(hdr.comments['TEST'], 'This is a comment')


class Test_ExtractHeader():
    def test_extract_header_nowcs(self):
        header = fits.Header.fromstring(_base_header, sep='\n')
        h, wcs = extract_header_wcs(header)
        assert_is_none(wcs)
        assert_is_instance(h, fits.Header)
        assert_equal(h, header)
        assert_false(h is header)

    def test_extract_header_nosip(self):
        header = fits.Header.fromstring(_base_header+_wcs_no_sip, sep='\n')
        h, wcs = extract_header_wcs(header)
        assert_is_instance(wcs, WCS)
        assert_equal(wcs.wcs.ctype[0], 'RA---TAN')
        assert_equal(wcs.wcs.ctype[1], 'DEC--TAN')
        assert_is_instance(h, fits.Header)
        for i in _comon_wcs_keys:
            assert_not_in(f'{i}1', h.keys())
            assert_not_in(f'{i}2', h.keys())
        assert_in('DATE-OBS', h.keys())
        assert_false(h is header)
        assert_not_equal(h, header)

    def test_extract_header_sip(self):
        header = fits.Header.fromstring(_base_header+_wcs_sip, sep='\n')
        h, wcs = extract_header_wcs(header)
        assert_is_instance(wcs, WCS)
        assert_equal(wcs.wcs.ctype[0], 'RA---TAN-SIP')
        assert_equal(wcs.wcs.ctype[1], 'DEC--TAN-SIP')
        assert_is_instance(h, fits.Header)
        for i in _comon_wcs_keys:
            assert_not_in(f'{i}1', h.keys())
            assert_not_in(f'{i}2', h.keys())
        for i in ('A_0_2', 'AP_2_0', 'BP_ORDER', 'A_DMAX'):
            assert_not_in(i, h.keys())
        assert_in('DATE-OBS', h.keys())
        assert_false(h is header)
        assert_not_equal(h, header)

    def test_extract_invalid_wcs_header(self):
        # It should no raise, just return empty wcs
        # No header change too
        header = fits.Header.fromstring(_base_header+_invalid_wcs, sep='\n')
        del header['']
        h, wcs = extract_header_wcs(header)

        assert_is_none(wcs)
        assert_is_instance(h, fits.Header)
        assert_equal(h, header)
        assert_false(h is header)


class Test_Extract_CCDData():
    shape = (10, 10)
    meta = {'observer': 'testing', 'very long keyword': 1}
    unit = 'adu'

    @property
    def mask(self):
        mask = np.zeros(self.shape, dtype=bool)
        mask[1:3, 1:3] = 1
        return mask

    @property
    def ccd(self):
        data = 100*np.ones(self.shape)
        uncert = StdDevUncertainty(np.sqrt(data), unit=self.unit)
        unit = self.unit
        return CCDData(data, uncertainty=uncert, mask=self.mask,
                       unit=unit, meta=self.meta)

    def test_simple_import(self):
        ccd = self.ccd
        f = _extract_ccddata(ccd)
        assert_equal(f['data'], 100*np.ones(self.shape))
        assert_equal(f['mask'], self.mask)
        assert_equal(f['unit'], self.unit)
        assert_equal(f['meta'], self.meta)
        assert_equal(f['uncertainty'], 10*np.ones(self.shape))

    def test_variance_uncert(self):
        ccd = self.ccd
        uncert = VarianceUncertainty(100*np.ones(self.shape),
                                     unit=self.unit+'2')
        ccd.uncertainty = uncert

        f = _extract_ccddata(ccd)
        assert_equal(f['data'], 100*np.ones(self.shape))
        assert_equal(f['mask'], self.mask)
        assert_equal(f['unit'], self.unit)
        assert_equal(f['meta'], self.meta)
        assert_equal(f['uncertainty'], 10*np.ones(self.shape))

    def test_inverse_variance_uncert(self):
        ccd = self.ccd
        uncert = InverseVariance(0.01*np.ones(self.shape),
                                 unit=self.unit+'-2')
        ccd.uncertainty = uncert

        f = _extract_ccddata(ccd)
        assert_equal(f['data'], 100*np.ones(self.shape))
        assert_equal(f['mask'], self.mask)
        assert_equal(f['unit'], self.unit)
        assert_equal(f['meta'], self.meta)
        assert_equal(f['uncertainty'], 10*np.ones(self.shape))

    def test_no_mask(self):
        ccd = self.ccd
        ccd.mask = None
        f = _extract_ccddata(ccd)
        assert_is_none(f['mask'])

    def test_no_uncertainty(self):
        ccd = self.ccd
        ccd.uncertainty = None
        f = _extract_ccddata(ccd)
        assert_is_none(f['uncertainty'])

    def test_invalid_uncert(self):
        ccd = self.ccd
        ccd.uncertainty = UnknownUncertainty(np.ones(self.shape))

        with pytest.raises(TypeError):
            _extract_ccddata(ccd)

    def test_invalid_uncert_unit(self):
        ccd = self.ccd
        # we must do this to avoid ccddata self protection
        uncert = StdDevUncertainty(np.sqrt(ccd.data), unit='m')
        ccd._uncertainty = uncert
        ccd._uncertainty._parent_nddata = ccd

        with pytest.raises(ValueError):
            _extract_ccddata(ccd)


class Test_Fits_Extract():
    shape = (10, 10)

    @property
    def mask(self):
        mask = np.zeros(self.shape, dtype=bool)
        mask[1:3, 1:3] = 1
        return mask

    def create_hdu(self, uncert=False, mask=False,
                   unit='adu', unit_key='BUNIT'):
        li = []
        data = 100*np.ones(self.shape)
        header = self.create_header(unit=unit, unit_key=unit_key)
        data_hdu = fits.PrimaryHDU(data, header=header)
        li.append(data_hdu)

        if uncert:
            uncert_hdu = fits.ImageHDU(np.ones(self.shape), name='UNCERT')
            li.append(uncert_hdu)

        if mask:
            mask_hdu = fits.ImageHDU(self.mask.astype('uint8'), name='MASK')
            li.append(mask_hdu)

        hdul = fits.HDUList(li)
        return hdul

    def create_header(self, unit=None, unit_key=None):
        header = fits.Header()
        if unit is not None:
            header[unit_key] = unit
        return header

    def test_simple_hdu(self):
        hdu = self.create_hdu()[0]
        header = self.create_header('adu', 'BUNIT')
        f = _extract_fits(hdu)
        assert_equal(f['data'], 100*np.ones(self.shape))
        for i in header:
            assert_equal(f['meta'][i], header[i])
        assert_equal(f['unit'], 'adu')

    def test_simple_hdulist(self):
        hdu = self.create_hdu()
        header = self.create_header('adu', 'BUNIT')
        f = _extract_fits(hdu)
        assert_equal(f['data'], 100*np.ones(self.shape))
        for i in header:
            assert_equal(f['meta'][i], header[i])
        assert_equal(f['unit'], 'adu')

    def test_hdulist_with_uncertainty(self):
        hdu = self.create_hdu(uncert=True)
        header = self.create_header('adu', 'BUNIT')
        f = _extract_fits(hdu, hdu_uncertainty='UNCERT')
        assert_equal(f['data'], 100*np.ones(self.shape))
        for i in header:
            assert_equal(f['meta'][i], header[i])
        assert_equal(f['unit'], 'adu')
        assert_equal(f['uncertainty'], np.ones(self.shape))

    def test_hdulist_with_mask(self):
        hdu = self.create_hdu(mask=True)
        header = self.create_header('adu', 'BUNIT')
        f = _extract_fits(hdu, hdu_mask='MASK')
        assert_equal(f['data'], 100*np.ones(self.shape))
        for i in header:
            assert_equal(f['meta'][i], header[i])
        assert_equal(f['unit'], 'adu')
        assert_equal(f['mask'], self.mask.astype('uint8'))

    def test_hdulist_full(self):
        hdu = self.create_hdu(uncert=True, mask=True)
        header = self.create_header('adu', 'BUNIT')
        f = _extract_fits(hdu, hdu_mask='MASK')
        assert_equal(f['data'], 100*np.ones(self.shape))
        for i in header:
            assert_equal(f['meta'][i], header[i])
        assert_equal(f['unit'], 'adu')
        assert_equal(f['uncertainty'], np.ones(self.shape))
        assert_equal(f['mask'], self.mask.astype('uint8'))

    def test_invalid_unit(self):
        hdu = self.create_hdu(unit='invalid')
        with pytest.raises(ValueError):
            _extract_fits(hdu)

    def test_file(self, tmpdir):
        hdu = self.create_hdu(uncert=True, mask=True)
        header = self.create_header('adu', 'BUNIT')
        fname = tmpdir.join('test_file.fits').strpath
        hdu.writeto(fname)

        f = _extract_fits(fname)

        assert_equal(f['data'], 100*np.ones(self.shape))
        for i in header:
            assert_equal(f['meta'][i], header[i])
        assert_equal(f['unit'], 'adu')
        assert_equal(f['uncertainty'], np.ones(self.shape))
        assert_equal(f['mask'], self.mask.astype('uint8'))

    def test_data_in_other_hdu(self, tmpdir):
        tbl = Table(np.ones(10).reshape(5, 2))
        data = 100*np.ones(self.shape)
        hdul = fits.HDUList(hdus=[fits.PrimaryHDU(),
                                  fits.TableHDU(tbl.as_array()),
                                  fits.ImageHDU(data)])
        fname = tmpdir.join('test_table.fits').strpath
        hdul.writeto(fname)

        logs = []
        lh = log_to_list(logger, logs, full_record=True)
        f = _extract_fits(fname)
        assert_equal(f['data'], 100*np.ones(self.shape))
        assert_equal(f['unit'], None)

        # ensure log emitting
        logs = [i for i in logs if i.message == 'First hdu with image data: 2']
        assert_equal(len(logs), 1)
        assert_equal(logs[0].levelname, 'INFO')

        logger.removeHandler(lh)

    def test_invalid_type(self):
        with pytest.raises(TypeError):
            _extract_fits(None)

        with pytest.raises(TypeError):
            _extract_fits(np.array([1, 2, 3]))

        with pytest.raises(TypeError):
            _extract_fits(CCDData([1, 2, 3], unit='adu'))

    def test_no_data_in_hdu(self):
        hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU()])
        with pytest.raises(ValueError):
            _extract_fits(hdul, hdu=1)

    def test_unit_incompatible(self):
        hdul = self.create_hdu(uncert=True, mask=True)
        hdul[0].header['BUNIT'] = 'm'

        with pytest.raises(ValueError):
            _extract_fits(hdul, hdu_uncertainty='UNCERT',
                          unit_key='BUNIT', unit='adu')
