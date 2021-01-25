# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Some parts stolen from Astropy CCDData testing bench

from astropop.framedata.compat import extract_header_wcs
from astropy.io import fits
from astropy.wcs import WCS

from astropop.testing import assert_equal, assert_is_instance, \
                             assert_is_none, assert_not_in, \
                             assert_false, assert_in, \
                             assert_not_equal


_comon_wcs_keys = ('CTYPE', 'CRVAL', 'CRPIX', 'CD1_', 'CD2_', 'PC1_', 'PC2_')


_base_header = """SIMPLE  =                    T / Fits standard
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


def test_extract_header_nowcs():
    header = fits.Header.fromstring(_base_header, sep='\n')
    h, wcs = extract_header_wcs(header)
    assert_is_none(wcs)
    assert_is_instance(h, fits.Header)
    assert_equal(h, header)
    assert_false(h is header)


def test_extract_header_nosip():
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


def test_extract_header_sip():
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


def test_extract_invalid_wcs_header():
    # It should no raise, just return empty wcs
    # No header change too
    header = fits.Header.fromstring(_base_header+_invalid_wcs, sep='\n')
    h, wcs = extract_header_wcs(header)
    assert_is_none(wcs)
    assert_is_instance(h, fits.Header)
    assert_equal(h, header)
    assert_false(h is header)
