# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np
import numpy.testing as npt
import pytest_check as check

from astropop.image_processing.register import (translate, create_fft_shift_list, 
                                                create_chi2_shift_list, apply_shift, 
                                                apply_shift_list, hdu_shift_images)


exptm2 = np.array([[1., 1., 1., 1., 1., 1., 1., 1., 0, 0],
                   [1., 1., 1., 1., 1., 1., 1., 1., 0, 0],
                   [1., 1., 1., 1., 1., 1., 1., 1., 0, 0],
                   [1., 1., 2., 1., 1., 1., 1., 1., 0, 0],
                   [1., 2., 4., 2., 1., 1., 1., 1., 0, 0],
                   [1., 1., 2., 1., 1., 1., 1., 1., 0, 0],
                   [1., 1., 1., 1., 1., 1., 1., 1., 0, 0],
                   [1., 1., 1., 1., 1., 1., 1., 1., 0, 0],
                   [1., 1., 1., 1., 1., 1., 1., 1., 0, 0],
                   [1., 1., 1., 1., 1., 1., 1., 1., 0, 0]])

exptm1 = np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                   [1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                   [1., 1., 1., 2., 1., 1., 1., 1., 1., 0.],
                   [1., 1., 2., 4., 2., 1., 1., 1., 1., 0.],
                   [1., 1., 1., 2., 1., 1., 1., 1., 1., 0.],
                   [1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                   [1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                   [1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                   [1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

expt0 = np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                  [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                  [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                  [1., 1., 1., 1., 2., 1., 1., 1., 1., 1.],
                  [1., 1., 1., 2., 4., 2., 1., 1., 1., 1.],
                  [1., 1., 1., 1., 2., 1., 1., 1., 1., 1.],
                  [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                  [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                  [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                  [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])

exptp1 =   np.array([[0., 1. , 1., 1. , 1. , 1. , 1. , 1. , 1., 1.],
                     [0., 1. , 1., 1. , 1. , 1. , 1. , 1. , 1., 1.],
                     [0., 1. , 1., 1. , 1. , 1. , 1. , 1. , 1., 1.],
                     [0., 1. , 1., 1. , 1. , 1.5, 1.5, 1. , 1., 1.],
                     [0., 1. , 1., 1. , 1.5, 3. , 3. , 1.5, 1., 1.],
                     [0., 1. , 1., 1. , 1. , 1.5, 1.5, 1. , 1., 1.],
                     [0., 1. , 1., 1. , 1. , 1. , 1. , 1. , 1., 1.],
                     [0., 1. , 1., 1. , 1. , 1. , 1. , 1. , 1., 1.],
                     [0., 1. , 1., 1. , 1. , 1. , 1. , 1. , 1., 1.],
                     [0., 1. , 1., 1. , 1. , 1. , 1. , 1. , 1., 1.]])

exptp2 = np.array([[0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
                   [0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
                   [0., 0., 1., 1., 1., 1., 2., 1., 1., 1.],
                   [0., 0., 1., 1., 1., 2., 4., 2., 1., 1.],
                   [0., 0., 1., 1., 1., 1., 2., 1., 1., 1.],
                   [0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
                   [0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
                   [0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
                   [0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

exptm22 = np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1, 1],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1, 1],
                    [1., 1., 2., 1., 1., 1., 1., 1., 1, 1],
                    [1., 2., 4., 2., 1., 1., 1., 1., 1, 1],
                    [1., 1., 2., 1., 1., 1., 1., 1., 1, 1],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1, 1],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1, 1],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1, 1],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1, 1],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1, 1]])

expts2 = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 2., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 2., 4., 2., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 2., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])


@pytest.mark.parametrize(('shift, test_result'), [((2,0),exptm2), ((1,1),exptm1), 
                                                ((0,0),expt0), ((-1.5,0),exptp1),
                                                ((-2,1),exptp2)])
def test_translate(shift, test_result):
    
    dat1 = np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                     [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                     [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                     [1., 1., 1., 1., 2., 1., 1., 1., 1., 1.],
                     [1., 1., 1., 2., 4., 2., 1., 1., 1., 1.],
                     [1., 1., 1., 1., 2., 1., 1., 1., 1., 1.],
                     [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                     [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                     [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                     [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
    
    calculated = translate(dat1, shift)
        
    npt.assert_array_equal(calculated, test_result) 

# @pytest.mark.parametrize('r', [2, 3, 4])
def test_create_fft_shift_list():
    pass

# @pytest.mark.parametrize('r', [2, 3, 4])
def test_create_chi2_shift_list():
    pass

@pytest.mark.parametrize(('shift2, test_result2'), [((-2,0),expts2)])#, ((1,1),exptm1), 
                                                # ((0,0),expt0), ((-1.5,0),exptp1),
                                                # ((2,-1),exptp2)])
def test_apply_shift(shift2, test_result2):
    # apply_shift(image, shift, method='fft', subpixel=True, footprint=False,
    #             logger=logger):
    """Apply a shifts of (dx, dy) to a list of images.

    Parameters:
        image : ndarray_like
            The image to be shifted.
        shift: array_like
            shift to be applyed (dx, dy)
        method : string
            The method used for shift images. Can be:
            - 'fft' -> scipy fourier_shift
            - 'simple' -> simples translate using scipy

    Return the shifted images.
    """
    
    
    dat1 = np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                     [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                     [1., 1., 1., 1., 2., 1., 1., 1., 1., 1.],
                     [1., 1., 1., 2., 4., 2., 1., 1., 1., 1.],
                     [1., 1., 1., 1., 2., 1., 1., 1., 1., 1.],
                     [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                     [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                     [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                     [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                     [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
    
    calculated_fft_nofootprint, = apply_shift(dat1, shift2, 'fft', True, False)
    npt.assert_array_almost_equal(calculated_fft_nofootprint, test_result2) 
    
    calculated_fft,[], foot = apply_shift(dat1, shift2, 'fft', True, True)
    npt.assert_array_almost_equal(calculated_fft, test_result2) 

    dat0 = dat1 - 1
    dat2 = dat1 + 2
    
    calculated_fft_nofootprint, = apply_shift(dat0, shift2, 'fft', True, False)
    npt.assert_array_almost_equal(calculated_fft_nofootprint, test_result2-1) 
    
    calculated_fft,[], foot = apply_shift(dat0, shift2, 'fft', True, True)
    npt.assert_array_almost_equal(calculated_fft, test_result2-1)     
    
    calculated_fft_nofootprint, = apply_shift(dat2, shift2, 'fft', True, False)
    npt.assert_array_almost_equal(calculated_fft_nofootprint, test_result2+2) 
    
    calculated_fft,[], foot = apply_shift(dat2, shift2, 'fft', True, True)
    npt.assert_array_almost_equal(calculated_fft, test_result2+2)

# @pytest.mark.parametrize('r', [2, 3, 4])
def test_apply_shift_list():
    pass

# @pytest.mark.parametrize('r', [2, 3, 4])
def test_hdu_shift_images():
    #nunca utilizado, testar com cuidado
    pass

