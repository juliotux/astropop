# -*- coding: utf-8 -*-

# This file contains a simple code to generate synthetic image to simulate the data obtained by a polarimeter like IAGPOL
# It also reduce the images using astropop routines.

# from astropy.io import fits
# from astropy.table import vstack
from astropy.stats import gaussian_fwhm_to_sigma
# from astropop.py_utils import mkdir_p
# from astropop.pipelines.polarimetry_scripts import run_pccdpack, process_polarimetry
import numpy as np

def gaussian_2d(x, y, x0, y0, sigma_x, sigma_y, theta, flux, sky):
    cost2 = np.cos(theta)**2
    sint2 = np.sin(theta)**2
    sin2t = np.sin(2*theta)
    sigx2 = 2*sigma_x**2
    sigy2 = 2*sigma_y**2
    a = (cost2/sigx2) + (sint2/sigy2)
    b = -(sin2t/(2*sigx2)) + (sin2t/(2*sigy2))
    c = (sint2/sigx2) + (cost2/sigy2)
    xi = x - x0
    yi = y - y0
    return sky + (flux/(np.sqrt(2*np.pi*sigma_x*sigma_y))) * np.exp(-(a*xi**2 + 2*b*xi*yi + c*yi**2))

def gen_image(scale, psi, p, theta):
    rdnoise = 18.5
    bias = 300
    coords = [(80, 135), (150, 47), (60, 60), (161, 105), (225, 150)]
    scales = [1, 0.2, 0.05, 0.1, 0.01]
    sigma = 5.0*gaussian_fwhm_to_sigma

    q = p*np.cos(2*np.radians(theta))
    u = p*np.sin(2*np.radians(theta))
    
    z = q*np.cos(4*np.radians(psi)) + u*np.sin(4*np.radians(psi))
    c = (z+1)/2
    
    base_im = np.random.normal(bias, rdnoise, (256, 256)).astype('uint16')
    
    i_y, i_x = np.indices((256, 256))
    for coo, sca in zip(coords, scales):
        im_o = gaussian_2d(i_x, i_y, coo[0], coo[1], sigma, sigma, 0, sca*scale, 0)*c
        im_e = gaussian_2d(i_x, i_y, coo[0]-20, coo[1]+20, sigma, sigma, 0, sca*scale, 0)*(1-c)
        im_o = np.round(im_o).astype('uint16')
        im_e = np.round(im_e).astype('uint16')
        
        base_im += im_o
        base_im += im_e

    return np.random.poisson(base_im).astype('uint16')

