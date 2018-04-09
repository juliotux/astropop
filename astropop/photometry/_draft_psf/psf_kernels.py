'''
Kernels for psf fitting in the psf_fitting module. May be optimized with numba.
'''

from ..math.gaussian import gaussian_r, gaussian_2d
from ..math.moffat import moffat_r, moffat_2d

moffat_radial = moffat_r
moffat_spatial = moffat_2d
gaussian_radial = gaussian_r
gaussian_spatial = gaussian_2d
