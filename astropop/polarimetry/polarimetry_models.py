import numpy as np
from astropy.modeling import custom_model

# from ._vectorize import vectorize, vectorize_target


# @vectorize('float64(float64,float64,float64,float64)',
#            target=vectorize_target)
def quarter(psi, q=1.0, u=1.0, v=1.0):
    '''Polarimetry z(psi) model for quarter wavelenght retarder.

    Z= Q*cos(2psi)**2 + U*sin(2psi)*cos(2psi) - V*sin(2psi)'''
    psi2 = 2*psi
    z = q*(np.cos(psi2)**2) + u*np.sin(psi)*np.cos(psi2) - v*np.sin(psi2)
    return z


# @vectorize('float64(float64,float64,float64,float64)',
#            target=vectorize_target)
def quarter_deriv(psi, q=1.0, u=1.0, v=1.0):
    x = 2*psi
    dq = np.cos(x)**2
    du = 0.5*np.sin(2*x)
    dv = -np.sin(2*x)
    return (dq, du, dv)


# @vectorize('float64(float64,float64,float64)',
#            target=vectorize_target)
def half(psi, q=1.0, u=1.0):
    '''Polarimetry z(psi) model for half wavelenght retarder.

    Z(I)= Q*cos(4psi(I)) + U*sin(4psi(I))'''
    return q*np.cos(4*psi) + u*np.sin(4*psi)


# @vectorize('float64(float64,float64,float64)',
#            target=vectorize_target)
def half_deriv(psi, q=1.0, u=1.0):
    return (np.cos(4*psi), np.sin(4*psi))


HalfWaveModel = custom_model(half, fit_deriv=half_deriv)
QuarterWaveModel = custom_model(quarter, fit_deriv=quarter_deriv)
