import numpy as np
from .ds9norm import DS9Normalize
from .skyview import get_dss_image


def plot_vector(ax, x, y, p, theta, scale, **kwargs):
    '''Plot a single vector centered in xy, with modulus p and angle theta.
    scale = p/vector_size
    '''
    # Compute dx and dy for the vectors
    dx = p*np.cos(np.radians(theta))/(2*scale)
    dy = p*np.sin(np.radians(theta))/(2*scale)

    ax.plot([x+dx, x-dx], [y+dy, y-dy], '-', **kwargs)


_default_map_params = {'clip_lo': 0,
                       'clip_hi': 98,
                       'stretch': 'linear',
                       'bias': 0.5,
                       'contrast': 1.0,
                       'alpha': 1.0,
                       'cmap': 'Greys'}

_default_vec_params = {'color': 'r',
                       'alpha': 1.0,
                       'lw': 1.0}


def plot_polarimetry_field(ax, x, y, p, theta, wcs=None,
                           image_shape=None, survey=None,
                           scale=1.0/0.1, map_params={},
                           vector_params={}):
    '''Plot polarimetry data for all stars of a field.

    Parameters:
        ax : matplotlib axes
        x, y : x and y coordinates of the stars in the image.
        p, theta : P and Theta polarimetric parameters for the stars.
        image_shape : shape of the original image. tuple. ex: (1024, 1024)
        wcs : the WCS of the original field.
        survey : any astroquey's SkyView compatible survey. If None, no image
                 will be plot.
        scale : scale of the polarimetry vectors. In P%/pixel.
        map_params : parameters to be passed to DS9Normalize and imshow.
        vector_params : parameters to be passed to ax.plot for vetors
    '''
    if survey is not None:
        dss = get_dss_image(image_shape, wcs, survey=survey)
        ds9kwargs = {}
        for i in ['stretch', 'clip_lo', 'clip_hi', 'bias', 'contrast']:
            ds9kwargs[i] = map_params.pop(i, _default_map_params[i])
        norm = DS9Normalize(**ds9kwargs)
        ax.imshow(dss, origin='lower', norm=norm,
                  cmap=map_params.pop('cmap', _default_map_params['cmap']),
                  alpha=map_params.pop('alpha', _default_map_params['alpha']))

    # Plot the vectors as lines in the field
    for xi, yi, pi, ti in zip(x, y, p, theta):
        vecpars = _default_vec_params.copy()
        vecpars.update(vector_params)
        plot_vector(ax, xi, yi, pi, ti, scale=scale, **vecpars)
    return ax
