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


def plot_polarimetry_field(ax, x, y, p, theta, wcs=None,
                           image_shape=None, survey=None,
                           scale=1.0/0.1, ds9_clip=(0, 98),
                           ds9_stretch='linear', ds9_bias=0.5,
                           ds9_contrast=1.0, ds9_cmap='Greys',
                           ds9_alpha=1.0,
                           vector_color='r', vector_alpha=1.0,
                           vector_width=1.0):
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
    '''
    if survey is not None:
        dss = get_dss_image(image_shape, wcs, survey=survey)
        norm = DS9Normalize(stretch=ds9_stretch, clip_lo=ds9_clip[0],
                            clip_hi=ds9_clip[1], bias=ds9_bias,
                            contrast=ds9_contrast)
        ax.imshow(dss, origin='lower', norm=norm, cmap=ds9_cmap,
                  alpha=ds9_alpha)

    # Plot the vectors as lines in the field
    for xi, yi, pi, ti in zip(x, y, p, theta):
        plot_vector(ax, xi, yi, pi, ti, scale=scale, color=vector_color,
                    alpha=vector_alpha, lw=vector_width)
    return ax
