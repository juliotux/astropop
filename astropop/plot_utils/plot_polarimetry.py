import numpy as np
from .ds9norm import DS9Normalize
from .skyview import get_dss_image


def plot_polarimetry_field(ax, x, y, p, theta, wcs,
                           image_shape, survey='DSS',
                           scale=1.0/20, ds9_clip=(0, 98),
                           ds9_stretch='linear', ds9_bias=0.5,
                           ds9_contrast=1.0, ds9_cmap='Greys',
                           vector_color='r'):
    '''Plot polarimetry data for all stars of a field.

    Parameters:
        ax : matplotlib axes
        x, y : x and y coordinates of the stars in the image.
        p, theta : P and Theta polarimetric parameters for the stars.
        image_shape : shape of the original image. tuple. ex: (1024, 1024)
        wcs : the WCS of the original field.
        survey : any astroquey's SkyView compatible survey.
        scale : scale of the polarimetry vectors. In P%/pixel.
    '''
    dss = get_dss_image(image_shape, wcs, survey=survey)
    norm = DS9Normalize(stretch=ds9_stretch, clip_lo=ds9_clip[0],
                        clip_hi=ds9_clip[1], bias=ds9_bias,
                        contrast=ds9_contrast)
    ax.imshow(dss, origin='lower', norm=norm, cmap=ds9_cmap)

    # Compute dx and dy for the vectors
    dx = p*scale*np.cos(theta)
    dy = p*scale*np.sin(theta)

    # Plot the vectors as lines in the field
    ax.plot(list(zip(x+dx, x-dx)), list(zip(y+dy, y-dy)), '-',
            color=vector_color)
    return ax
