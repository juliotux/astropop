# Licensed under a 3-clause BSD style license - see LICENSE.rst
'''Functions to easy handle with astronomical coordinates.'''

from astropy.coordinates import SkyCoord
from astropy import units

__all__ = ['guess_coordinates']


def guess_coordinates(ra, dec):
    """Try to guess the format or ra and dec passed."""
    try:
        ra = float(ra)
        dec = float(dec)
        return SkyCoord(ra, dec, unit=(units.deg, units.deg))
    except ValueError:
        # Assume (at least for now) that it's in sexagesimal
        return SkyCoord(ra, dec, unit=(units.hourangle, units.deg))
