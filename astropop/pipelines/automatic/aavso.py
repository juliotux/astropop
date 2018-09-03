from astropy.time import Time, TimezoneInfo
from astropy import units
import numpy as np

from .common_photometry import StackedPhotometryPipeline
from .common_processing import PipeProd
from ...file_manager import FileManager
from ...py_utils import check_iterable
from ...logger import logger


class BSMStackedPhotometry(StackedPhotometryPipeline):
    photometry_parameters = dict(detect_snr=5,
                                 detect_fwhm=3,
                                 photometry_type='aperture',
                                 r='auto',
                                 r_ann='auto',
                                 solve_photometry_type='montecarlo',
                                 montecarlo_iters=300,
                                 montecarlo_percentage=0.2,
                                 round_lim=(-5.0, 5.0),
                                 sharp_lim=(0.0, 4.0))
    astrometry_parameters = dict(ra_key='RA',
                                 dec_key='DEC',
                                 plate_scale=[1,  10],
                                 identify_limit_angle='5 arcsec')
    combine_parameters = dict(method='sum',
                              weights=None,
                              scale=None,
                              mem_limit=1e8,
                              reject=None)
    group_keys = ['object', 'filter', 'observat', 'night']
    save_file_name = '{object}_{filter}_{night}_{observat}.fits'
    save_file_dir = 'stacked_photometry/'
    def __init__(self, product_dir, image_ext=0):
        super(BSMStackedPhotometry, self).__init__(product_dir, image_ext)
        self.fm = FileManager(image_ext, compression=True)

    def _compute_night(self, dateobs, timezone):
        t = Time(dateobs, format='isot', scale='utc')
        if timezone is not None:
            timezone = TimezoneInfo(utc_offset=timezone*units.hour)
        # Assume all images where taken in the same night
        dat = t.to_datetime(timezone)
        yyyy = dat.year
        mm = dat.month
        dd = dat.day
        # Assume one night goes from dd 12pm to nextdd-1 12pm
        if dat.hour < 12:
            dd -= 1
        return '{:04d}-{:02d}-{:02d}'.format(yyyy, mm, dd)

    def get_timezone(self, fg):
        """Get sun difference between local and UTC."""
        return np.round(np.array(fg.values('long-obs'))/15, 0).astype('int')

    def get_night(self, fg, timezone, iter=False):
        """Get date-obs(UT), lat, lon and alt and return the observing night.

        If iter is False, just the first value will be computed. Else an array
        with all values will be returned.
        """
        if iter:
            if not check_iterable(timezone):
                timezone = [timezone]*len(fg.summary)
            return [self._compute_night(dateobs, t) for
                    dateobs, t in zip(fg.values('date-obs'), timezone)]
        else:
            return self._compute_night(fg.values('date-obs')[0], timezone)

    def run(self, processed_folder, science_catalog=None, astrometry=False):
        """Process the photometry for calibed images from AAVSOnet."""
        logger.debug('Creating file groups.')
        fg = self.fm.create_filegroup(processed_folder)
        timezone = self.get_timezone(fg)
        night = self.get_night(fg, timezone, iter=True)
        fg.add_column('night', night)

        prods = []
        logger.debug('Grouping the data.')
        for g in self.fm.group_by(fg, self.group_keys):
            fname = self.get_filename(g)
            prods.append(PipeProd(g, bias=None, flat=None, dark=None,
                                  calibed_files=g.files, sci_result=fname))

        self.process_products(prods, science_catalog=science_catalog,
                              astrometry=astrometry)
