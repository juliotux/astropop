# Licensed under a 3-clause BSD style license - see LICENSE.rst

from ..common.singleccd import SingleCCDCamera
from ...fits_utils import check_image_hdu


__all__ = ['OPDAcqCamera', 'OPDIxonCamera', 'OPDIkonLCamera']


class OPDAcqCamera(SingleCCDCamera):
    """Provide the interface to data obatined with OPDAcq software."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._read_config(self._config_filename)

    def read_raw_file(self, filename):
        # ODPAcq save the image in the first HDU
        return check_image_hdu(filename, hdu=0)


class OPDIxonCamera(OPDAcqCamera):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class OPDIkonLCamera(OPDAcqCamera):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
