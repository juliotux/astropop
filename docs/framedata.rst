FramaData Container
===================

``FrameData`` is a special container to store important data of astronomical images, just like ``astropy.CCDData`` instances. It is designed to handle data, uncertainties, masks, phyisical units, metadata, memmapping, and other things. However, it is not fully compatible with ``astropy.CCDData`` due to important design differences that are needed to Astropop.

FrameData Usage
---------------

.. TODO:: Usage

Metadata
--------

.. TODO:: Metadata and header handling

Masks and Uncertainties
-----------------------

.. TODO:: Uncertainty storage and masking

Data IO
-------

.. TODO:: Data IO (FITS, CCDData, HDULists, HDUs, etc.)

FrameData API
---------------
.. automodapi:: astropop.framedata
    :no-inheritance-diagram:
    :noindex: