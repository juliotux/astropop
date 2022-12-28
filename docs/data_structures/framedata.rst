.. include:: ../references.txt

FramaData Container
===================

|FrameData| is a special container to store important data of astronomical images, just like |CCDData| instances. It is designed to handle data, uncertainties, masks, phyisical units, metadata, memmapping, and other things. However, it is not fully compatible with |CCDData| due to important design differences that are needed to Astropop.

FrameData Usage
---------------

.. note::
    There is no way to create |FrameData| directly from |CCDData| or |HDUList|. Please, see `Data IO`_ for data interchanging.

.. TODO:: Usage

Metadata
--------

.. TODO:: Metadata and header handling

Masks and Uncertainties
-----------------------

.. TODO:: Uncertainty storage and masking

.. _Data IO:

Data IO
-------

.. TODO:: Data IO (FITS, |CCDData|, |HDUList|, HDUs, etc.)

FrameData API
---------------

.. automodapi:: astropop.framedata
    :no-inheritance-diagram:
