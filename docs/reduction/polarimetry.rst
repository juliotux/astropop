.. include:: ../references.txt

Polarimetry Processing
======================

|astropop| polarimetry computation currently is desgined to work only with dual-beam polarimeters with half-wave plate (HWP) and quarter-wave plate (QWP) retarders, performing both circular and linear polarization computations, according the instrument capabilities and design. Savart or Wollaston analyzers are supported. For example of instruments that can be used with |astropop|, we already tested with success the following instruments:

* IAGPOL [1]_ - Gaveta Polarimétrica do IAG `link <http://astroweb.iag.usp.br/~polarimetria/gaveta/default.htm>`_ at Observatório Pico dos Dias, Brazil
* SPARC4 [2]_ - Simultaneous Polarimeter and Rapid Camera in Four bands `link <http://www.das.inpe.br/sparc4>`_ at Observatório Pico dos Dias, Brazil
* LE2POL [3]_ - Leicester dual-beam imaging polarimeter at University of Leicester, UK
* FORS2 - FORS2, FOcal Reducer/low dispersion Spectrograph 2 at ESO, Chile, in IPOL mode. `link <https://www.eso.org/sci/facilities/paranal/instruments/fors/overview.html>`_
* POLIMA2 - Optical Dual-Beam Polarizer at OAN-SPM, Mexico. `link <https://www.astrossp.unam.mx/en/users/instruments/ccd-imaging/polima-2>`_

Description of the Polarimetry Reduction
----------------------------------------

The reduction process overview is derived from the work of Magalhães et al. [1]_ and Rodrigues et al. [4]_ , using a fitting process better described by Campagnolo (2019) [5]_. In a few words, the code computes the normalized flux difference between the ordinary and extraordinary beams ($F_{e}$ and $F_{o}$) for each retarder plate position:

.. math::

    Z_{i} = \frac{F_{e} - F_{o}}{F_{e} + F_{o}}

And fits a equation for a given set o positions. There are two equations that can be used:

* For half-wave plates, only linear polarization is computed, and the equation is:

  .. math::

    Z_i = Q \cos{4 \psi_i} + U \sin{4 \psi_i}

* For quarter-wave plates, linear and circular polarization is computed, and the equation is:

  .. math::

    Z_i = Q \cos^2{2 \psi_i} + U \sin{2 \psi_i} \cos{2 \psi_i} - V \sin{2 \psi_i}

Where :math:`Q`, :math:`U` and :math:`V` are the linear and circular polarization fractions and $\psi_i$ is the retarder plate position.

Normalization
^^^^^^^^^^^^^

If the instrument have different sensibility for each beam or if the PSF of the two beams do not match (a common thing in Savart analyzers), a normalization :math:`k` factor must be included in the equation of the relative flux ($Z_i$). This normalization factor scales one of the beams to compensate the difference in sensibility. So, the flux difference equation becomes:

.. math::

    Z_{i} = \frac{F_{e} - F_{o} \cdot k}{F_{e} + F_{o} \cdot k}

This :math:`k` factor is computed in different ways for half-wave and quarter-wave plates.

* For half-wave plates, :math:`k` is computed as the ratio between the sum of the extraordinary and ordinary fluxes along a complete set of 4 consecutive retarder positions (0, 22.5, 45 and 67.5 degrees). The explanation is better described in [1]_. This is possible because the equation for half-wave plates is symmetrical in 0. So:

  .. math::

    k = \frac{\sum_{i=1}^{4} F_{e, i}}{\sum_{i=1}^{4} F_{o, i}}

* For quarter-wave plates, the term :math:`Q\cos^2{2 \psi_i}` in the equation is always positive, so the results are not symmetrical in 0. Due to this, the equation above cannot be used to compute :math:`k`. Rodrigues (1998) [4]_ and Lima (2021) [6]_ describe a iterative process, where :math:`k` depends on :math:`Q`. According this process, the Stokes parameters are computed with some :math:`k`, and then a new :math:`k` is computed using the new Stokes parameters. This process is repeated until the computed Stokes parameters converge within a given tolerance.

Both methods are implemented in |astropop|.

Angle Zero of the Retarder
^^^^^^^^^^^^^^^^^^^^^^^^^^

It is expected that the angle zero of the retarder plate do not match perfectly with the analyzer and the equatorial system of the sky. Due to this, the observation of standard stars may be needed to compute the correction to be applied to fix the instrumental computed position angle and the real position angle of the sky. As this is a standard, post-reduction, data processing step, we will not go deep into this topic here. An offset zero do not affect the results for linear polarization in half-wave plates.

However, for circular polarization, the zero offset is important due to the assymmetry of the equation. So, with this configuration, there is an additional offset to be computed within the Stokes computation process. This instrumental offset, if wrong, leads to wrong results and with high errors. Additionally, the best wy to compute it is by reducing data from a star with strong circular polarization level. Once this offset is determined, it can be used in other reductions, since it is commonly a instrument-dependent value.

Polarimetry Reduction in |astropop|
===================================

.. TODO:: reduction process

Stokes-Least-Squares Polarimetry Processing
-------------------------------------------

Helper Tools
------------

References
----------
.. [1] Magalhães et al. 1996. `https://ui.adsabs.harvard.edu/abs/1996ASPC...97..118M`_

.. [2] Rodrigues et al. 2011. `https://ui.adsabs.harvard.edu/abs/2012AIPC.1429..252R`_

.. [3] Wiersema et al. 2023. `https://academic.oup.com/rasti/article/2/1/106/7040587`_

.. [4] Rodrigues et al. 1998. `https://ui.adsabs.harvard.edu/abs/1998A&A...335..979R`_

.. [5] Campagnolo, 2019. `https://ui.adsabs.harvard.edu/abs/2019PASP..131b4501N`_

.. [6] Lima et al. 2021. `https://ui.adsabs.harvard.edu/abs/2021AJ....161..225L`_

Polarimetry API
---------------

.. automodapi:: astropop.polarimetry
    :no-inheritance-diagram:
