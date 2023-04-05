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

  .. plot::
    :include-source: False

    import numpy as np
    import matplotlib.pyplot as plt
    from astropop.polarimetry.dualbeam import halfwave_model

    psi_lin = np.linspace(0, 360, 600)
    psi_i = np.arange(0, 361, 22.5)
    z_i = halfwave_model(psi_i, 0.12, -0.07)
    z_lin = halfwave_model(psi_lin, 0.12, -0.07)

    plt.figure(figsize=(5, 3))
    plt.plot(psi_lin, z_lin, '-', label='Model')
    plt.plot(psi_i, z_i, 'o', label='Data Points')
    plt.xlabel(r'$\psi_i$')
    plt.ylabel(r'$Z_i$')
    plt.title('Example of half-wave plate\nQ=0.12, U=-0.07')
    plt.legend()
    plt.tight_layout()
    plt.show()

* For quarter-wave plates, linear and circular polarization is computed, and the equation is:

  .. math::

    Z_i = Q \cos^2{2 \psi_i} + U \sin{2 \psi_i} \cos{2 \psi_i} - V \sin{2 \psi_i}

  .. plot::
    :include-source: False

    import numpy as np
    import matplotlib.pyplot as plt
    from astropop.polarimetry.dualbeam import quarterwave_model

    psi_lin = np.linspace(0, 360, 600)
    psi_i = np.arange(0, 361, 22.5)
    z_i = quarterwave_model(psi_i, 0.05, -0.07, 0.08)
    z_lin = quarterwave_model(psi_lin, 0.05, -0.07, 0.08)

    plt.figure(figsize=(5, 3))
    plt.plot(psi_lin, z_lin, '-', label='Model')
    plt.plot(psi_i, z_i, 'o', label='Data Points')
    plt.xlabel(r'$\psi_i$')
    plt.ylabel(r'$Z_i$')
    plt.legend()
    plt.title('Example of quarter-wave plate\nQ=0.05, U=-0.07, V=0.08')
    plt.tight_layout()
    plt.show()

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

There are 2 main steps in the polarimetry reduction process: the pairs matching and the Stokes parameters computation.

To compute the Stokes parameters of a given source, |astropop| uses arrays of fluxes and retarder positions. So, it is not directly dependent on the photometry routines to process the data and you can use your own already extracted photometry for this process. Also, our helper functions to match the pairs of ordinary and extraordinary beams also use just arrays of x and y positions, and are not dependent on the photometry routines too. So, **no part of the polarimetry reduction depend on the use of |astropop| for source extraction and photometry and you can use your own data or code in this process**.

For example, in this documentation we will use totally fake photometry and random pairs positions to illustrate the polarimetry reduction process.


Stokes-Least-Squares Polarimetry Processing
-------------------------------------------

The Stokes-Least-Squares (SLS) method is a method to compute the Stokes parameters by fitting the modulation equations described above to the relative flux difference of the data using a least-squares method with Trust Region Reflective algorithm. For a better description and validation of this method, see [5]_.

First of all, lets create the fake data to be used here for example.

.. ipython:: python

  import numpy as np
  from astropop.polarimetry import halfwave_model, quarterwave_model
  flux = 1e7  # total flux of the source, in electrons
  # retarder positions, in degrees
  psi_i = np.arange(0, 360, 22.5)

  # half-wave plate with only linear polarization
  # q=0.05, u=-0.07
  z_i_half = halfwave_model(psi_i, 0.05, -0.07)
  fo_half = flux*(1+z_i_half)/2  # ordinary beam fluxes
  fe_half = flux*(1-z_i_half)/2  # extraordinary beam fluxes

  # quarter-wave plate with linear and circular polarization
  # q=0.05, u=-0.07, v=0.08
  z_i_quarter = quarterwave_model(psi_i, 0.05, -0.07, 0.08)
  fo_quarter = flux*(1+z_i_quarter)/2  # ordinary beam fluxes
  fe_quarter = flux*(1-z_i_quarter)/2  # extraordinary beam fluxes


.. plot::
  :include-source: False
  :caption: Simulated data for the Stokes-Least-Squares polarimetry processing.

  import numpy as np
  from astropop.polarimetry import halfwave_model, quarterwave_model
  flux = 1e7  # total flux of the source, in electrons
  # retarder positions, in degrees
  psi_i = np.arange(0, 360, 22.5)
  lin = np.linspace(0, 360, 360)

  # half-wave plate with only linear polarization
  # q=0.05, u=-0.07
  z_i_half = halfwave_model(psi_i, 0.05, -0.07)
  fo_half = flux*(1+z_i_half)/2  # ordinary beam fluxes
  fe_half = flux*(1-z_i_half)/2  # extraordinary beam fluxes
  lin_e_half = flux*(1-halfwave_model(lin, 0.05, -0.07))/2
  lin_o_half = flux*(1+halfwave_model(lin, 0.05, -0.07))/2

  # quarter-wave plate with linear and circular polarization
  # q=0.05, u=-0.07, v=0.08
  z_i_quarter = quarterwave_model(psi_i, 0.05, -0.07, 0.08)
  fo_quarter = flux*(1+z_i_quarter)/2  # ordinary beam fluxes
  fe_quarter = flux*(1-z_i_quarter)/2  # extraordinary beam fluxes
  lin_e_quarter = flux*(1-quarterwave_model(lin, 0.05, -0.07, 0.08))/2
  lin_o_quarter = flux*(1+quarterwave_model(lin, 0.05, -0.07, 0.08))/2

  fig, ax = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey='row')
  ax[0][0].plot(psi_i, fo_half, 'C0o', label='Ordinary')
  ax[0][0].plot(psi_i, fe_half, 'C1o', label='Extraordinary')
  ax[0][0].plot(lin, lin_o_half, 'C0--')
  ax[0][0].plot(lin, lin_e_half, 'C1--')
  ax[0][0].legend()
  ax[0][0].set_title('Half-wave plate\nQ=0.05, U=-0.07')
  ax[0][0].set_ylabel('Flux (e-)')

  ax[0][1].plot(psi_i, fo_quarter, 'C0o', label='Ordinary')
  ax[0][1].plot(psi_i, fe_quarter, 'C1o', label='Extraordinary')
  ax[0][1].plot(lin, lin_o_quarter, 'C0--')
  ax[0][1].plot(lin, lin_e_quarter, 'C1--')
  ax[0][1].legend()
  ax[0][1].set_title('Quarter-wave plate\nQ=0.05, U=-0.07, V=0.08')

  ax[1][0].plot(psi_i, z_i_half, 'C2o')
  ax[1][0].plot(lin, halfwave_model(lin, 0.05, -0.07), 'C2--')
  ax[1][0].set_ylabel(r'Relative flux difference ($Z_i$)')
  ax[1][0].set_xlabel(r'Retarder position $\psi_i$ in (deg)')

  ax[1][1].plot(psi_i, z_i_quarter, 'C2o')
  ax[1][1].plot(lin, quarterwave_model(lin, 0.05, -0.07, 0.08), 'C2--')
  ax[1][1].set_xlabel(r'Retarder position $\psi_i$ in (deg)')

  ax[1][0].set_xticks(np.arange(0, 361, 45))
  ax[1][0].set_xlim(0, 360)

  fig.tight_layout()
  plt.show()


Helper Tools
------------

.. ipython:: python

  # random positions for the pairs
  # 300 pairs of sources in a image of 1024x1024 pixels
  # dx=25, dy=35
  x = np.random.uniform(0, 1024, 300)
  x = np.concatenate((x, x+25))  # x positions of the rouces
  y = np.random.uniform(0, 1024, 300)
  y = np.concatenate((y, y+35))  # y positions of the rouces

References
----------
.. [1] Magalhães et al. 1996. `bibcode: 1996ASPC...97..118M <https://ui.adsabs.harvard.edu/abs/1996ASPC...97..118M>`_

.. [2] Rodrigues et al. 2011. `bibcode: 2012AIPC.1429..252R <https://ui.adsabs.harvard.edu/abs/2012AIPC.1429..252R>`_

.. [3] Wiersema et al. 2023. `doi: 10.1093/rasti/rzad005 <https://doi.org/10.1093/rasti/rzad005>`_

.. [4] Rodrigues et al. 1998. `bibcode: 1998A&A...335..979R <https://ui.adsabs.harvard.edu/abs/1998A&A...335..979R>`_

.. [5] Campagnolo, 2019. `bibcode: 2019PASP..131b4501N <https://ui.adsabs.harvard.edu/abs/2019PASP..131b4501N>`_

.. [6] Lima et al. 2021. `bibcode: 2021AJ....161..225L <https://ui.adsabs.harvard.edu/abs/2021AJ....161..225L>`_

Polarimetry API
---------------

.. automodapi:: astropop.polarimetry
    :no-inheritance-diagram:
