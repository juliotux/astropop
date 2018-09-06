*************************
Astropop Documentation
*************************

Documentation under construction and code under heavy development.


Installing
==========

This code is developed for Python>=3.5, so it will no run on previous versions.

To manual build the code, you have to install all the code depencencies::

   astropy
   sep
   numpy
   scipy
   scikit-image
   matplotlib
   cython
   astroquery
   fortranformat
   astroscrappy
   
Checkout the code in your computer::

   git clone https://github.com/juliotux/astropop.git ~/astropop
   
Build and install using Python setup tools::
   
   cd ~/astropop
   python setup.py install

Using pip:
----------

This code, for now, can be installed only from the Github repository. Soon, it will be available as PyPi and Anaconda Packages.

You can install it using `pip` directly on the github repo, with::

   pip install git+https://github.com/juliotux/astropop.git
   
Or by using the lastest version .zip download in github::

   pip install https://github.com/juliotux/astropop/archive/master.zip

Tutorials
=========

.. toctree::
   :maxdepth:2

   tutorials


API Reference
=============

.. toctree::
   :maxdepth:1

   references
