ROADMAP
-------

Image Processing
================
    [ ] Study the use of ccdproc or builtin functions
        [ ] Base image container: astropy.nddata.ccddata
    [ ] Refactor the image processing to use Astropy's cdproc (if above)
        [ ] Wrap cdproc to use better logging
        [ ] Refactor IO, instrument specific
        [ ] Refactor processing, registering and combining
        [ ] Check/test conventions
    [ ] Better slicing support (astropy.nddata.Cutout2D)

File Management
===============
    [ ] Refactor file collections to better performance
    [ ] Refine the standarts

Pipelines
=========
    [ ] Impacton
    [ ] OPD ROBO40
        [ ] .fz handling
    [ ] OPD cameras (IAGPOL and CAMs)
    [ ] OPD old cameras
    [ ] AAVSO BSM data

Config System
=============
    [ ] Implement a system file configuration

Fits Utility
============
    [ ] Complete refactor according the processing and instrument needs

Testing
=======
    [ ] Implement unit tests for every single task
    [ ] Integrate Travis-CI and Coveralls