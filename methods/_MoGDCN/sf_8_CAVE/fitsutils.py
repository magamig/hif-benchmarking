#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015 IAS / CNRS / Univ. Paris-Sud
# BSD License - see attached LICENSE file
# Author: Alexandre Boucaud <alexandre.boucaud@ias.u-psud.fr>

"""
fitsutils.py
------------
A set of convenience methods to deal with FITS files

"""
from __future__ import absolute_import, print_function, division

import astropy.io.fits as pyfits
from astropy.io.fits import getdata, writeto


PIXSCL_KEY_DEG = ['CD1_1', 'CD2_2', 'CDELT1', 'CDELT2']
PIXSCL_KEY_ARCSEC = ['PIXSCALE', 'SECPIX', 'PIXSCALX', 'PIXSCALY']
PIXSCL_KEYS = PIXSCL_KEY_DEG + PIXSCL_KEY_ARCSEC


def has_pixelscale(fits_file):
    """
    Find pixel scale keywords in FITS file

    Parameters
    ----------
    fits_file: str
        Path to a FITS image file

    """
    header = pyfits.getheader(fits_file)
    return [key
            for key in PIXSCL_KEYS
            if key in list(header.keys())]


def write_pixelscale(fits_file, value, ext=0):
    """
    Write pixel scale information to a FITS file header

    The input pixel scale value is given in arcseconds but is stored
    in degrees since the chosen header KEYS are the linear
    transformation matrix parameters CDi_ja

    Parameters
    ----------
    fits_file: str
        Path to a FITS image file
    value: float
        Pixel scale value in arcseconds
    ext: int, optional
        Extension number in the FITS file

    """
    pixscl = value / 3600
    comment = 'Linear transformation matrix'

    pyfits.setval(fits_file, 'CD1_1', value=pixscl, ext=ext, comment=comment)
    pyfits.setval(fits_file, 'CD1_2', value=0.0, ext=ext, comment=comment)
    pyfits.setval(fits_file, 'CD2_1', value=0.0, ext=ext, comment=comment)
    pyfits.setval(fits_file, 'CD2_2', value=pixscl, ext=ext, comment=comment)


def get_pixscale(fits_file):
    """
    Retreive the image pixel scale from its FITS header

    Parameters
    ----------
    fits_file: str
        Path to a FITS image file

    Returns
    -------
    pixel_scale: float
        The pixel scale of the image in arcseconds

    """
    pixel_keys = has_pixelscale(fits_file)

    if not pixel_keys:
        raise IOError("Pixel scale not found in {0}.".format(fits_file))

    pixel_key = pixel_keys.pop()
    pixel_scale = abs(pyfits.getval(fits_file, pixel_key))

    if pixel_key in PIXSCL_KEY_DEG:
        pixel_scale *= 3600

    return round(pixel_scale, 6)


def clear_comments(fits_file):
    """
    Delete the COMMENTS in the FITS header

    Parameters
    ----------
    fits_file: str
        Path to a FITS image file

    """
    for comment_key in ['COMMENT', 'comment', 'Comment']:
        try:
            pyfits.delval(fits_file, comment_key)
        except KeyError:
            pass


def add_comments(fits_file, values):
    """
    Add comments to the FITS header

    Parameters
    ----------
    fits_file: str
        Path to a FITS image file
    values: str or str list
        Comment(s) to add

    """
    if isinstance(values, str):
        pyfits.setval(fits_file, 'COMMENT', value=values)
    else:
        for value in values:
            pyfits.setval(fits_file, 'COMMENT', value=value)
