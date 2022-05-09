#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015 IAS / CNRS / Univ. Paris-Sud
# BSD License - see attached LICENSE file
# Author: Alexandre Boucaud <alexandre.boucaud@ias.u-psud.fr>

"""
addpixscl
---------
Write the pixel scale in FITS file headers

Usage:
  addpixscl fits_files pixel_scale [--ext EXT]
  addpixscl (-h | --help)

Example:
  addpixscl psf_*.fits 0.1
"""
from __future__ import absolute_import, print_function, division

import sys

from .parser import ThrowingArgumentParser, ArgumentParserError
from .fitsutils import has_pixelscale, write_pixelscale


def parse_args():
    """Argument parser for the command line interface of `addpixscl`"""
    parser = ThrowingArgumentParser(
        description='Write the pixel scale in FITS file headers')

    parser.add_argument('fits_files', nargs='+', type=str,
                        help='Name of FITS files')

    parser.add_argument('pixel_scale', type=float,
                        help='Pixel scale value in arcseconds')

    parser.add_argument('-e', '--ext', type=int, default=0,
                        help='FITS extension number')

    return parser.parse_args()


def main():  # pragma: no cover
    """Main script for addpixscl"""
    try:
        args = parse_args()
    except ArgumentParserError:
        print(__doc__)
        sys.exit()

    for fits_file in args.fits_files:
        if has_pixelscale(fits_file):
            print("Found keywords refering to the pixel scale "
                  "in {0} header.".format(fits_file))
            continue

        write_pixelscale(fits_file, args.pixel_scale, args.ext)
