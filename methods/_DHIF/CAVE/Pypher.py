#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015 IAS / CNRS / Univ. Paris-Sud
# BSD License - see attached LICENSE file
# Author: Alexandre Boucaud <alexandre.boucaud@ias.u-psud.fr>

"""
PyPHER - Python-based PSF Homogenization kERnels
================================================

Compute the homogenization kernel between two PSFs

Usage:
  pypher psf_source psf_target output
         [-s ANGLE_SOURCE] [-t ANGLE_TARGET] [-r REG_FACT]
  pypher (-h | --help)

Example:
  pypher psf_a.fits psf_b.fits kernel_a_to_b.fits -r 1.e-5
"""
from __future__ import absolute_import, print_function, division

import os
import sys
import logging
import logging.handlers
import argparse
import numpy as np

from scipy.ndimage import rotate, zoom

# from . import fitsutils as fits
# from .parser import ThrowingArgumentParser, ArgumentParserError

__version__ = '0.6.4'


# def parse_args():
#     """Argument parser for the command line interface of `pypher`"""
#     parser = ThrowingArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#         prog='pypher',
#         description="Compute the homogenization kernel between two PSFs")
#
#     parser.add_argument('psf_source', type=str,
#                         help="FITS file of PSF image with highest resolution")
#
#     parser.add_argument('psf_target', type=str,
#                         help="FITS file of PSF image with lowest resolution")
#
#     parser.add_argument('output', type=str,
#                         help="File name for the output kernel")
#
#     parser.add_argument('-s', '--angle_source', type=float, default=0.0,
#                         help="Rotation angle to apply to `psf_source` (deg)")
#
#     parser.add_argument('-t', '--angle_target', type=float, default=0.0,
#                         help="Rotation angle to apply to `psf_target` (deg)")
#
#     parser.add_argument('-r', '--reg_fact', type=float, default=1.e-4,
#                         help="Regularisation parameter for the Wiener filter")
#
#     return parser.parse_args()

################
# IMAGE METHODS
################


# def format_kernel_header(fits_file, args, pixel_scale):
#     """
#     Write the input parameters of pypher as comments in the header
#
#     The kernel header therefore contains the name of the PSF files
#     it has been created from.
#     The pixel scale of the kernel is also written as a dedicated
#     kernel key.
#
#     Parameters
#     ----------
#     fits_file: str
#         Path to the FITS kernel image
#     args: `argparse.Namespace`
#         Container for the parsed values
#     pixel_scale: float
#         Pixel scale of the kernel
#
#     """
#     fits.clear_comments(fits_file)
#
#     pypher_comments = [
#         '=' * 50, '',
#         'File written with PyPHER',
#         '------------------------', '',
#         'Kernel from PSF', '',
#         '=> {0}'.format(os.path.basename(args.psf_source)), '',
#         'to PSF', '',
#         '=> {0}'.format(os.path.basename(args.psf_target)), '',
#         'using a regularisation parameter '
#         'R = {0:1.1e}'.format(args.reg_fact), '',
#         '=' * 50
#     ]
#     fits.add_comments(fits_file, pypher_comments)
#
#     fits.write_pixelscale(fits_file, pixel_scale)


def imrotate(image, angle, interp_order=1):
    """
    Rotate an image from North to East given an angle in degrees

    Parameters
    ----------
    image : `numpy.ndarray`
        Input data array
    angle : float
        Angle in degrees
    interp_order : int, optional
        Spline interpolation order [0, 5] (default 1: linear)

    Returns
    -------
    output : `numpy.ndarray`
        Rotated data array

    """
    return rotate(image, -1.0 * angle,
                  order=interp_order, reshape=False, prefilter=False)


def imresample(image, source_pscale, target_pscale, interp_order=1):
    """
    Resample data array from one pixel scale to another

    The resampling ensures the parity of the image is conserved
    to preserve the centering.

    Parameters
    ----------
    image : `numpy.ndarray`
        Input data array
    source_pscale : float
        Pixel scale of ``image`` in arcseconds
    target_pscale : float
        Pixel scale of output array in arcseconds
    interp_order : int, optional
        Spline interpolation order [0, 5] (default 1: linear)

    Returns
    -------
    output : `numpy.ndarray`
        Resampled data array

    """
    old_size = image.shape[0]
    new_size_raw = old_size * source_pscale / target_pscale
    new_size = int(np.ceil(new_size_raw))

    if new_size > 10000:
        raise MemoryError("The resampling will yield a too large image. "
                          "Please resize the input PSF image.")

    # Chech for parity
    if (old_size - new_size) % 2 == 1:
        new_size += 1

    ratio = new_size / old_size

    return zoom(image, ratio, order=interp_order) / ratio**2


def trim(image, shape):
    """
    Trim image to a given shape

    Parameters
    ----------
    image: 2D `numpy.ndarray`
        Input image
    shape: tuple of int
        Desired output shape of the image

    Returns
    -------
    new_image: 2D `numpy.ndarray`
        Input image trimmed

    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("TRIM: null or negative shape given")

    dshape = imshape - shape
    if np.any(dshape < 0):
        raise ValueError("TRIM: target size bigger than source one")

    if np.any(dshape % 2 != 0):
        raise ValueError("TRIM: source and target shapes "
                         "have different parity")

    idx, idy = np.indices(shape)
    offx, offy = dshape // 2

    return image[idx + offx, idy + offy]


def zero_pad(image, shape, position='corner'):
    """
    Extends image to a certain size with zeros

    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        Input image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered

    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image

    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img


##########
# FOURIER
##########


def udft2(image):
    """Unitary fft2"""
    norm = np.sqrt(image.size)
    return np.fft.fft2(image) / norm


def uidft2(image):
    """Unitary ifft2"""
    norm = np.sqrt(image.size)
    return np.fft.ifft2(image) * norm


def psf2otf(psf, shape):
    """
    Convert point-spread function to optical transfer function.

    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.

    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.

    Parameters
    ----------
    psf : `numpy.ndarray`
        PSF array
    shape : int
        Output shape of the OTF array

    Returns
    -------
    otf : `numpy.ndarray`
        OTF array

    Notes
    -----
    Adapted from MATLAB psf2otf function

    """
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf


################
# DECONVOLUTION
################

LAPLACIAN = np.array([[ 0, -1,  0],
                      [-1,  4, -1],
                      [ 0, -1,  0]])


def deconv_wiener(psf, reg_fact):
    r"""
    Create a Wiener filter using a PSF image

    The signal is $\ell_2$ penalized by a 2D Laplacian operator that
    serves as a high-pass filter for the regularization process.
    The key to the process is to use optical transfer functions (OTF)
    instead of simple Fourier transform, since it ensures the phase
    of the psf is adequately placed.

    Parameters
    ----------
    psf: `numpy.ndarray`
        PSF array
    reg_fact: float
        Regularisation parameter for the Wiener filter

    Returns
    -------
    wiener: complex `numpy.ndarray`
        Fourier space Wiener filter

    """
    # Optical transfer functions
    trans_func = psf2otf(psf, psf.shape)
    reg_op = psf2otf(LAPLACIAN, psf.shape)

    wiener = np.conj(trans_func) / (np.abs(trans_func)**2 +
                                    reg_fact * np.abs(reg_op)**2)

    return wiener


def homogenization_kernel(psf_target, psf_source, reg_fact=1e-4, clip=True):
    r"""
    Compute the homogenization kernel to match two PSFs

    The deconvolution step is done using a Wiener filter with $\ell_2$
    penalization.
    The output is given both in Fourier and in the image domain to serve
    different purposes.

    Parameters
    ----------
    psf_target: `numpy.ndarray`
        2D array
    psf_source: `numpy.ndarray`
        2D array
    reg_fact: float, optional
        Regularisation parameter for the Wiener filter
    clip: bool, optional
        If `True`, enforces the non-amplification of the noise
        (default `True`)

    Returns
    -------
    kernel_image: `numpy.ndarray`
        2D deconvolved image
    kernel_fourier: `numpy.ndarray`
        2D discrete Fourier transform of deconvolved image

    """
    wiener = deconv_wiener(psf_source, reg_fact)

    kernel_fourier = wiener * udft2(psf_target)
    kernel_image = np.real(uidft2(kernel_fourier))

    if clip:
        kernel_image.clip(-1, 1)

    return kernel_image, kernel_fourier


########
# DEBUG
########


def setup_logger(log_filename='pypher.log'):  # pragma: no cover
    """
    Set up and return a logger

    The logger records the time, modulename, method and message

    Parameters
    ----------
    log_filename: str
        Name of the output logfile

    """
    # create logger
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    # Add the log message handler to the logger
    handler = logging.handlers.RotatingFileHandler(log_filename)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - '
                                  '%(module)s - '
                                  '%(levelname)s - '
                                  '%(message)s')
    handler.setFormatter(formatter)
    # add handler to logger
    logger.addHandler(handler)

    return logger


#######
# MAIN
#######


# def main():  # pragma: no cover
#     """Main script for pypher"""
#     try:
#         args = parse_args()
#     except ArgumentParserError:
#         print(__doc__)
#         sys.exit()
#
#     kernel_basename, _ = os.path.splitext(args.output)
#     kernel_fits = kernel_basename + '.fits'
#
#     logname = '%s.log' % kernel_basename
#     if os.path.exists(logname):
#         os.remove(logname)
#     log = setup_logger(logname)
#
#     # Load images (NaNs are set to 0)
#     psf_source = fits.getdata(args.psf_source)
#     psf_target = fits.getdata(args.psf_target)
#
#     log.info('Source PSF loaded: %s', args.psf_source)
#     log.info('Target PSF loaded: %s', args.psf_target)
#
#     # Set NaNs to 0.0
#     psf_source = np.nan_to_num(psf_source)
#     psf_target = np.nan_to_num(psf_target)
#
#     # Retrieve the pixel scale of each image
#     pixscale_source = fits.get_pixscale(args.psf_source)
#     pixscale_target = fits.get_pixscale(args.psf_target)
#
#     log.info('Source PSF pixel scale: %.2f arcsec', pixscale_source)
#     log.info('Target PSF pixel scale: %.2f arcsec', pixscale_target)
#
#     # Rotate images (if necessary)
#     if args.angle_source != 0.0:
#         psf_source = imrotate(psf_source, args.angle_source)
#     if args.angle_target != 0.0:
#         psf_target = imrotate(psf_target, args.angle_target)
#
#     log.info('Source PSF rotated by %.2f degrees', args.angle_source)
#     log.info('Target PSF rotated by %.2f degrees', args.angle_target)
#
#     # Normalize the PSFs
#     psf_source /= psf_source.sum()
#     psf_target /= psf_target.sum()
#
#     # Resample high resolution image to the low one
#     if pixscale_source != pixscale_target:
#         try:
#             psf_source = imresample(psf_source,
#                                     pixscale_source,
#                                     pixscale_target)
#         except MemoryError:
#             log.error('- COMPUTATION ABORTED -')
#             log.error('The size of the resampled PSF would have '
#                       'exceeded 10K x 10K')
#             log.error('Please resize your image and try again')
#
#             print('Issue during the resampling step - see pypher.log')
#             sys.exit()
#
#         log.info('Source PSF resampled to the target pixel scale')
#
#     # check the new size of the source vs. the target
#     if psf_source.shape > psf_target.shape:
#         psf_source = trim(psf_source, psf_target.shape)
#     else:
#         psf_source = zero_pad(psf_source, psf_target.shape, position='center')
#
#     kernel, _ = homogenization_kernel(psf_target, psf_source,
#                                       reg_fact=args.reg_fact)
#
#     log.info('Kernel computed using Wiener filtering and a regularisation '
#              'parameter r = %.2e', args.reg_fact)
#
#     # Write kernel to FITS file
#     fits.writeto(kernel_fits, data=kernel)
#     format_kernel_header(kernel_fits, args, pixscale_target)
#
#     log.info('Kernel saved in %s', kernel_fits)
#
#     print("pypher: Output kernel saved to %s" % kernel_fits)
#
#
# if __name__ == '__main__':
#     main()
