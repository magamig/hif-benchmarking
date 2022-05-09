#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015 IAS / CNRS / Univ. Paris-Sud
# BSD License - see attached LICENSE file
# Author: Alexandre Boucaud <alexandre.boucaud@ias.u-psud.fr>

"""
parser.py
---------
Custom parser that wrap up the common argparse ArgumentParser

"""
from __future__ import absolute_import

import argparse


class ArgumentParserError(Exception):
    """Custom exception for the ThrowingArgumentParser"""
    pass


class ThrowingArgumentParser(argparse.ArgumentParser):
    """
    Wrapper around ArgumentParser to overwrite the raised error

    When an error occurs, this parser throws an ArgumentParserError
    instead of exiting the code and printing a very basic usage form.

    The ArgumentParserError exception can thus be caught to do
    something else like printing your own docstring.

    Reference
    ---------
    http://stackoverflow.com/a/14728477

    """
    def error(self, message):
        """Custom empty error"""
        raise ArgumentParserError(message)
