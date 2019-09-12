#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for external library wrappers.
"""

import sys
import os.path
import ctypes
import numpy as np
import scipy as sp
import logging
import six
import xraylib as xl

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['project']



PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]
SPEED_OF_LIGHT = 299792458e+2  # [cm/s]
#LIB_MASKS = ctypes.cdll.LoadLibrary('libmasks.cpython-36m-darwin.so')
LIB_MASKS = ctypes.cdll.LoadLibrary('build/lib.linux-x86_64-3.7/libmasks.cpython-37m-x86_64-linux-gnu.so')

def project(obj, gridx, gridy, gridz, detgridx, detgridy, srcgridx, srcgridy, dsrc, ddet):
    prj = np.zeros((detgridx.size, detgridy.size), dtype='float32')
    LIB_MASKS.project.restype = as_c_void_p()
    LIB_MASKS.project(
        as_c_float_p(obj),
        as_c_int(obj.shape[0]),
        as_c_int(obj.shape[1]),
        as_c_int(obj.shape[2]),
        as_c_float_p(gridx),
        as_c_float_p(gridy),
        as_c_float_p(gridz),
        as_c_float(dsrc),
        as_c_float(ddet),
        as_c_float_p(detgridx),
        as_c_float_p(detgridy),
        as_c_int(detgridx.size),
        as_c_int(detgridy.size),
        as_c_float_p(srcgridx),
        as_c_float_p(srcgridy),
        as_c_float_p(prj),
        as_c_int(prj.shape[0]),
        as_c_int(prj.shape[1]))
    return prj


def genmask(size, type='checkerboard'):
    if type is 'checkerboard':
        re = np.r_[int(size * 0.5) * (1, 0)] # even-numbered rows
        ro = np.r_[int(size * 0.5) * (0, 1)] # odd-numbered rows
        mask = np.row_stack(int(size * 0.5) * (re, ro))
        mask = np.expand_dims(mask, 2)
    return mask.astype('float32')


def as_ndarray(arr, dtype=None, copy=False):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr, dtype=dtype, copy=copy)
    return arr


def as_dtype(arr, dtype, copy=False):
    if not arr.dtype == dtype:
        arr = np.array(arr, dtype=dtype, copy=copy)
    return arr


def as_float32(arr):
    arr = as_ndarray(arr, np.float32)
    return as_dtype(arr, np.float32)


def as_int32(arr):
    arr = as_ndarray(arr, np.int32)
    return as_dtype(arr, np.int32)


def as_uint16(arr):
    arr = as_ndarray(arr, np.uint16)
    return as_dtype(arr, np.uint16)


def as_uint8(arr):
    arr = as_ndarray(arr, np.uint8)
    return as_dtype(arr, np.uint8)


def as_c_float_p(arr):
    c_float_p = ctypes.POINTER(ctypes.c_float)
    return arr.ctypes.data_as(c_float_p)


def as_c_int(arr):
    return ctypes.c_int(arr)


def as_c_int_p(arr):
    c_int_p = ctypes.POINTER(ctypes.c_int)
    return arr.ctypes.data_as(c_int_p)


def as_c_float(arr):
    return ctypes.c_float(arr)


def as_c_char_p(arr):
    return ctypes.c_char_p(six.b(arr))


def as_c_void_p():
    return ctypes.POINTER(ctypes.c_void_p)