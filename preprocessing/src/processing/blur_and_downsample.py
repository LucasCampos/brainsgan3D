#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Lucas Costa Campos <rmk236@gmail.com>
#
# Distributed under terms of the MIT license.

import nibabel as nib
import numpy as np
import argparse

from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter

def gaussian_blur(scan, blur_sigma):
    res = gaussian_filter(scan, blur_sigma)
    return res

def downsample(scan, downsampling_factor):

    factors = np.ones(3)/downsampling_factor
    res = zoom(scan, factors, prefilter=False, order=0)
    return res


def load(name):
    nifti_scan = nib.load(name)
    data=nifti_scan.get_fdata()
    return data, nifti_scan.affine, nifti_scan.header


def save(scan, affine, name):
    scan = np.asarray(scan, dtype=np.float32)
    res = nib.Nifti1Image(scan, affine)
    nib.save(res, name)


def parse_arguments():

    parser = argparse.ArgumentParser(description='Adjusts a brain scan to become isotropic and to a certain size')

    # requiredNamed = parser.add_argument_group('required named arguments')

    parser.add_argument('-i', '--input', metavar='file', type=str,
                               nargs=1, help="""Input file""", required=True)
    parser.add_argument('-o', '--output', metavar='file', type=str,
                               nargs=1, help="""Output file""", required=True)
    parser.add_argument('-f', '--factor', metavar='factor', type=float,
                               nargs=1, help="""Downsampling factor""", required=True)
    parser.add_argument('-s', '--blur_sigma', metavar='sigma', type=float,
                               nargs=1, default=1., help="""Sigma for gaussian blur""", required=False)

    args = parser.parse_args()
    return args


def main():

    args = parse_arguments()
    scan, affine, header=load(args.input[0])
    blurred_scan = gaussian_blur(scan, args.blur_sigma[0])
    low_res_scan = downsample(blurred_scan, args.factor)
    save(low_res_scan, affine, args.output[0])


if __name__ == "__main__":
    main()
