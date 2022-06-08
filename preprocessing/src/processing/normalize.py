#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Lucas Costa Campos <rmk236@gmail.com>
#
# Distributed under terms of the MIT license.

import nibabel as nib
import numpy as np
import warnings
import argparse

def normalize(img):
    mx = np.amax(img)
    mn = np.amin(img)
    img_norm = (img - mn) / (mx - mn)
    return img_norm

def load(name):
    nifti_scan = nib.load(name)
    data=nifti_scan.get_fdata()
    return data, nifti_scan.affine, nifti_scan.header


def save(scan, affine, name):
    res = nib.Nifti1Image(scan, affine)
    nib.save(res, name)


def parse_arguments():

    parser = argparse.ArgumentParser(description='Adjusts a brain scan to become isotropic and to a certain size')

    # requiredNamed = parser.add_argument_group('required named arguments')

    parser.add_argument('-i', '--input', metavar='file', type=str,
                               nargs=1, help="""Input file""", required=True)
    parser.add_argument('-o', '--output', metavar='file', type=str,
                               nargs=1, help="""Output file""", required=True)

    args = parser.parse_args()
    return args


def main():

    args = parse_arguments()
    scan, affine, header = load(args.input[0])

    float32_scan = np.asarray(scan, dtype=np.float32)
    normed = normalize(float32_scan)
    save(normed, np.eye(4), args.output[0])


if __name__ == "__main__":
    main()
