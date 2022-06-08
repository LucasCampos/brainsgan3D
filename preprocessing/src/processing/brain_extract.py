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

def brain_extract(brain, mask):
    res = np.multiply(brain, mask)
    return res


def load(name):
    nifti_scan = nib.load(name)
    data=nifti_scan.get_fdata()
    return data, nifti_scan.affine

def save(scan, affine, name):
    scan = np.asarray(scan, dtype=np.float32)
    res = nib.Nifti1Image(scan, affine)
    nib.save(res, name)


def parse_arguments():

    parser = argparse.ArgumentParser(description='Adjusts a brain scan to become isotropic and to a certain size')

    # requiredNamed = parser.add_argument_group('required named arguments')

    parser.add_argument('-i', '--input', metavar='files', type=str,
                               nargs=2, help="""Input files. First one: brain, Second one: mask""", required=True)
    parser.add_argument('-o', '--output', metavar='file', type=str,
                               nargs=1, help="""Output file""", required=True)

    args = parser.parse_args()
    return args


def main():

    args = parse_arguments()
    brain_scan, brain_affine=load(args.input[0])
    mask_scan, _ = load(args.input[1])
    brain_extracted_scan = brain_extract(brain_scan, mask_scan)
    save(brain_extracted_scan, brain_affine, args.output[0])


if __name__ == "__main__":
    main()
