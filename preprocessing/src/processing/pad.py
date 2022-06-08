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

def pad_to_size(resliced_float32_scan, image_size_goal_shape):

    s=resliced_float32_scan.shape

    # check that the function was called with correct parameters
    # especially, ensure that the function is called with a larger goal_shape than input_shape
    axis_size_difference=np.asarray(image_size_goal_shape)-np.asarray(s)
    for i,e in enumerate(axis_size_difference):
        if e<0:
            warnings.warn("Padding dimension "+ str(i)+ " incorrect!")
            warnings.warn(str(s[i])+' in image vs. '+str(image_size_goal_shape[i])+'in desired padding-result shape')
            warnings.warn("ERROR: goal_shape < original_shape. You cannot make an image smaller by adding more padding!")
            assert False

    axis_size_difference = axis_size_difference/2

    left_pads = np.floor(axis_size_difference)
    left_pads = np.array(left_pads, dtype=np.int)
    right_pads = np.ceil(axis_size_difference)
    right_pads = np.array(right_pads, dtype=np.int)

    pads = np.array([left_pads, right_pads], dtype=np.int).transpose()
    res = np.pad(resliced_float32_scan, pads, 'constant', constant_values=0)
    return res



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
    parser.add_argument('-s', '--size', metavar='N', type=int,
                               nargs=3, help="""New size of the image, in voxels""", required=True)

    args = parser.parse_args()
    return args


def main():

    args = parse_arguments()
    scan, affine, header=load(args.input[0])

    float32_scan = np.asarray(scan, dtype=np.float32)
    padded_float32_scan = pad_to_size(float32_scan, args.size)
    save(padded_float32_scan, np.eye(4), args.output[0])


if __name__ == "__main__":
    main()
