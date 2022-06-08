#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright Â© 2020 Lucas Costa Campos <rmk236@gmail.com>
#
# Distributed under terms of the MIT license.

import numpy as np

class Hyperparameters(object):

    """This small class just holds some often-used parameters. It is just a
    convenience layer on top of the configuration drawn from the YAML"""

    def __init__(self, config_hyper):
        self.upsampling_factor = config_hyper["upsampling_factor"]
        self.num_epochs = config_hyper["num_epochs"]
        self.residual_blocks = config_hyper["residual_blocks"]
        self.feature_size = config_hyper["feature_size"]
        self.batch_size = config_hyper["batch_size"]

        self.learning_rate_gen = config_hyper["learning_rate_gen"]
        self.learning_rate_disc = config_hyper["learning_rate_disc"]
        self.learning_rate_decay = config_hyper["learning_rate_decay"]
        self.learning_rate_steps = config_hyper["learning_rate_steps"]


class ImageSizes(object):

    def __init__(self, config_img):

        self.x_overlap_length=config_img["x_overlap_length"]
        self.y_overlap_length=config_img["y_overlap_length"]
        self.z_overlap_length=config_img["z_overlap_length"]

        self.width = config_img["width"]
        self.height = config_img["height"]
        self.depth = config_img["depth"]

        self.num_x_patches=config_img["num_x_patches"]
        self.num_y_patches=config_img["num_y_patches"]
        self.num_z_patches=config_img["num_z_patches"]

        self.padded_patch_with_overlap_x_length, self.padded_patch_with_overlap_y_length, self.padded_patch_with_overlap_z_length = self.determine_padded_patch_with_overlap_sizes(config_img)

    def determine_padded_patch_with_overlap_sizes(self, config_img):
        s=(self.width, self.height, self.depth)

        #Calculate: If we want a specific number of patches on each axis, how large can each patch on that axis be?
        # patch_x_length = int(s[0]/self.num_x_patches)+self.x_overlap_length
        # patch_y_length = int(s[1]/self.num_y_patches)+self.y_overlap_length
        # patch_z_length = int(s[2]/self.num_z_patches)+self.z_overlap_length

        #explanation:
        #1) np.ceil(s[0]/num_x_patches) gives you the size of a single patch after it is padded and before the overlap is added
        #2) then we add the size of the overlap to get the final container-size
        padded_patch_with_overlap_x_length = np.ceil(s[0]/self.num_x_patches) + self.x_overlap_length
        padded_patch_with_overlap_y_length = np.ceil(s[1]/self.num_y_patches) + self.y_overlap_length
        padded_patch_with_overlap_z_length = np.ceil(s[2]/self.num_z_patches) + self.z_overlap_length

        return int(padded_patch_with_overlap_x_length), int(padded_patch_with_overlap_y_length), int(padded_patch_with_overlap_z_length)
