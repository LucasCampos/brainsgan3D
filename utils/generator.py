#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Lucas Costa Campos <rmk236@gmail.com>
#
# Distributed under terms of the MIT license.

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import logging

def generator(img_shape=[None, None, None], upsampling_factor=2, kernel_size=3, filters=32, number_of_blocks=6):

    def residual_block(layer_input, filters):
        """Residual block described in paper"""
        d = layers.Conv3D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
        d = layers.BatchNormalization()(d)
        d = layers.LeakyReLU(0.25)(d)
        d = layers.Conv3D(filters, kernel_size=3, strides=1, padding='same')(d)
        d = layers.BatchNormalization()(d)
        d = layers.Add()([d, layer_input])
        return d

    number_of_SS_blocks = int(np.log2(upsampling_factor))

    img_lr = layers.Input(shape=[*img_shape, 1])
    c1 = layers.Conv3D(filters, kernel_size=kernel_size, strides=1, padding="same")(img_lr)
    c2 = layers.BatchNormalization()(c1)
    c3 = layers.LeakyReLU(0.25)(c2)

    # Residual part of the network
    r = residual_block(c3, filters)
    for _ in range(1, number_of_blocks):
        r = residual_block(r, filters)

    # Large skip connection
    s1 = layers.Conv3D(filters, kernel_size=kernel_size, strides=1, padding="same")(r)
    s2 = layers.BatchNormalization()(s1)
    s3 = layers.Add()([s2, c3])
    inp = s3

    for _ in range(number_of_SS_blocks):
        up1 = layers.Conv3D(filters*2, kernel_size=kernel_size, strides=1, padding="same")(inp)
        up2 = layers.LeakyReLU(0.25)(up1)
        up3 = layers.UpSampling3D()(up2)

        up4 = layers.Conv3D(filters*2, kernel_size=kernel_size, strides=1, padding="same")(up3)
        up5 = layers.LeakyReLU(0.25)(up4)
        inp = up5

    end = layers.Conv3D(1, kernel_size=kernel_size, strides=1, padding="same")(inp)

    model = tf.keras.Model(img_lr, end, name="Generator")
    model.summary(line_length=150, print_fn=logging.debug)
    #print(model.summary(line_length=150))
    return model
