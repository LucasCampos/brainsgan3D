#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Lucas Costa Campos <rmk236@gmail.com>
#
# Distributed under terms of the MIT license.

import tensorflow as tf
from tensorflow.keras import layers
import logging

import numpy as np

def discriminator(img_shape, kernel_size=3, filters=32):

    model = tf.keras.Sequential(name="Discriminator")

    # First Block
    model.add(layers.Input(shape=[*img_shape, 1]))
    model.add(layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=1, padding="same"))
    model.add(layers.LeakyReLU())

    for _ in range(3):
        # sublock1
        model.add(layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=2, padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        filters = filters*2

        # sublock2
        model.add(layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=1, padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

    # Fifth block 
    model.add(layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=2, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Final block 
    model.add(layers.Flatten())
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(1, activation="sigmoid"))

    #print(model.summary(line_length=150))
    model.summary(line_length=150, print_fn=logging.debug)
    return model
