#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2020 Lucas Costa Campos <rmk236@gmail.com>
#
#
# Distributed under terms of the MIT license.

import argparse
import datetime
import logging
import math
import nibabel as nib
import numpy as np
import os
import sys
import yaml
import time
import glob

import tensorflow as tf
import tensorflow.keras.backend as K

from pathlib import Path
from utils.universal_dataset import Dataset

from utils.my_io import *
from utils.generator import generator
from utils.discriminator import discriminator
from utils.helper_parameters import Hyperparameters, ImageSizes
from utils.image_analyses import calculate_ssim_masked, calculate_psnr_masked


#######################################################################
#                          TRAINING FUNCTION                          #
#######################################################################

def discriminator_loss(output_real, output_generated):

    if tf.random.uniform([1]) > 0.1:
        # give correct classifications
        valid = tf.random.uniform(shape=output_real.get_shape(), dtype=tf.float32, minval=0.7, maxval=1.2)
        fake = tf.random.uniform(shape=output_generated.get_shape(), dtype=tf.float32, minval=0.0, maxval=0.3)
    else:
        # give wrong classifications (noisy labels) in 10% of all cases
        valid = tf.random.uniform(shape=output_real.get_shape(), dtype=tf.float32, minval=0.0, maxval=0.3)
        fake = tf.random.uniform(shape=output_generated.get_shape(), dtype=tf.float32, minval=0.7, maxval=1.2)

    return 0.5*( tf.square(output_real - valid) + tf.square(output_generated - fake))


def generator_loss(generated_image, original_image, discriminator_output):

    """
    There are three components in this loss.
    1. The adversarial part, i.e., how well we fooled the discriminator
    2. How well the generated image reproduces the original image
    3. How sharp is the final image
    """

    # 1.
    valid = tf.ones_like(discriminator_output, dtype=tf.float32)
    adversarial_loss = 0.5*tf.square(discriminator_output - valid)

    # 2.
    mse_loss = tf.math.reduce_mean(tf.math.squared_difference(generated_image, original_image))

    # 3.
    grad_x_gen = tf.math.abs(generated_image[:, 1:, :, :, :] - generated_image[:, :-1, :, :, :])
    grad_y_gen = tf.math.abs(generated_image[:, :, 1:, :, :] - generated_image[:, :, :-1, :, :])
    grad_z_gen = tf.math.abs(generated_image[:, :, :, 1:, :] - generated_image[:, :, :, :-1, :])

    grad_x_ori = tf.math.abs(original_image[:, 1:, :, :, :] - original_image[:, :-1, :, :, :])
    grad_y_ori = tf.math.abs(original_image[:, :, 1:, :, :] - original_image[:, :, :-1, :, :])
    grad_z_ori = tf.math.abs(original_image[:, :, :, 1:, :] - original_image[:, :, :, :-1, :])

    gradient_loss = tf.math.reduce_mean(tf.math.squared_difference(grad_x_ori, grad_x_gen)) + \
                    tf.math.reduce_mean(tf.math.squared_difference(grad_y_ori, grad_y_gen)) + \
                    tf.math.reduce_mean(tf.math.squared_difference(grad_z_ori, grad_z_gen))

    total_loss = 1e-3*adversarial_loss + mse_loss + gradient_loss

    return total_loss, adversarial_loss, mse_loss, gradient_loss


@tf.function
def train_step(images_HR, images_LR, gen, disc, generator_optimizer, discriminator_optimizer):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gen(images_LR, training=True)

        output_real = disc(images_HR, training=True)
        output_fake = disc(generated_images, training=True)

        gen_loss, adversarial_loss, mse_loss, gradient_loss = generator_loss(generated_images, images_HR, output_fake)
        disc_loss = discriminator_loss(output_real, output_fake)

    gradients_of_generator = gen_tape.gradient(gen_loss, gen.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, disc.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, disc.trainable_variables))

    return tf.math.reduce_mean(disc_loss), tf.math.reduce_mean(gen_loss), tf.reduce_mean(adversarial_loss), tf.math.reduce_mean(mse_loss), tf.math.reduce_mean(gradient_loss)


@tf.function
def evaluate_step(images_HR, images_LR, gen):

    generated_images = gen(images_LR, training=False)

    # Create fake fake date. This is such the adversarial part of the
    # generator loss is zero
    output_fake = tf.zeros(len(images_LR), dtype=tf.float32)

    gen_loss, adversarial_loss, mse_loss, gradient_loss = generator_loss(generated_images, images_HR, output_fake)

    return generated_images, tf.math.reduce_mean(gen_loss), tf.reduce_mean(adversarial_loss), tf.math.reduce_mean(mse_loss), tf.math.reduce_mean(gradient_loss)


def train(config):

    hyper = Hyperparameters(config["hyperparameters"])
    img = ImageSizes(config["image_sizes"])
    # Instanciate helper class to deal with loading data from harddisk
    train_dataset = Dataset(config, config["inputs"]["data_train"])

    iterations_train = len(train_dataset.datalist)

    number_of_patches = img.num_x_patches * img.num_y_patches*img.num_z_patches
    subiterations_per_brain = number_of_patches//hyper.batch_size

    img_shape_HR = train_dataset.final_patch_length_HR
    img_shape_LR = train_dataset.final_patch_length_LR

    if not number_of_patches % hyper.batch_size == 0:
        print("The number of patches is not an integer multiple of the batch size! Killing program")
        exit()

    logging.debug("Started training")
    print("Started training")

    # ##========================== DEFINE MODEL ============================##

    disc = discriminator(img_shape_HR, kernel_size=3, filters=hyper.feature_size)
    gen = generator(img_shape_LR, upsampling_factor=hyper.upsampling_factor,
                     kernel_size=3,
                     filters=hyper.feature_size,
                     number_of_blocks=hyper.residual_blocks)

    # Set training parameters
    decay_rate = hyper.learning_rate_decay
    decay_steps = hyper.learning_rate_steps

    learning_rate_fn_gen = tf.keras.optimizers.schedules.InverseTimeDecay(hyper.learning_rate_gen, decay_steps, decay_rate)
    learning_rate_fn_disc = tf.keras.optimizers.schedules.InverseTimeDecay(hyper.learning_rate_disc, decay_steps, decay_rate)
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate_fn_gen)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate_fn_disc)


    # Prepare checkpoint and manager
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=gen,
                                     discriminator=disc,
                                     step=tf.Variable(1))

    #reset the loss document.
    filepath = config["output_dirs"]["loss_dir"] + "/loss.txt"


    if config["inputs"]["restore_checkpoint_training"] is not None:
        latest_ckp = tf.train.latest_checkpoint(config["inputs"]["restore_checkpoint_training"])
        checkpoint.restore(latest_ckp)
        first_epoch = int(checkpoint.step)
        logging.debug("Restored from {}".format(latest_ckp))
        print("Restored from {}".format(latest_ckp))
        loss_file = open(filepath, "a+")  # append mode
    else:
        logging.debug("Initializing from scratch.")
        print("Initializing from scratch.")
        first_epoch = 0
        loss_file = open(filepath, "w")  # write mode

        # write headlines into the loss document.
        loss_file.write("#Disc Gen (Adversarial + MSE + Gradient) Disc+Gen\n")

    print("First epoch: ", first_epoch)
    logging.debug("First epoch: {}".format(first_epoch))

    checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                                    config["output_dirs"]["checkpoint_dir"],
                                                    max_to_keep=config["outputs"]["max_checkpoints"])

    epoch_format = "Epoch [{:2}/{:2}] Brain [{:4}/{:4}] Minibatch [{:4}/{:4}] : d_loss: {:.8e} g_loss: {:.8e} (adv: {:.8e} + mse: {:.8e} + grad: {:.8e}), total_loss: {:.8e}"

    val_restore = 0
    count = 0
    iteration = 0
    for epoch in range(first_epoch, hyper.num_epochs):
        try:

            loss_file.flush()
            for brain in range(0, iterations_train):

                # ====================== LOAD DATA =========================== #
                # High resrain
                xt_total, xm_total, xt_total_LR, filename = train_dataset.batch_load_patches(brain)

                for batch in range(0, subiterations_per_brain):

                    xt = xt_total[batch*hyper.batch_size:(batch+1)*hyper.batch_size]
                    xm = xm_total[batch*hyper.batch_size:(batch+1)*hyper.batch_size]
                    xt_LR = xt_total_LR[batch*hyper.batch_size:(batch+1)*hyper.batch_size]

                    logging.debug("xt shape: "+ str(xt.shape))
                    logging.debug("xm shape: "+ str(xt.shape))
                    logging.debug("xt_LR shape: "+ str(xt_LR.shape))

                    logging.debug("max color in HR before normalizing: " + str(np.amax(xt)))
                    # NORMALIZING
                    for t in range(0, xt.shape[0]):
                        mx = np.amax(xt[t])
                        mn = np.amin(xt[t])
                        xt[t] = (xt[t] - mn) / (mx - mn)
                    logging.debug("max color in HR after normalizing " + str(np.amax(xt)))

                    logging.debug("max color in LR before normalizing: " + str(np.amax(xt)))
                    # Now process the LR data
                    # NORMALIZING
                    for t in range(0, xt_LR.shape[0]):
                        mx = np.amax(xt_LR[t])
                        mn = np.amin(xt_LR[t])
                        xt_LR[t] = (xt_LR[t] - mn) / (mx - mn)
                    logging.debug("max color in LR after normalizing " + str(np.amax(xt)))

                    xt = tf.convert_to_tensor(xt)
                    xt_LR = tf.convert_to_tensor(xt_LR)

                    disc_loss, gen_loss, adversarial_loss, mse_loss, gradient_loss = train_step(xt, xt_LR, gen, disc, generator_optimizer, discriminator_optimizer)

                    str_epoch = epoch_format.format(epoch, hyper.num_epochs + val_restore, brain, iterations_train, batch, subiterations_per_brain,
                                                    disc_loss, gen_loss, adversarial_loss, mse_loss, gradient_loss, disc_loss + gen_loss)
                    print(str_epoch)
                    logging.debug(str_epoch)
                    if count % config["outputs"]["store_loss_every_n_iters"] == 0:
                        logging.debug("Writing loss")
                        loss_file.write(f"{count} {disc_loss} {gen_loss} {adversarial_loss} {mse_loss} {gradient_loss} {disc_loss + gen_loss}\n")
                    count += 1

                if iteration % config["outputs"]["save_every_n_iters"] == 0:
                    print("Saving checkpoint")
                    checkpoint_manager.save()
                else:
                    print("NOT saving checkpoint")

                checkpoint.step.assign_add(1)
                iteration += 1

        except KeyboardInterrupt:
            raise

#######################################################################
#                              EVALUATE                               #
#######################################################################
def evaluate(config, checkpoint_path):

    hyper = Hyperparameters(config["hyperparameters"])
    img = ImageSizes(config["image_sizes"])

    eval_dataset = Dataset(config, config["inputs"]["data_eval"])
    iterations_eval = len(eval_dataset.datalist)

    logging.debug(str(iterations_eval) + " iterations for evaluation will be performed")

    number_of_patches = 1

    img_sizes = [img.width//hyper.upsampling_factor,
                 img.height//hyper.upsampling_factor,
                 img.depth//hyper.upsampling_factor]

    gen = generator(img_sizes, upsampling_factor=hyper.upsampling_factor,
                     kernel_size=3,
                     filters=hyper.feature_size,
                     number_of_blocks=hyper.residual_blocks)
    # Prepare checkpoint and manager
    checkpoint = tf.train.Checkpoint(generator=gen, step=tf.Variable(1))
    checkpoint.restore(checkpoint_path).expect_partial()
    restored_epoch = int(checkpoint.step)
    logging.info(f"Using data from epoch {restored_epoch}.")
    print(f"Using data from epoch {restored_epoch}.")
    eval_dir = config["output_dirs"]["eval_dir"] + f"/epoch_{restored_epoch:04}/"
    Path(eval_dir).mkdir(parents=True, exist_ok=True)

    array_psnr = np.empty(iterations_eval)
    array_ssim = np.empty(iterations_eval)
    array_gen_loss = np.empty(iterations_eval)

    for count in range(iterations_eval):

        print(count)

        # extract volumes
        xt_total_HR, xt_mask, xt_total_LR, filename = eval_dataset.batch_load_brain(count)

        #Normalized
        mx = np.amax(xt_total_HR)
        mn = np.amin(xt_total_HR)
        xt_total_HR = (xt_total_HR - mn) / (mx - mn)

        mx = np.amax(xt_total_LR)
        mn = np.amin(xt_total_LR)
        xt_total_LR = (xt_total_LR - mn) / (mx - mn)


        #Change shape to fit TF expectation
        xt_total_LR = xt_total_LR[np.newaxis, :, :, :, np.newaxis]
        xt_total_HR = xt_total_HR[np.newaxis, :, :, :, np.newaxis]

        xt_total_LR_tensor = tf.convert_to_tensor(xt_total_LR)
        xt_total_HR_tensor = tf.convert_to_tensor(xt_total_HR)

        logging.debug("xt total_LR shape: "+ str(xt_total_LR.shape))
        logging.debug("xt total_HR shape: "+ str(xt_total_HR.shape))
        logging.debug("xt mask shape: "+ str(xt_mask.shape))
        logging.debug("upsampling factor: "+ str(hyper.upsampling_factor))

        generated_image, gen_loss, _, _, _ = evaluate_step(xt_total_HR_tensor, xt_total_LR_tensor, gen)

        array_gen_loss[count] = gen_loss

        volume_mask = xt_mask
        volume_real = xt_total_HR[0, :, :, :, 0]
        volume_real_LR = xt_total_LR[0, :, :, :, 0]
        volume_generated = generated_image.numpy()[0, :, :, :, 0]

        val_psnr = calculate_psnr_masked(volume_real, volume_generated, volume_mask)
        array_psnr[count] = val_psnr

        val_ssim = calculate_ssim_masked(volume_real, volume_generated, volume_mask)
        array_ssim[count] = val_ssim


        mx = np.amax(volume_generated)
        mn = np.amin(volume_generated)

        # There is a small bug in FastSurfer that crashes if our normalization
        # is between [0,1]. So we do it between [0, 255], as it is the range
        # of the CHAR they use internally
        volume_generated_norm = (volume_generated - mn) / (mx - mn) * 255.0
        volume_generated_norm[volume_mask == 0] = 0
        volume_generated[volume_mask == 0] = 0

        # compute metrics
        logging.debug("mask: " + str(volume_mask.shape))
        logging.debug("real: " + str(volume_real.shape))
        logging.debug("mask: " + str(volume_mask.shape))
        logging.debug("generated: " + str(volume_generated.shape))

        # save volumes
        if False:
            filename_gen = os.path.join(eval_dir, 'gen_' + filename)
            img_volume_gen = nib.Nifti1Image(volume_generated, np.eye(4))
            img_volume_gen.to_filename(filename_gen)

            filename_gen_norm = os.path.join(eval_dir,'gen_norm_' +  filename)
            img_volume_gen_norm = nib.Nifti1Image(volume_generated_norm, np.eye(4))
            img_volume_gen_norm.to_filename(filename_gen_norm)

    print("Final evaluation results")
    print('{}{}'.format('Mean PSNR: ', array_psnr.mean()))
    print('{}{}'.format('Mean SSIM: ', array_ssim.mean()))
    print('{}{}'.format('Variance PSNR: ', array_psnr.var()))
    print('{}{}'.format('Variance SSIM: ', array_ssim.var()))
    print('{}{}'.format('Max PSNR: ', array_psnr.max()))
    print('{}{}'.format('Min PSNR: ', array_psnr.min()))
    print('{}{}'.format('Max SSIM: ', array_ssim.max()))
    print('{}{}'.format('Min SSIM: ', array_ssim.min()))
    print('{}{}'.format('Median PSNR: ', np.median(array_psnr)))
    print('{}{}'.format('Median SSIM: ', np.median(array_ssim)))
    print()

    print('{}{}'.format('Mean Gen Loss: ', array_gen_loss.mean()))
    print('{}{}'.format('Variance Gen Loss: ', array_gen_loss.var()))
    print('{}{}'.format('Max Gen Loss: ', array_gen_loss.max()))
    print('{}{}'.format('Min Gen Loss: ', array_gen_loss.min()))
    print('{}{}'.format('Median Gen Loss: ', np.median(array_gen_loss)))

    np.savetxt(config["output_dirs"]["eval_dir"] + f"/evaluation_results_{restored_epoch}.txt", np.array([array_psnr, array_ssim, array_gen_loss]).transpose(), header="# PSNR SSIM Generator")


def perform_init(type, config):

    # Create important directories
    for p in config["output_dirs"].values():
        Path(p).mkdir(parents=True, exist_ok=True)

    # Initialize the logging
    now_str = datetime.datetime.now().isoformat()
    log_path = f"{config['output_dirs']['log_dir']}/{type}_{now_str}.log"
    if config["outputs"]["log_level"] == "critical":
        logging.basicConfig(level=logging.CRITICAL, handlers=[logging.FileHandler(log_path)])
    else:
        logging.basicConfig(level=logging.DEBUG, handlers=[logging.FileHandler(log_path)])


def main():

    parser = argparse.ArgumentParser(description='Parse the input arguments.')

    parser.add_argument('mode', type=str, help='train or eval')
    parser.add_argument('config_file', type=str, default="configuration.yaml", help='choose configuration file')
    args=parser.parse_args()

    with open(sys.argv[2], 'r') as f:
        config = yaml.safe_load(f)
    print("config:", config)
    perform_init(sys.argv[1], config)

    if sys.argv[1] == "train":
        train(config)
    elif sys.argv[1] == "eval_all":
        base_path = config["inputs"]["restore_checkpoint_evaluation"]
        paths = sorted(glob.glob(base_path + "/*.index"))
        # Now lets remove that ".index"
        paths = [ p[:-6] for p in paths ]
        for p in paths:
            evaluate(config, p)
    elif sys.argv[1] == "eval":
        base_path = config["inputs"]["restore_checkpoint_evaluation"]
        paths = sorted(glob.glob(base_path + "/*.index"))
        # Now lets remove that ".index"
        paths = [ p[:-6] for p in paths ]
        evaluate(config, paths[-1])


if __name__ == "__main__":
    main()
