# BrainsGAN 3D

This is the code for paper "Anatomical fidelity of cross-population Super
Resolution Generative Adversarial Network based MRI upsampling". It uses a
three dimentional variation of SRGANs to do supersampling of MRI data.

This source code has originally been based on [the 3D-GAN-superresolution
repo](https://github.com/imatge-upc/3D-GAN-superresolution), but most of the
code has been rewritten/modified at one point or another.

## Installation

In order to dowload the packages necessary, the environment file included can be used as

```bash
conda env create -f environment.yml
```

It is highly recommended that this is code is ran on GPUs. For the default
parameters, at least 16GB of VRAM are necessary, however it is possible to use
GPUs with less by increasing the number of patches/reducing the batch size. The
code has been tested has been tested on Nvidia P100 and A100. In order to use
it, an installation of CUDA 11 is necessary (including cuDNN 8). It is possible
to create a local install using conda as

```bash
conda install cudatoolkit==11.0.221 cuDNN
```

## Usage

The single entrypoint of the code is the `main.py` file. In order to input
parameters, an `YAML` file is required. The entries of this file are explained
in the next section, and an example is available.

In order to train the network, use the following line 

```bash
python main.py train <path/to/config>
```

## Parameter file

```
hyperparameters:
  batch_size: 2
  feature_size: 32
  learning_rate_decay: 0.99
  learning_rate_disc: 1.0e-6
  learning_rate_gen: 1.0e-4
  learning_rate_steps: 4920
  max_checkpoints: 265
  num_epochs: 100
  residual_blocks: 6
  upsampling_factor: 2
image_sizes:
  # Shape of the original (and upsampled) image
  depth: 168
  height: 196
  width: 160
  # Number of patches to break the image into
  # increasing this number should decrease
  # memory pressure
  num_x_patches: 2
  num_y_patches: 2
  num_z_patches: 2
  # How much each patch should overlap with one another
  x_overlap_length: 32
  y_overlap_length: 32
  z_overlap_length: 32
inputs:
  # File with paths to data. See section below
  data_eval: 1000Brains_eval.txt
  data_train: 1000Brains_train.txt
  restore_checkpoint_evaluation: checkpoints/
  restore_checkpoint_training: null
output_dirs:
  checkpoint_dir: checkpoints/
  eval_dir: eval_predictions/
  log_dir: runtime_logs/
  loss_dir: loss-info/
outputs:
  log_level: debug
  max_checkpoints: 100
  save_every_n_epochs: 10
```

## Path files

The entries in `data_eval` and `data_train` expect a file format asa
 
```
sub1_HR.nii.gz sub1_Mask.nii.gz sub1_LowRes.nii.gz
sub2_HR.nii.gz sub2_Mask.nii.gz sub2_LowRes.nii.gz
sub3_HR.nii.gz sub3_Mask.nii.gz sub3_LowRes.nii.gz
sub3_HR.nii.gz sub4_Mask.nii.gz sub3_LowRes.nii.gz
...
```

In principle, any format accepted by `nibabel` should work, but only NIFTY was tested.

## Authors
* **Leona Förster** - *2D Prototype*
* **Lucas da Costa Campos** 
