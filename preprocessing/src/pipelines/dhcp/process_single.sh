#! /bin/sh
#
# test.sh
# Copyright (C) 2020 Lucas Costa Campos <rmk236@gmail.com>
#
# Distributed under terms of the MIT license.
#

fullfile_brain=$1
fullfile_mask=$2
image_shape="$3 $4 $5"
downsample_factor=$6
blur_sigma=$7
extension=".nii.gz"

filename=$(basename -- "$1" | cut -d . -f 1)
filename_mask=${filename}_mask
extension=".nii.gz"

echo "Started to process " $filename
echo "Originally " $fullfile_brain

# Create helper folders
tmp_folder=tmp/tmp_$filename
mkdir -p $tmp_folder
mkdir -p derived/HighRes derived/Mask derived/LowRes

# First we orient properly
post_fsl=${tmp_folder}/${filename}_reoriented
post_fsl_mask=${tmp_folder}/${filename_mask}_reoriented

echo "Reorienting data for brain" 
fslreorient2std $fullfile_brain $post_fsl$extension || exit 1
echo "Reorienting data for mask" 
fslreorient2std $fullfile_mask $post_fsl_mask$extension || exit 1

# Then we make the size and slice standard
post_reslice=${post_fsl}_reslice
post_reslice_mask=${post_fsl_mask}_reslice
echo "Reslicing data for brain"
python src/processing/reslice.py -i $post_fsl$extension -o $post_reslice$extension || exit 1
echo "Reslicing data for mask"
python src/processing/reslice.py -i $post_fsl_mask$extension -o $post_reslice_mask$extension || exit 1

# Crop the image to the mininum size possible.
post_cropping=${post_reslice}_cropped
post_cropping_mask=${post_reslice_mask}_cropped
echo "Cropping data for brain"
mrcrop ${post_reslice}$extension      -mask ${post_reslice_mask}$extension ${post_cropping}$extension
echo "Cropping data for mask"
mrcrop ${post_reslice_mask}$extension -mask ${post_reslice_mask}$extension ${post_cropping_mask}$extension

# Then we make the size and slice standard
post_pad=${post_cropping}_pad
post_pad_mask=${post_cropping_mask}_pad
echo "Reslicing data for brain"
python src/processing/pad.py -i $post_cropping$extension -o $post_pad$extension -s $image_shape || exit 1
echo "Reslicing data for mask"
python src/processing/pad.py -i $post_cropping_mask$extension -o $post_pad_mask$extension -s $image_shape || exit 1

post_brain=$post_pad
post_mask=$post_pad_mask

##########################################################################################################

# Now it is time to skullstrip and get the mask of the brain
echo "Skullstripping data for "$post_brain$extension
post_mask_mult=${post_brain}_no_skull
python src/processing/brain_extract.py -i $post_brain$extension ${post_mask}$extension -o $post_mask_mult$extension || exit 1

# Normalize the high_res version
post_norm=${post_mask_mult}_norm
python src/processing/normalize.py -i $post_mask_mult$extension -o $post_norm$extension || exit 1


# Finally, time to get the low-res version of our data
echo "Making low resolution for "$post_mask_mult$extension
post_low_res=${post_norm}_low_res
python src/processing/blur_and_downsample.py -i $post_norm$extension -o $post_low_res$extension -f $downsample_factor -s $blur_sigma || exit 1

# Normalize the low res version
post_low_norm=${post_low_res}_norm
python src/processing/normalize.py -i $post_low_res$extension -o $post_low_norm$extension || exit 1


# Put the skullstripped and mask in their final folders
echo "Copying data to final folders for " $fullfile_brain
mv $post_norm$extension derived/HighRes/$filename$extension
mv ${post_mask}$extension derived/Mask/$filename$extension
mv $post_low_norm$extension derived/LowRes/$filename$extension

# rm -r ${tmp_folder}
