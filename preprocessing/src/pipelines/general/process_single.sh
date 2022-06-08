#! /bin/sh
#
# test.sh
# Copyright (C) 2020 Lucas Costa Campos <rmk236@gmail.com>
#
# Distributed under terms of the MIT license.
#

fullfile=$1
image_shape="$2 $3 $4"
downsample_factor=$5
blur_sigma=$6

filename=$(basename -- "$1" | cut -d . -f 1)
extension=".nii.gz"

echo "Started to process " $filename
echo "Originally " $fullfile

# Create helper folders
tmp_folder=tmp/tmp_$filename
mkdir -p $tmp_folder
mkdir -p derived/HighRes derived/Mask derived/LowRes

# First we orient properly
echo "Reorienting data for " $fullfile
post_fsl=${tmp_folder}/${filename}_reoriented
echo $post_fsl
fslreorient2std $fullfile $post_fsl$extension || exit 1

# Then we make the size and slice standard
echo "Reslicing data for "$fullfile
post_standard=${post_fsl}_standard
python src/processing/reslice.py -i $post_fsl$extension -o $post_standard$extension || exit 1

# Now it is time to skullstrip and get the mask of the brain
echo "Skullstripping data for "$fullfile
post_hd_bet=${post_standard}_bet
post_hd_bet_mask=${post_standard}_bet_mask
hd-bet -i $post_standard$extension -o $post_hd_bet || exit 1

# Crop the image to the mininum size possible. Both the 
# main image and the mask
echo "Cropping data for "$fullfile
post_cropping=${post_hd_bet}_cropped
post_cropping_mask=${post_hd_bet_mask}_cropped
mrcrop ${post_hd_bet}$extension      -mask ${post_hd_bet_mask}$extension ${post_cropping}$extension
mrcrop ${post_hd_bet_mask}$extension -mask ${post_hd_bet_mask}$extension ${post_cropping_mask}$extension

# This is how one would do in more recent versions of MRTrix
#mrgrid ${post_hd_bet} crop -mask ${post_hd_bet_mask} ${post_cropping}
#mrgrid ${post_hd_bet_mask} crop -mask ${post_hd_bet_mask} ${post_cropping_mask}

# Pad so all images are in a standard size
echo "Padding data for "$fullfile
post_pad=${post_cropping}_pad
post_pad_mask=${post_cropping_mask}_pad
python src/processing/pad.py -i $post_cropping$extension -o $post_pad$extension -s $image_shape || exit 1
python src/processing/pad.py -i $post_cropping_mask$extension -o $post_pad_mask$extension -s $image_shape || exit 1

# Finally, time to get the low-res version of our data
echo "Making low resolution for "$fullfile
post_low_res=${post_pad}_low_res
python src/processing/blur_and_downsample.py -i $post_pad$extension -o $post_low_res$extension -f $downsample_factor -s $blur_sigma || exit 1


# Put the skullstripped and mask in their final folders
echo "Copying data to final folders for " $fullfile
mv $post_pad$extension derived/HighRes/$filename$extension
mv $post_pad_mask$extension derived/Mask/$filename$extension
mv $post_low_res$extension derived/LowRes/$filename$extension

rm -r ${tmp_folder}
