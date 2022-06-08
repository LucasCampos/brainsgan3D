import nibabel as nib
import numpy as np
import logging


class Dataset(object):
    def __init__(self, config, filename):


        # Tempory variables to make stuff below cleaner
        num_x_patches = config["image_sizes"]["num_x_patches"]
        num_y_patches = config["image_sizes"]["num_y_patches"]
        num_z_patches = config["image_sizes"]["num_z_patches"]

        x_overlap_length = config["image_sizes"]["x_overlap_length"]
        y_overlap_length = config["image_sizes"]["y_overlap_length"]
        z_overlap_length = config["image_sizes"]["z_overlap_length"]

        width = config["image_sizes"]["width"]
        height = config["image_sizes"]["height"]
        depth = config["image_sizes"]["depth"]


        # Now we actually save things
        self.upsampling_factor = int( config["hyperparameters"]["upsampling_factor"])
        self.num_patches = np.array([num_x_patches, num_y_patches, num_z_patches], dtype=np.int)
        self.overlap_length = np.array([x_overlap_length, y_overlap_length, z_overlap_length], dtype=np.int)
        self.size_HR = np.array([width, height, depth], dtype=np.int)
        self.size_LR = self.size_HR//self.upsampling_factor

        patch_length_HR, _, _ = self.find_patch_size(self.size_HR, self.upsampling_factor)
        patch_length_LR, _, _ = self.find_patch_size(self.size_LR, 1)

        self.final_patch_length_HR  = patch_length_HR + self.overlap_length
        self.final_patch_length_LR  = patch_length_LR + self.overlap_length//self.upsampling_factor

        with open(filename, "r") as file_reader:
            lines = file_reader.readlines()
            self.datalist = [ l.split() for l in lines]

    def find_patch_size(self, size, upsampling_factor):

        # Calculate: If we want a specific number of patches on each axis, how large can each patch on that axis be?
        og_patch_length = np.array(np.ceil(size/self.num_patches), dtype=np.int)

        # Make the path a multiple of the upsampling factor
        new_patch_length = np.ceil(og_patch_length/upsampling_factor) * upsampling_factor
        new_patch_length = np.array(new_patch_length, dtype=np.int)

        new_size = new_patch_length * self.num_patches
        diff = new_size - size

        pad_left = np.zeros_like(diff)
        pad_right = np.zeros_like(diff)
        np.floor(diff/2, pad_left, casting='unsafe')
        np.ceil(diff/2, pad_right, casting='unsafe')

        return new_patch_length, pad_left, pad_right



    #base function for loading Nifti scans. both masks and brains (depending on filename)
    def load_full_scan(self, filename):
        logging.debug("loading scan:" + filename)
        scan = np.asarray(nib.load(filename).get_fdata(), dtype=np.float32)
        return scan


    def chop_scan_into_patches(self, original_scan, size, overlap_length, upsampling_factor):

        total_num_of_patches = np.prod(self.num_patches)

        patch_length, pad_left, pad_right = self.find_patch_size(size, upsampling_factor)
        # Pad the image with enough voxels to make the path the same size plus however much we need for overlap
        padding_shapes = np.array([pad_left, pad_right + overlap_length], dtype=np.int).transpose()
        padded_scan = np.pad(original_scan, padding_shapes, 'constant', constant_values=0)

        lx, ly, lz = patch_length
        lxp, lyp, lzp = patch_length + overlap_length

        res = np.empty((total_num_of_patches, lxp, lyp, lzp, 1), dtype=np.float32)


        count = 0
        for x in range(self.num_patches[0]):
            for y in range(self.num_patches[1]):
                for z in range(self.num_patches[2]):

                    res[count, :, :, :, 0] = padded_scan[
                        x*lx:(x+1)*lx + overlap_length[0],
                        y*ly:(y+1)*ly + overlap_length[1],
                        z*lz:(z+1)*lz + overlap_length[2]
                    ]

                    count += 1

        return res

    # depending on whether you pass the mask- or brain-datalist, this function
    # will return a batch of full scans, each of which is cut into your
    # specified number of patches with your specified overlap between them.
    # The overlap is important because the neural network also needs to learn
    # kernels to upsample the border-regions between patches.  During
    # evaluation we load the full scan, without cutting it into smaller
    # patches. We can only do this during evaluation and not during training,
    # because during training we need the memory for optimizing the network's
    # weights.
    def batch_load_brain(self, itercount_offset):

        files = self.datalist[itercount_offset]
        filename = files[0].split("/")[-1]

        batch_of_HR = self.load_full_scan(files[0])
        batch_of_Mask = self.load_full_scan(files[1])
        batch_of_LR = self.load_full_scan(files[2])

        batch_of_Mask = np.array(batch_of_Mask, dtype=np.bool)

        return batch_of_HR, batch_of_Mask, batch_of_LR, filename

    def batch_load_patches(self, itercount_offset):

        batch_of_HR, batch_of_Mask, batch_of_LR, filename = self.batch_load_brain(itercount_offset)

        patches_LR = self.chop_scan_into_patches(batch_of_LR, self.size_LR, self.overlap_length//self.upsampling_factor, 1)
        patches_HR = self.chop_scan_into_patches(batch_of_HR, self.size_HR, self.overlap_length, self.upsampling_factor)
        patches_Mask = self.chop_scan_into_patches(batch_of_Mask,  self.size_HR, self.overlap_length, self.upsampling_factor)

        return patches_HR, patches_Mask, patches_LR, filename
