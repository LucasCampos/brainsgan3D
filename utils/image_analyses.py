from skimage.metrics import structural_similarity as ssim
import numpy as np

def calculate_ssim_masked(volume_generated, volume_real, mask):

    val_max = np.amax(np.maximum(volume_generated, volume_real))
    val_min = np.amin(np.minimum(volume_generated, volume_real))

    _, total_ssim = ssim(volume_real, volume_generated, data_range=val_max - val_min, full=True)
    masked_ssim = np.average(total_ssim[mask])
    return masked_ssim

def calculate_psnr_masked(volume_generated, volume_real, mask):

    val_max = np.amax(np.maximum(volume_generated, volume_real))
    val_min = np.amin(np.minimum(volume_generated, volume_real))

    r = val_max - val_min

    mses = np.square(volume_real[mask] - volume_generated[mask])
    mse = np.average(mses)

    masked_psnr = 10*np.log10(r**2/mse)

    return masked_psnr

