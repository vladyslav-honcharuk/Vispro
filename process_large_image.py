import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import numpy as np
from src.utils import get_image_var
from src.utils.patch_utils import divide_image,stitch_patches_incremental
from src.main_pipeline import get_marker_mask,get_inpainting_result,remove_background,tissue_segregation,get_inpainting_result_batch


def run_high_res_Vispro(high_res_image_path):
    img_var, img_np = get_image_var(high_res_image_path)
    marker_mask = get_marker_mask(img_var)
    inpainted_image = get_inpainting_result(img_np, marker_mask)
    tissue_rgba, tissue_rgb, tissue_mask = remove_background(inpainted_image, model_name='u2net')
    segregation_image, segment_mask = tissue_segregation(tissue_rgb, tissue_mask, tissue_value=20)

    return marker_mask, tissue_mask, segment_mask



def run_single_tiff(tiff_image_path: str, high_res_image_path: str, patch_size: int = 1000) -> None:
    """
    Processes a TIFF image by applying a CNN-based mask, dividing into patches, performing inpainting,
    and stitching the result. The processed image is saved with and without a white background.

    Args:
        tiff_image_path (str): Path to the input TIFF image.
        high_res_image_path (str): Path to the high-resolution image used for CNN masking.
        patch_size (int): Size of each patch for processing. Default is 3000.
    """

    marker_mask, tissue_mask, segment_mask = run_high_res_Vispro(high_res_image_path)
    # Load TIFF image and mask
    tiff_image = plt.imread(tiff_image_path)
    # Resize the mask to match the TIFF image dimensions
    zoom_factors = (tiff_image.shape[0] / marker_mask.shape[0], tiff_image.shape[1] / marker_mask.shape[1])
    marker_mask_large = zoom(marker_mask, zoom_factors, order=0)
    tissue_mask_large = zoom(tissue_mask,zoom_factors,order=0)
    segment_mask_large = zoom(segment_mask,zoom_factors,order=0)

    # Ensure image is RGB by discarding alpha channel if present
    if tiff_image.shape[2] == 4:
        tiff_image = tiff_image[:, :, :3]

    img_patches, mask_patches, positions, original_shape = divide_image(tiff_image, marker_mask_large, patch_size)
    # Collect patches that need inpainting
    inpainting_img_patches = []
    inpainting_mask_patches = []
    inpainting_positions = []

    # Separate patches that require inpainting
    for img_patch, mask_patch, position in zip(img_patches, mask_patches, positions):
        if np.any(mask_patch == 1):
            inpainting_img_patches.append(img_patch)
            inpainting_mask_patches.append(mask_patch)
            inpainting_positions.append(position)

    # Convert lists to batched numpy arrays for batch processing
    if inpainting_img_patches:
        # Stack the patches into batch dimensions
        inpainting_img_batch = np.stack(inpainting_img_patches)  # Shape: (b, w, h, 3)
        inpainting_mask_batch = np.stack(inpainting_mask_patches)  # Shape: (b, w, h)
        # Perform batch inpainting
        inpainted_results = get_inpainting_result_batch(inpainting_img_batch, inpainting_mask_batch)

    # Prepare final results, inserting inpainted patches where needed
    results = []
    for img_patch, mask_patch, position in zip(img_patches, mask_patches, positions):
        if position in inpainting_positions:
            # Insert inpainted patch
            idx = inpainting_positions.index(position)
            result = inpainted_results[idx]
        else:
            # Use original patch if no inpainting was needed
            result = img_patch
        results.append(result)
    # Stitch patches back into a full image
    stitched_result = stitch_patches_incremental(results, positions, original_shape)

    f,a = plt.subplots(2,2)
    a[0,0].imshow(tiff_image)
    a[0,1].imshow(stitched_result)
    a[1,0].imshow(tissue_mask_large)
    a[1,1].imshow(segment_mask_large)
    plt.show()


imglist= "/media/huifang/data/fiducial/tiff_data/data_list.txt"
file = open(imglist)
lines = file.readlines()
num_files = len(lines)
for i in range(11,num_files):
    print(i)
    line = lines[i]
    line = line.rstrip().split(' ')
    tiff_image_path = line[0]

    high_res_image_path = line[2]
    run_single_tiff(tiff_image_path,high_res_image_path)

