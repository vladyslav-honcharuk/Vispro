import numpy as np
from typing import List,Tuple

__all__ = ["divide_image", "stitch_patches_incremental"]

def divide_image(image: np.ndarray, mask: np.ndarray, patch_size: int) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int]], Tuple[int, int, int]]:
    """
    Divides the input image and mask into smaller patches, with padding if necessary.

    Args:
        image (np.ndarray): The input image array in (H,W,C) format.
        mask (np.ndarray): The mask array in (H, W) format.
        patch_size (int): The size of each square patch.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int]], Tuple[int, int, int]]:
            - img_patches: List of image patches, each in (C, patch_size, patch_size) format.
            - mask_patches: List of mask patches, each in (patch_size, patch_size) format.
            - positions: List of (i, j) tuples indicating the top-left corner of each patch in the original padded image.
            - original_shape: Tuple representing the original shape of the input image (H,W,C).
    """
    h,w,c = image.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    # Pad the image and mask
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w),(0, 0)), mode='constant', constant_values=0)
    padded_mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    img_patches = []
    mask_patches = []
    positions = []

    padded_h, padded_w = padded_image.shape[0], padded_image.shape[1]

    # Extract patches and their positions
    for i in range(0, padded_h, patch_size):
        for j in range(0, padded_w, patch_size):
            img_patch = padded_image[i:i + patch_size, j:j + patch_size,:]
            mask_patch = padded_mask[i:i + patch_size, j:j + patch_size]
            img_patches.append(img_patch)
            mask_patches.append(mask_patch)
            positions.append((i, j))
    return img_patches, mask_patches, positions, (h, w, c)


def stitch_patches_incremental(patches: List[np.ndarray], positions: List[Tuple[int, int]], original_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Stitches smaller image patches back into a single image, handling overlapping regions incrementally
    to reduce memory usage.

    Args:
        patches (List[np.ndarray]): List of image patches, each with shape (patch_height, patch_width, C).
        positions (List[Tuple[int, int]]): List of (i, j) positions representing the top-left corner of each patch.
        original_shape (Tuple[int, int, int]): The original image shape (H, W, C) before division.

    Returns:
        np.ndarray: The stitched image array in the original shape (H, W, C).
    """
    h, w, c = original_shape
    stitched_image = np.zeros((h, w, c), dtype=patches[0].dtype)

    # Place each patch at its original position in the stitched image
    for patch, (i, j) in zip(patches, positions):
        patch_h, patch_w, patch_c = patch.shape
        stitched_image[i:i + patch_h, j:j + patch_w, :] = patch[:min(patch_h, h - i), :min(patch_w, w - j), :]

    return stitched_image