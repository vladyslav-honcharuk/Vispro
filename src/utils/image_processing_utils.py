import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple
import torch

__all__ = ["binarize_array", "find_nearest_multiple_of_32", "get_image_var", "get_image_var"]

cuda = True if torch.cuda.is_available() else False
torch.cuda.set_device(0)
if cuda:
    device = 'cuda'
else:
    device = 'cpu'



def binarize_array(array, threshold):
    """
    Binarizes a numpy array based on a threshold determined by the given percentile.

    :param array: numpy array to be binarized
    :param percentile: percentile value used to determine the threshold, defaults to 50 (median)
    :return: binarized numpy array
    """
    binary_array = (array >= threshold).astype(int)

    return binary_array

def find_nearest_multiple_of_32(x):
    base = 32
    remainder = x % base
    if remainder == 0:
        return x
    else:
        return x + (base - remainder)

def get_image_var(image_name: str) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Loads an image, resizes it to the nearest multiple of 32,
    and transforms it to a tensor format compatible with a model.

    Args:
        image_name (str): The path to the image file.

    Returns:
        Tuple[torch.Tensor, np.ndarray]: A tuple containing:
            - img_var: The transformed image tensor ready for model input.
            - img_np: The image as a NumPy array in its resized form.
    """
    # Load the image and extract dimensions
    img_pil = Image.open(image_name.split(' ')[0])
    h, w = img_pil.size

    # Resize dimensions to nearest multiple of 32
    h_new = find_nearest_multiple_of_32(h)
    w_new = find_nearest_multiple_of_32(w)
    img_pil = img_pil.resize((h_new, w_new), Image.LANCZOS)

    # Convert to NumPy array and prepare tensor
    img_np = np.array(img_pil)
    transforms_rgb = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img_var = transforms_rgb(img_pil)
    img_var = torch.unsqueeze(img_var, dim=0).to(device)

    return img_var, img_np
