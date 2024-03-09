import numpy as np
from skimage.util import view_as_windows
from skimage import io


def extract_patches(gt_images, noisy_images, patch_size=40, stride=40):
    """
    Extract patches of given size from a list of images with a specified stride.

    Parameters
    ----------
    -
    gt_images: list
        A list of NumPy arrays representing the images.
    patch_size: tuple
        A tuple specifying the height and width of the patches.
    stride: int
        The stride with which to slide the window across the image.

    Returns
    -------
    A list of patches extracted from the images.
    """
    noisy_patches = []
    reference_patches = []
    patch_size = (patch_size, patch_size, 3)
    stride = (stride, stride, 3)
    for reference_image_path, noisy_image_path in zip(gt_images, noisy_images):
        # Read images
        reference_image = io.imread(reference_image_path).astype(np.float32) / 255.0
        noisy_image = io.imread(noisy_image_path).astype(np.float32) / 255.0

        if len(reference_image.shape) == 2:
            reference_image = np.stack((reference_image,) * 3, axis=-1)
        if len(noisy_image.shape) == 2:
            noisy_image = np.stack((noisy_image,) * 3, axis=-1)

        if reference_image.shape != noisy_image.shape:
            print('Not Ok')

        # Ensure images are large enough for at least one patch
        if noisy_image.shape[0] < patch_size[0] or noisy_image.shape[1] < patch_size[1]:
            continue

        # Extract patches using a sliding window approach
        noisy_image_patches = view_as_windows(noisy_image, patch_size, step=stride)
        reference_image_patches = view_as_windows(reference_image, patch_size, step=stride)

        # Reshape to list of patches: (num_patches, patch_size[0], patch_size[1], channels)
        noisy_image_patches = noisy_image_patches.reshape(-1, *patch_size)
        reference_image_patches = reference_image_patches.reshape(-1, *patch_size)

        noisy_patches.extend(noisy_image_patches)
        reference_patches.extend(reference_image_patches)

    return np.array(noisy_patches), np.array(reference_patches)