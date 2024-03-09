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

        # Ensure images are large enough for at least one patch
        if noisy_image.shape[0] < patch_size[0] or noisy_image.shape[1] < patch_size[1]:
            continue

        # Extract patches using a sliding window approach
        noisy_image_patches = view_as_windows(noisy_image, patch_size, step=stride)
        reference_image_patches = view_as_windows(reference_image, patch_size, step=stride)

        # Reshape to list of patches: (num_patches, patch_size[0], patch_size[1], channels)
        noisy_image_patches = noisy_image_patches.reshape(-1, *patch_size)
        reference_image_patches = reference_image_patches.reshape(-1, *patch_size)

        # Limit the number of patches per image to 20
        num_patches_to_select = min(10, noisy_image_patches.shape[0])

        # Generate random indexes
        random_indices = np.random.choice(noisy_image_patches.shape[0], num_patches_to_select, replace=False)

        # Select patches at these indices from both arrays
        selected_noisy_patches = noisy_image_patches[random_indices]
        selected_reference_patches = reference_image_patches[random_indices]

        noisy_patches.extend(selected_noisy_patches)
        reference_patches.extend(selected_reference_patches)


    return np.array(noisy_patches), np.array(reference_patches)


def extract_test_patches(orig_path, noisy_path, patch_size=40, stride=40):
    # Read the image and clip to the [0, 1] range
    reference_image = io.imread(orig_path).astype(np.float32) / 255.0
    noisy_image = io.imread(noisy_path).astype(np.float32) / 255.0

    # Adjust shape (e.g. grayscale images -> add 2 additional chanels)
    if len(reference_image.shape) == 2:
        reference_image = np.stack((reference_image,) * 3, axis=-1)
    if len(noisy_image.shape) == 2:
        noisy_image = np.stack((noisy_image,) * 3, axis=-1)

    # Initial shape
    initial_shape = reference_image.shape

    # Patch size
    patch_size = (patch_size, patch_size, 3)
    stride = (stride, stride, 3)

    # Pad images to fit the patch size
    reference_image_padded = pad_image_to_fit_patch_size(image=reference_image,
                                                         patch_size=patch_size)
    noisy_image_padded = pad_image_to_fit_patch_size(image=noisy_image,
                                                     patch_size=patch_size)

    # Extract patches using a sliding window approach
    reference_image_patches = view_as_windows(reference_image_padded, patch_size, step=stride)
    noisy_image_patches = view_as_windows(noisy_image_padded, patch_size, step=stride)

    # Reshape to list of patches: (num_patches, patch_size[0], patch_size[1], channels)
    noisy_image_patches = noisy_image_patches.reshape(-1, *patch_size)
    reference_image_patches = reference_image_patches.reshape(-1, *patch_size)

    # Return
    return (reference_image_patches,
            noisy_image_patches,
            initial_shape)


# Function to pad an image so that its dimensions are multiples of the patch size
def pad_image_to_fit_patch_size(image, patch_size):
    h, w, _ = patch_size
    pad_height = (h - image.shape[0] % h) % h
    pad_width = (w - image.shape[1] % w) % w

    # Pad image - replicate the edge values for padding
    padded_image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='reflect')
    return padded_image



