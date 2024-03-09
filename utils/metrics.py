import torch


def psnr(pred, orig, max_val=None):
    """
    Compute the Peak Signal-to-Noise Ratio between two images.

    Args:
    - true_img (torch.Tensor): The ground truth image.
    - pred_img (torch.Tensor): The predicted image.
    - max_val (float): The maximum possible pixel value of the images.

    Returns:
    - float: The PSNR value.
    """
    if max_val is None:
        max_val = torch.max(orig)

    mse = torch.mean((orig - pred) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))