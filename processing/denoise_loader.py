import random
from processing.denoise_dataset import DnDataset
import os
from processing.data_processing import extract_patches
from torch.utils.data import DataLoader


def get_dn_data_loaders(cfg):
    """
    Creates a PyTorch data using the image patch custom dataset

    Parameters
    ----------
    cfg: dict
        configuration parameters

    Returns
    -------
    data_loader: torch.utils.data
    """
    # Get the data path
    base_path = cfg['base_path']
    train_path = base_path + cfg['train_path']
    train_noisy_path = base_path + cfg['train_noisy_path']
    val_path = base_path + cfg['val_path']
    val_noisy_path = base_path + cfg['val_noisy_path']

    # Get the batch size
    batch_size = cfg['dn_batch_size']

    # Get the images paths
    train_images = [os.path.join(train_path, s) for s in os.listdir(train_path)
                    if s.endswith('.jpg') or s.endswith('.jpeg')]
    train_noisy_images = [os.path.join(train_noisy_path, s) for s in os.listdir(train_noisy_path)
                          if s.endswith('.jpg') or s.endswith('.jpeg')]
    val_images = [os.path.join(val_path, s) for s in os.listdir(val_path)
                  if s.endswith('.jpg') or s.endswith('.jpeg')]
    val_noisy_images = [os.path.join(val_noisy_path, s) for s in os.listdir(val_noisy_path)
                        if s.endswith('.jpg') or s.endswith('.jpeg')]

    # train_images = train_images[:20]
    # train_noisy_images = train_noisy_images[:20]
    #
    # val_images = val_images[:10]
    # val_noisy_images = val_noisy_images[:10]

    patch_size = cfg['dn_patch_size']
    patch_stride = cfg['dn_patch_stride']

    # Get patches
    train_patches, train_noisy_patches = extract_patches(train_images,
                                                         train_noisy_images,
                                                         patch_size,
                                                         patch_stride)
    val_patches, val_noisy_patches = extract_patches(val_images,
                                                     val_noisy_images,
                                                     patch_size,
                                                     patch_stride)

    # Create training dataloader
    train_dataset = DnDataset(cfg,
                              train_patches,
                              train_noisy_patches,
                              mode='train')
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        pin_memory=True
    )

    # Create validation dataloader
    val_dataset = DnDataset(cfg,
                            val_patches,
                            val_noisy_patches,
                            mode='val')
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size
    )

    return train_loader, val_loader
