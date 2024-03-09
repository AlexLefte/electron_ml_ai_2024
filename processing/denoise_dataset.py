import torch
from torch.utils.data import Dataset
import os


class DnDataset(Dataset):
    def __init__(self,
                 cfg: dict,
                 orig_patches: list,
                 noisy_patches: list,
                 mode: str):
        """
        Constructor
        :param cfg:
        :param orig_patches:
        :param mode:
        """
        # Store gt, noise image paths
        self.orig_patches = orig_patches
        self.noisy_patches = noisy_patches

        # Train/Val datasets
        self.mode = mode
        if self.mode == 'train':
            self.data_path = cfg['train_path']
        else:
            self.data_path = cfg['val_path']

        # Length
        self.count = len(self.orig_patches)

    def __len__(self):
        """
        Returns the length of the custom dataset.
        Must be implemented.
        """
        return self.count

    def __getitem__(self, idx):
        """
        Returns the reference and the noisy image.
        Must be implemented.
        """
        orig, noisy = (self.orig_patches[idx, :, :, :].transpose(2, 0, 1),
                       self.noisy_patches[idx, :, :, :].transpose(2, 0, 1))

        return {
            'orig': orig,
            'noisy': noisy
        }
