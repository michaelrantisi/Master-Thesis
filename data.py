import os
import numpy as np
import torch
from torch.utils.data import Dataset
from mmf import eps_eff_map_from_rgb
from tiling import read_rgb01, tile_image, tissue_fraction, aug_geometric, aug_photometric

class ChoroidTileDataset(Dataset):
    def __init__(
        self,
        image_paths,
        label: int,
        tile=256,
        stride=128,
        min_tissue=0.10,
        bg_thresh=0.95,
        n_aug=4,               
        eps_host=2.0,
        eps_channels=(3.0,4.0,5.0),
        eps_clip=(1.0,20.0),
        seed=42,
    ):
        self.image_paths = list(image_paths)
        self.label = int(label)
        self.tile = tile
        self.stride = stride
        self.min_tissue = min_tissue
        self.bg_thresh = bg_thresh
        self.n_aug = int(n_aug)
        self.eps_host = eps_host
        self.eps_channels = eps_channels
        self.eps_clip = eps_clip
        self.rng = np.random.default_rng(seed)

        # Pre-index all tiles (so __len__ is stable)
        self.index = []  
        for p in self.image_paths:
            img = read_rgb01(p)
            for patch, (y, x) in tile_image(img, tile=self.tile, stride=self.stride):
                if tissue_fraction(patch, bg_thresh=self.bg_thresh) >= self.min_tissue:
                    self.index.append((p, y, x))

        # Each tile expanded by n_aug
        self.total = len(self.index) * self.n_aug

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        base_i = idx // self.n_aug
        p, y, x = self.index[base_i]

        img = read_rgb01(p)
        patch = img[y:y+self.tile, x:x+self.tile].copy()

        # Augment
        rng = np.random.default_rng(self.rng.integers(0, 2**31-1))
        patch = aug_geometric(patch, rng)
        patch = aug_photometric(patch, rng)

        # ε_eff map
        eps = eps_eff_map_from_rgb(
            patch,
            eps_host=self.eps_host,
            eps_channels=self.eps_channels,
            clip=self.eps_clip,
        )  
        # normalize eps to 0..1 for NN input
        eps01 = (eps - self.eps_clip[0]) / (self.eps_clip[1] - self.eps_clip[0] + 1e-8)
        eps01 = np.clip(eps01, 0.0, 1.0).astype(np.float32)

        x4 = np.concatenate([patch, eps01[..., None]], axis=2)  
        x4 = torch.from_numpy(x4).permute(2,0,1).float()        

        ylab = torch.tensor(self.label).long()
        return x4, ylab
