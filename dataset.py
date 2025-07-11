import os.path
import torch, csv, os, rasterio, glob
import numpy as np
from torch.utils.data import Dataset
np.random.seed(42)

class HySpecNet11k(Dataset):
    def __init__(self, dataset_dir, dataset_difficulty, mode="train", bit_shift=8, is_total=True):
        self.is_clip = True
        self.is_total = is_total
        self.bit_shift = bit_shift
        self.minimum_value = 0
        self.maximum_value = 10000
        invalid_channels = (
                [126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140] +
                [160, 161, 162, 163, 164, 165, 166])
        self.valid_channels_ids = [c + 1 for c in range(224) if c not in invalid_channels]
        csv_path = os.path.join(dataset_dir, "splits", f'{dataset_difficulty}', f"{mode}.csv")
        with open(csv_path, newline='') as f: files = sum(list(csv.reader(f)), [])
        self.files = [os.path.join(dataset_dir, "patches", x.replace("DATA", "SPECTRAL_IMAGE"))
                      for x in files]
        self.imgs, self.msbs = [], []
        if self.is_total:
            for file in self.files:
                imgs, msbs = self.load_data(file)
                self.imgs.append(self.prepare_img(imgs))
                self.msbs.append(self.prepare_msb(msbs))


    def __getitem__(self, idx):
        if self.is_total:
            img, msb = self.imgs[idx], self.msbs[idx]
            return msb, img
        else:
            img, msb = self.load_data(self.files[idx])
            img, msb = self.prepare_img(img), self.prepare_msb(msb)
            return msb, img

    def __len__(self):
        return len(self.files)

    def load_data(self, file):
        img = rasterio.open(file)
        img = img.read(self.valid_channels_ids)
        if self.is_clip:
            img = np.clip(img, a_min=self.minimum_value, a_max=self.maximum_value)
        msb = img >> self.bit_shift
        return img, msb

    def prepare_msb(self, msb):
        msb = msb << self.bit_shift
        msb = (msb - self.minimum_value) / (self.maximum_value - self.minimum_value)
        msb = torch.from_numpy(msb.astype(np.float32))
        return msb

    def prepare_img(self, img):
        img = (img - self.minimum_value) / (self.maximum_value - self.minimum_value)
        img = torch.from_numpy(img.astype(np.float32))
        return img