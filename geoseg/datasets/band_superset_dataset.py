import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import tifffile as tiff
import albumentations as albu


class BandSupersetSegDataset(Dataset):
    """
    Each sample entry should provide:
      - image_path
      - mask_path
      - available_bands: e.g. ["R","G","B","NIR"]
      - band_to_index: mapping inside the raster, e.g. {"R":0,"G":1,"B":2,"NIR":3}
    """

    def __init__(
        self,
        samples,
        superset_bands,
        num_classes,
        transform=None,
        train=True,
        band_dropout_prob=0.3,
        band_dropout_mode="per_band",   # "per_band" or "subset"
        min_keep_bands=1,
        use_presence_planes=True,
        band_mean=None,
        band_std=None,
        ignore_index=255,
    ):
        self.samples = samples
        self.superset_bands = superset_bands
        self.band_to_slot = {b: i for i, b in enumerate(superset_bands)}
        self.num_bands = len(superset_bands)
        self.num_classes = num_classes
        self.transform = transform
        self.train = train
        self.band_dropout_prob = band_dropout_prob
        self.band_dropout_mode = band_dropout_mode
        self.min_keep_bands = min_keep_bands
        self.use_presence_planes = use_presence_planes
        self.ignore_index = ignore_index

        self.band_mean = np.array(band_mean, dtype=np.float32) if band_mean is not None else None
        self.band_std = np.array(band_std, dtype=np.float32) if band_std is not None else None

    def __len__(self):
        return len(self.samples)

    def _read_image(self, image_path):
        # For multispectral TIFF/GeoTIFF stacks
        arr = tiff.imread(image_path)

        # Force to HWC
        if arr.ndim == 3 and arr.shape[0] < arr.shape[-1]:
            # likely CHW -> HWC
            arr = np.transpose(arr, (1, 2, 0))

        arr = arr.astype(np.float32)
        return arr

    def _read_mask(self, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask.ndim == 3:
            mask = mask[..., 0]
        return mask.astype(np.int64)

    def _choose_active_bands(self, available_bands):
        available_bands = list(available_bands)

        if not self.train or self.band_dropout_prob <= 0:
            return available_bands

        if len(available_bands) <= self.min_keep_bands:
            return available_bands

        if self.band_dropout_mode == "per_band":
            kept = [b for b in available_bands if random.random() > self.band_dropout_prob]
            if len(kept) < self.min_keep_bands:
                kept = random.sample(available_bands, self.min_keep_bands)
            return kept

        if self.band_dropout_mode == "subset":
            k_min = self.min_keep_bands
            k_max = len(available_bands)
            k = random.randint(k_min, k_max)
            return random.sample(available_bands, k)

        return available_bands

    def _pack_superset(self, image_hwc, available_bands, band_to_index):
        """
        image_hwc contains only the physically available bands in some order.
        band_to_index tells which image channel corresponds to band name.
        """
        h, w, _ = image_hwc.shape
        x = np.zeros((self.num_bands, h, w), dtype=np.float32)
        presence = np.zeros((self.num_bands,), dtype=np.float32)

        active_bands = self._choose_active_bands(available_bands)

        for band_name in active_bands:
            slot = self.band_to_slot[band_name]
            src_idx = band_to_index[band_name]
            x[slot] = image_hwc[:, :, src_idx]
            presence[slot] = 1.0

        return x, presence, active_bands

    def _normalize_present_bands(self, x, presence):
        if self.band_mean is None or self.band_std is None:
            return x

        for i in range(self.num_bands):
            if presence[i] > 0:
                x[i] = (x[i] - self.band_mean[i]) / (self.band_std[i] + 1e-6)

        return x

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        mask_path = sample["mask_path"]
        available_bands = sample["available_bands"]
        band_to_index = sample["band_to_index"]

        img_hwc = self._read_image(image_path)         # HWC
        mask = self._read_mask(mask_path)              # HW

        x, presence_vec, active_bands = self._pack_superset(
            img_hwc, available_bands, band_to_index
        )  # x: [B,H,W]

        # Apply geometric transforms only
        if self.transform is not None:
            x_hwc = np.transpose(x, (1, 2, 0))
            aug = self.transform(image=x_hwc, mask=mask)
            x = np.transpose(aug["image"], (2, 0, 1))
            mask = aug["mask"]

        x = self._normalize_present_bands(x, presence_vec)

        if self.use_presence_planes:
            h, w = x.shape[1], x.shape[2]
            presence_planes = np.repeat(
                presence_vec[:, None, None], h, axis=1
            )
            presence_planes = np.repeat(
                presence_planes[:, :, None], w, axis=2
            ).reshape(self.num_bands, h, w)

            x = np.concatenate([x, presence_planes], axis=0)

        img = torch.from_numpy(x).float()
        mask = torch.from_numpy(mask).long()
        present_bands = torch.from_numpy(presence_vec).float()

        return {
            "img": img,
            "gt_semantic_seg": mask,
            "img_id": os.path.basename(image_path),
            "present_bands": present_bands,
            "active_bands": active_bands,
        }
