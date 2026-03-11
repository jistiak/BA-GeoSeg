import argparse
import glob
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import tifffile as tiff
import torch
import ttach as tta
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from tools.cfg import py2cfg
from train_supervision import Supervision_Train


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# -----------------------------
# Optional color renderers
# -----------------------------
def building_to_rgb(mask):
    h, w = mask.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask_rgb[mask == 0] = [255, 255, 255]
    mask_rgb[mask == 1] = [0, 0, 0]
    return mask_rgb


def pv2rgb(mask):
    h, w = mask.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask_rgb[mask == 3] = [0, 255, 0]
    mask_rgb[mask == 0] = [255, 255, 255]
    mask_rgb[mask == 1] = [255, 0, 0]
    mask_rgb[mask == 2] = [255, 255, 0]
    mask_rgb[mask == 4] = [0, 204, 255]
    mask_rgb[mask == 5] = [0, 0, 255]
    return mask_rgb


def landcoverai_to_rgb(mask):
    h, w = mask.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask_rgb[mask == 3] = [255, 255, 255]
    mask_rgb[mask == 0] = [233, 193, 133]
    mask_rgb[mask == 1] = [255, 0, 0]
    mask_rgb[mask == 2] = [0, 255, 0]
    return cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)


def uavid2rgb(mask):
    h, w = mask.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask_rgb[mask == 0] = [128, 0, 0]
    mask_rgb[mask == 1] = [128, 64, 128]
    mask_rgb[mask == 2] = [0, 128, 0]
    mask_rgb[mask == 3] = [128, 128, 0]
    mask_rgb[mask == 4] = [64, 0, 128]
    mask_rgb[mask == 5] = [192, 0, 192]
    mask_rgb[mask == 6] = [64, 64, 0]
    mask_rgb[mask == 7] = [0, 0, 0]
    return cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)


def get_color_mask(mask, dataset_name):
    if dataset_name == "landcoverai":
        return landcoverai_to_rgb(mask)
    if dataset_name == "pv":
        return pv2rgb(mask)
    if dataset_name == "uavid":
        return uavid2rgb(mask)
    if dataset_name == "building":
        return building_to_rgb(mask)
    return None


# -----------------------------
# CLI
# -----------------------------
def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg("-i", "--image_path", type=Path, required=True, help="Path to huge image folder or a single image")
    arg("-c", "--config_path", type=Path, required=True, help="Path to config")
    arg("-o", "--output_path", type=Path, required=True, help="Path to save results")
    arg("-t", "--tta", default=None, choices=[None, "d4", "lr"], help="Test-time augmentation")
    arg("-ph", "--patch-height", type=int, default=512, help="Patch height")
    arg("-pw", "--patch-width", type=int, default=512, help="Patch width")
    arg("-b", "--batch-size", type=int, default=2, help="Batch size")
    arg(
        "-d",
        "--dataset",
        default="none",
        choices=["none", "pv", "landcoverai", "uavid", "building"],
        help="Optional color rendering preset"
    )

    # Band specification
    arg(
        "--bands",
        type=str,
        default=None,
        help='Comma-separated band names for all input images, e.g. "R,G,B,NIR"'
    )
    arg(
        "--band-map-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping image names to available bands and band_to_index. "
            "Overrides --bands on matching files."
        )
    )

    # Output behavior
    arg("--save-raw", action="store_true", help="Save raw class-index mask as .tif")
    arg("--save-color", action="store_true", help="Save colorized preview as .png")

    return parser.parse_args()


# -----------------------------
# Band metadata helpers
# -----------------------------
def parse_bands_string(bands_str):
    if bands_str is None:
        return None
    bands = [b.strip() for b in bands_str.split(",") if b.strip()]
    return bands if bands else None


def load_band_map_json(path):
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_band_info_for_image(img_path, band_map_json, fallback_bands):
    """
    Returns:
        available_bands: list[str]
        band_to_index: dict[str, int]
    """
    name = Path(img_path).name

    if name in band_map_json:
        meta = band_map_json[name]
        available_bands = meta["available_bands"]
        band_to_index = meta["band_to_index"]
        band_to_index = {k: int(v) for k, v in band_to_index.items()}
        return available_bands, band_to_index

    if fallback_bands is not None:
        available_bands = fallback_bands
        band_to_index = {b: i for i, b in enumerate(available_bands)}
        return available_bands, band_to_index

    ext = Path(img_path).suffix.lower()
    if ext in [".png", ".jpg", ".jpeg"]:
        available_bands = ["R", "G", "B"]
        band_to_index = {"R": 0, "G": 1, "B": 2}
        return available_bands, band_to_index

    raise ValueError(
        f"No band metadata found for {name}. Use --bands or --band-map-json."
    )


# -----------------------------
# Image IO
# -----------------------------
def read_image_hwc(img_path):
    ext = Path(img_path).suffix.lower()

    if ext in [".tif", ".tiff"]:
        img = tiff.imread(str(img_path))
        if img.ndim == 2:
            img = img[..., None]

        # Heuristic CHW -> HWC
        if img.ndim == 3:
            if img.shape[0] <= 64 and img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
                img = np.transpose(img, (1, 2, 0))

        if img.ndim != 3:
            raise ValueError(f"Unsupported TIFF shape for {img_path}: {img.shape}")

        return img.astype(np.float32)

    if ext in [".png", ".jpg", ".jpeg"]:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32)

    raise ValueError(f"Unsupported image extension: {ext}")


def get_img_padded(image_hwc, patch_size):
    oh, ow, c = image_hwc.shape
    rh, rw = oh % patch_size[0], ow % patch_size[1]
    width_pad = 0 if rw == 0 else patch_size[1] - rw
    height_pad = 0 if rh == 0 else patch_size[0] - rh

    img_pad = np.pad(
        image_hwc,
        ((0, height_pad), (0, width_pad), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    return img_pad, height_pad, width_pad


# -----------------------------
# Packing logic
# -----------------------------
def pack_tile_to_superset(
    tile_hwc,
    available_bands,
    band_to_index,
    superset_bands,
    band_mean,
    band_std,
    use_presence_planes=True,
):
    h, w, _ = tile_hwc.shape
    n = len(superset_bands)
    superset_slot = {b: i for i, b in enumerate(superset_bands)}

    x = np.zeros((n, h, w), dtype=np.float32)
    presence = np.zeros((n,), dtype=np.float32)

    for band_name in available_bands:
        if band_name not in superset_slot:
            continue
        src_idx = band_to_index[band_name]
        slot = superset_slot[band_name]
        x[slot] = tile_hwc[:, :, src_idx]
        presence[slot] = 1.0

    # Normalize present bands only
    if band_mean is not None and band_std is not None:
        for i in range(n):
            if presence[i] > 0:
                x[i] = (x[i] - band_mean[i]) / (band_std[i] + 1e-6)

    if use_presence_planes:
        presence_planes = np.broadcast_to(presence[:, None, None], (n, h, w)).astype(np.float32)
        x = np.concatenate([x, presence_planes], axis=0)

    return torch.from_numpy(x).float()


# -----------------------------
# Dataset
# -----------------------------
class InferenceSupersetDataset(Dataset):
    def __init__(
        self,
        image_pad,
        patch_size,
        available_bands,
        band_to_index,
        superset_bands,
        band_mean,
        band_std,
        use_presence_planes=True,
    ):
        self.image_pad = image_pad
        self.patch_h, self.patch_w = patch_size
        self.available_bands = available_bands
        self.band_to_index = band_to_index
        self.superset_bands = superset_bands
        self.band_mean = band_mean
        self.band_std = band_std
        self.use_presence_planes = use_presence_planes

        h, w = image_pad.shape[:2]
        self.coords = []
        for row in range(0, h, self.patch_h):
            for col in range(0, w, self.patch_w):
                self.coords.append((row, col))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, index):
        row, col = self.coords[index]
        tile = self.image_pad[row:row + self.patch_h, col:col + self.patch_w]

        img = pack_tile_to_superset(
            tile_hwc=tile,
            available_bands=self.available_bands,
            band_to_index=self.band_to_index,
            superset_bands=self.superset_bands,
            band_mean=self.band_mean,
            band_std=self.band_std,
            use_presence_planes=self.use_presence_planes,
        )

        return {
            "img": img,
            "row": row,
            "col": col,
        }


# -----------------------------
# Main
# -----------------------------
def main():
    args = get_args()
    seed_everything(42)

    patch_size = (args.patch_height, args.patch_width)
    config = py2cfg(args.config_path)

    # Required config fields from your superset-band training config
    superset_bands = list(config.SUP_BANDS)
    band_mean = getattr(config, "BAND_MEAN", None)
    band_std = getattr(config, "BAND_STD", None)
    use_presence_planes = getattr(config, "USE_PRESENCE_PLANES", True)

    if band_mean is not None:
        band_mean = np.array(band_mean, dtype=np.float32)
    if band_std is not None:
        band_std = np.array(band_std, dtype=np.float32)

    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name + ".ckpt"),
        config=config,
    )
    model.cuda()
    model.eval()

    if args.tta == "lr":
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
        ])
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.Scale(scales=[0.75, 1.0, 1.25, 1.5, 1.75]),
        ])
        model = tta.SegmentationTTAWrapper(model, transforms)

    band_map_json = load_band_map_json(args.band_map_json)
    fallback_bands = parse_bands_string(args.bands)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    img_paths = []
    if args.image_path.is_file():
        img_paths = [str(args.image_path)]
    else:
        for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"):
            img_paths.extend(glob.glob(os.path.join(str(args.image_path), ext)))
        img_paths.sort()

    if len(img_paths) == 0:
        raise FileNotFoundError(f"No images found in {args.image_path}")

    for img_path in img_paths:
        img_name = Path(img_path).name
        stem = Path(img_path).stem

        available_bands, band_to_index = get_band_info_for_image(
            img_path=img_path,
            band_map_json=band_map_json,
            fallback_bands=fallback_bands,
        )

        img = read_image_hwc(img_path)
        img_pad, height_pad, width_pad = get_img_padded(img, patch_size)

        output_height, output_width = img_pad.shape[:2]
        orig_height, orig_width = img.shape[:2]

        dataset = InferenceSupersetDataset(
            image_pad=img_pad,
            patch_size=patch_size,
            available_bands=available_bands,
            band_to_index=band_to_index,
            superset_bands=superset_bands,
            band_mean=band_mean,
            band_std=band_std,
            use_presence_planes=use_presence_planes,
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
        )

        output_mask = np.zeros((output_height, output_width), dtype=np.uint8)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Inferencing {img_name}"):
                logits = model(batch["img"].cuda())

                # Be robust if some model variant returns aux outputs
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]

                preds = logits.argmax(dim=1).cpu().numpy()
                rows = batch["row"].cpu().numpy()
                cols = batch["col"].cpu().numpy()

                for i in range(preds.shape[0]):
                    row = int(rows[i])
                    col = int(cols[i])
                    output_mask[row:row + patch_size[0], col:col + patch_size[1]] = preds[i]

        # Crop away padding
        if height_pad > 0:
            output_mask = output_mask[:-height_pad, :]
        if width_pad > 0:
            output_mask = output_mask[:, :-width_pad]

        assert output_mask.shape == (orig_height, orig_width), (
            f"Output shape mismatch: got {output_mask.shape}, expected {(orig_height, orig_width)}"
        )

        # Save raw class-index mask
        if args.save_raw or not args.save_color:
            raw_out = os.path.join(args.output_path, f"{stem}_mask.tif")
            tiff.imwrite(raw_out, output_mask.astype(np.uint8))

        # Save color preview
        if args.save_color:
            color_mask = get_color_mask(output_mask, args.dataset)
            if color_mask is None:
                raise ValueError("--save-color requires a valid --dataset palette, not 'none'")
            color_out = os.path.join(args.output_path, f"{stem}_color.png")
            cv2.imwrite(color_out, color_mask)

    print("Done.")


if __name__ == "__main__":
    main()
