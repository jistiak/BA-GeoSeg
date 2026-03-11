# GeoSeg Superset-Band Segmentation Pipeline

This README describes the **fixed-superset, variable-subset** segmentation workflow built on top of **GeoSeg + FT-UNetFormer**.

The goal:

* define a **superset of bands** once, for example `B, G, R, RE1, RE2, NIR, DSM`
* train **one model** on that superset
* allow each training sample to contain **any subset** of those bands
* allow inference on **any subset from the same superset** without retraining

This is not magic. If a model gets only RGB at inference time, it cannot invent NIR physics. But it **can** learn to degrade gracefully and remain usable, instead of forcing a new training run for every band combination.

---

## 1. What this pipeline does

Standard segmentation models assume a fixed input shape:

* RGB only: `[3, H, W]`
* RGB+NIR only: `[4, H, W]`
* 7-band multispectral: `[7, H, W]`

Each new band combination often means a new model and a new training run.

This pipeline avoids that by:

1. defining a **fixed band superset**
2. always packing inputs into the same band slots
3. filling missing bands with zeros
4. adding **presence planes** so the model knows which bands are actually present
5. training with **band dropout** so the model learns to handle incomplete inputs

The result is a single segmentation model that works with samples that have different available bands, as long as all bands belong to the same predefined superset.

---

## 2. Core idea

Suppose your superset is:

```python
SUP_BANDS = ["B", "G", "R", "RE1", "RE2", "NIR", "DSM"]
```

A sample might contain:

* RGB only
* RGB + NIR
* full 7 bands
* BGR + RE1 + NIR

All of these are packed into the same fixed layout.

### 2.1 Spectral planes

These are the actual band values.

For the superset above, the spectral planes are always shaped as `[7, H, W]`.

Each plane corresponds to one band slot:

| Slot | Band |
|------|------|
| 0    | B    |
| 1    | G    |
| 2    | R    |
| 3    | RE1  |
| 4    | RE2  |
| 5    | NIR  |
| 6    | DSM  |

If a band is missing for a sample, that slot is filled with zeros.

### 2.2 Presence planes

Zero-filling missing bands is not enough, because zero is ambiguous:

* the band is actually missing, **or**
* the band exists and the pixel value is legitimately near zero

To resolve this, one binary plane per band is appended. Each plane is all ones if the band is present, all zeros if absent.

For the same superset, the presence planes are `[7, H, W]`.

### 2.3 Final model input

With presence planes enabled, the final input becomes:

```
[14, H, W]
```

* first 7 channels = spectral planes
* next 7 channels = presence planes

This is the recommended setup.

---

## 3. Repository structure

The following files implement this pipeline:

```text
BA-GeoSeg/
├── config/
│   └── custom/
│       └── ft_unetformer_superset.py        # Training config
├── geoseg/
│   ├── datasets/
│   │   └── band_superset_dataset.py         # Dataset class
│   └── models/
│       └── FTUNetFormerSuperset.py          # Model wrapper + pretrained loader
├── inference_superset.py                    # Inference for normal-sized images
├── inference_huge_image_superset.py         # Inference for large rasters
└── train_supervision.py                     # Unchanged GeoSeg training entrypoint
```

---

## 4. Files overview

### 4.1 `geoseg/models/FTUNetFormerSuperset.py`

Wraps the existing FT-UNetFormer and exposes a configurable `in_chans`.

Why this is needed:

* GeoSeg's FT-UNetFormer backbone supports configurable input channels internally
* the stock wrapper is wired for 3-channel usage
* this wrapper feeds `N_bands` or `2 * N_bands` channels instead

Also includes a **safe pretrained loader** that:

* loads matching checkpoint weights
* adapts the first patch embedding conv to the new channel count
* skips incompatible tensors safely
* copies RGB weights into the first 3 channels and initializes remaining channels using the mean RGB weights

### 4.2 `geoseg/datasets/band_superset_dataset.py`

The heart of the workflow.

Responsibilities:

* read multiband raster data
* read segmentation masks
* understand which bands are available for each sample
* pack all samples into the same fixed superset layout
* apply training-time band dropout
* append presence planes
* normalize present bands only
* return a GeoSeg-compatible batch dictionary

Output keys per sample:

```python
{
    "img": tensor,             # packed + normalized superset tensor
    "gt_semantic_seg": mask,   # segmentation mask
    "img_id": ...,             # image filename
    "present_bands": ...,      # binary vector of which superset slots are filled
    "active_bands": ...        # list of band names actually used in this sample
}
```

GeoSeg only needs `img` and `gt_semantic_seg`, but the extra keys are useful for debugging and per-subset evaluation.

### 4.3 `config/custom/ft_unetformer_superset.py`

The config file that connects the dataset and model.

Defines:

* superset band names
* band statistics (mean and std per band)
* model input channel count
* train/val datasets and dataloaders
* loss, optimizer, and scheduler
* training hyperparameters

### 4.4 `inference_superset.py`

Inference script for normal-sized images or tiles.

Responsibilities:

* read an image
* determine available bands and channel order
* pack into superset format with presence planes
* run inference
* save the class-index mask and/or colorized preview

### 4.5 `inference_huge_image_superset.py`

Inference script for large rasters and huge scenes.

Responsibilities:

* read a large image
* pad to patch size
* split into tiles
* pack each tile into superset format with presence planes
* run model prediction patch by patch
* stitch the final mask back together
* crop away padding
* save the result

---

## 5. Dataset requirements

Each sample must provide:

* `image_path` — path to the raster file
* `mask_path` — path to the segmentation mask
* `available_bands` — list of band names present in this file
* `band_to_index` — mapping from band name to channel index in the raw file

Example sample list:

```python
samples = [
    {
        "image_path": "data/mydataset/train/images/tile_001.tif",
        "mask_path": "data/mydataset/train/masks/tile_001.png",
        "available_bands": ["R", "G", "B", "NIR"],
        "band_to_index": {"R": 0, "G": 1, "B": 2, "NIR": 3},
    },
    {
        "image_path": "data/mydataset/train/images/tile_002.tif",
        "mask_path": "data/mydataset/train/masks/tile_002.png",
        "available_bands": ["R", "G", "B"],
        "band_to_index": {"R": 0, "G": 1, "B": 2},
    },
]
```

You can build this from:

* a CSV file
* a JSON manifest
* a Python function that scans folders and reads raster metadata

A manifest-based approach is recommended because it makes mixed-band datasets explicit.

---

## 6. How packing works

Given:

```python
SUP_BANDS = ["B", "G", "R", "RE1", "RE2", "NIR", "DSM"]
available_bands = ["R", "G", "B", "NIR"]
band_to_index = {"R": 0, "G": 1, "B": 2, "NIR": 3}
```

The spectral planes become:

| Slot | Band | Source           |
|------|------|------------------|
| 0    | B    | raw channel 2    |
| 1    | G    | raw channel 1    |
| 2    | R    | raw channel 0    |
| 3    | RE1  | zeros (missing)  |
| 4    | RE2  | zeros (missing)  |
| 5    | NIR  | raw channel 3    |
| 6    | DSM  | zeros (missing)  |

The presence planes become `[1, 1, 1, 0, 0, 1, 0]` broadcast to `[7, H, W]`.

Final input shape: `[14, H, W]`.

---

## 7. Normalization

Provide one mean and std per band in **superset order**:

```python
BAND_MEAN = [1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 12.0]
BAND_STD  = [300.0, 320.0, 350.0, 370.0, 380.0, 400.0, 4.0]
```

Rules:

* normalize **present spectral bands only**
* do not normalize missing spectral slots
* presence planes stay binary 0/1
* do not use ImageNet RGB normalization

Compute band stats from your training set using all available samples for each band.

---

## 8. Augmentations

Use **geometric augmentations** that are valid for all bands:

```python
train_transform = albu.Compose([
    albu.RandomRotate90(p=0.5),
    albu.HorizontalFlip(p=0.5),
    albu.VerticalFlip(p=0.5),
])
```

Avoid RGB-only color transforms such as hue shift, saturation shift, or brightness/contrast. Those are often meaningless for multispectral bands.

---

## 9. Band dropout during training

Band dropout teaches the model to survive when some inputs are missing.

During training, from the bands physically available in a sample, a random subset is dropped before packing.

Example — a sample originally has `["R", "G", "B", "NIR"]`. Possible training-time active subsets:

* `R, G, B, NIR`
* `R, G, B`
* `R, G, NIR`
* `R, B`
* `NIR`

This forces the model to learn robustness to incomplete input combinations.

Recommended defaults:

```python
band_dropout_prob = 0.35
band_dropout_mode = "per_band"
min_keep_bands = 1
```

You can also bias dropout to reflect deployment scenarios. If real inference typically uses RGB-only, RGB+NIR, or full 7-band, oversampling those subsets during training is smarter than relying on random dropout alone.

---

## 10. Config example

```python
SUP_BANDS = ["B", "G", "R", "RE1", "RE2", "NIR", "DSM"]
NUM_SPECTRAL_BANDS = len(SUP_BANDS)
USE_PRESENCE_PLANES = True
MODEL_IN_CHANS = NUM_SPECTRAL_BANDS * 2 if USE_PRESENCE_PLANES else NUM_SPECTRAL_BANDS

BAND_MEAN = [1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 12.0]
BAND_STD  = [300.0, 320.0, 350.0, 370.0, 380.0, 400.0, 4.0]

net = ft_unetformer_superset(
    pretrained=True,
    num_classes=num_classes,
    freeze_stages=-1,
    decoder_channels=256,
    in_chans=MODEL_IN_CHANS,
    weight_path="pretrain_weights/stseg_base.pth",
)
```

See `config/custom/ft_unetformer_superset.py` for the full working config.

---

## 11. Training workflow

### 11.1 Prepare your dataset

Create:

* training sample manifest
* validation sample manifest
* segmentation masks
* per-band statistics (mean and std)

### 11.2 Implement `load_samples(split)`

In the config file, create a function that loads your train/val sample lists from CSV or JSON.

### 11.3 Build datasets

Use `BandSupersetSegDataset` for both train and val.

Training dataset — band dropout enabled, geometric augmentation enabled.

Validation dataset — no band dropout, no augmentation.

### 11.4 Run training

```bash
python GeoSeg/train_supervision.py -c GeoSeg/config/custom/ft_unetformer_superset.py
```

No trainer rewrite needed. This is the normal GeoSeg training entrypoint.

---

## 12. Inference on normal images

Use `inference_superset.py`.

### 12.1 Same band order for all images

If every image in a folder has the same channel order:

```bash
python inference_superset.py \
  -i data/mydataset/test_images \
  -c config/custom/ft_unetformer_superset.py \
  -o fig_results/mydataset \
  --bands R,G,B,NIR \
  --save-raw
```

This means raw file channels are in the order `R, G, B, NIR`, which are mapped to the superset slots automatically.

### 12.2 Different band order per image

Use a JSON metadata file:

```json
{
  "tile_001.tif": {
    "available_bands": ["R", "G", "B", "NIR"],
    "band_to_index": {"R": 0, "G": 1, "B": 2, "NIR": 3}
  },
  "tile_002.tif": {
    "available_bands": ["R", "G", "B"],
    "band_to_index": {"R": 0, "G": 1, "B": 2}
  }
}
```

```bash
python inference_superset.py \
  -i data/mydataset/test_images \
  -c config/custom/ft_unetformer_superset.py \
  -o fig_results/mydataset \
  --band-map-json band_map.json \
  --save-raw
```

---

## 13. Inference on huge images

Use `inference_huge_image_superset.py` for large rasters.

It reads the image, pads to patch size, tiles it, runs the model on each tile, stitches the predictions, crops away padding, and saves the result.

```bash
python inference_huge_image_superset.py \
  -i data/mydataset/huge_images \
  -c config/custom/ft_unetformer_superset.py \
  -o fig_results/mydataset_huge \
  --bands R,G,B,NIR \
  -ph 512 -pw 512 \
  -b 2 \
  --save-raw
```

Or with per-image band metadata:

```bash
python inference_huge_image_superset.py \
  -i data/mydataset/huge_images \
  -c config/custom/ft_unetformer_superset.py \
  -o fig_results/mydataset_huge \
  --band-map-json band_map.json \
  -ph 512 -pw 512 \
  -b 2 \
  --save-raw
```

---

## 14. Pretrained weights

When increasing the number of input channels, the first patch embedding conv no longer matches a normal RGB checkpoint.

The `load_pretrained_safely` function in `FTUNetFormerSuperset.py` handles this:

* all compatible layers are loaded normally
* the first conv is adapted if the checkpoint has 3 input channels
* RGB weights are copied into the RGB channels
* extra spectral and presence channels are initialized using the mean of the RGB weights
* incompatible tensors are skipped safely

This gives a better starting point than pure random initialization.

---

## 15. Evaluating the model properly

Do not evaluate only one aggregate score.

Because this is a flexible-input model, evaluate by subset:

* RGB only
* RGB + NIR
* full superset
* any common subset used in deployment

This tells you:

* whether the model truly handles missing bands
* how gracefully performance drops when informative bands are absent
* whether any particular subset is undertrained

---

## 16. Common mistakes

**Mistake 1: zero-fill missing bands without presence planes**  
This makes missingness ambiguous. Use presence planes.

**Mistake 2: normalize missing bands**  
Only normalize present spectral bands.

**Mistake 3: use RGB-only augmentations on multispectral data**  
Avoid brightness/hue/saturation unless you have a physically meaningful reason.

**Mistake 4: train only on full-band samples and expect robustness to subsets**  
Use band dropout or mixed real subset training.

**Mistake 5: ignore real deployment subset frequencies**  
If most inference is RGB-only or RGB+NIR, reflect that during training.

**Mistake 6: assume all modalities are perfectly aligned**  
This pipeline assumes all bands within a sample are co-registered and share the same resolution. If your DSM or other modality is on a different grid, resample and align it before training or inference.

**Mistake 7: forget geospatial metadata during huge-image inference**  
The inference scripts save masks as arrays. They do not automatically preserve CRS, transform, or georeferencing. If you need geospatially correct outputs, write a `rasterio` version that copies metadata from the source raster.

---

## 17. Recommended starting settings

```python
SUP_BANDS = ["B", "G", "R", "RE1", "RE2", "NIR", "DSM"]
USE_PRESENCE_PLANES = True
band_dropout_prob = 0.35
band_dropout_mode = "per_band"
min_keep_bands = 1
patch_size = 512
batch_size = 4
```

Model: FT-UNetFormer backbone, superset input channels, standard decoder, pretrained weight adaptation for the first conv.

Training: mixed real subsets, random band dropout, geometric augmentation only.

---

## 18. Summary

This pipeline provides:

* one segmentation model
* one fixed superset of bands
* flexible training on mixed subset availability
* flexible inference on any subset from the same superset
* no model retraining for every band combination

The key ingredients:

* fixed band slots with zero-filled missing spectral planes
* binary presence planes
* band dropout during training
* a model wrapper that accepts more than 3 input channels

In one sentence: **pack everything into the same superset tensor, tell the model which bands are present, and train it to survive incomplete inputs.**

---

## 19. Suggested next steps

1. Add a `load_samples()` implementation for your dataset in the config file
2. Prepare sample manifests for train and val
3. Compute band-wise mean and std
4. Train the model
5. Evaluate by subset (RGB only, RGB+NIR, full superset)
6. Run normal and huge-image inference
7. Optionally upgrade the huge-image script to preserve georeferencing with `rasterio`

---

## 20. Implementation checklist

- [x] `geoseg/models/FTUNetFormerSuperset.py` — model wrapper with safe pretrained loader
- [x] `geoseg/datasets/band_superset_dataset.py` — dataset class with band dropout and presence planes
- [x] `config/custom/ft_unetformer_superset.py` — training config
- [x] `inference_superset.py` — inference for normal-sized images
- [x] `inference_huge_image_superset.py` — inference for large rasters
- [ ] Implement `load_samples()` for your dataset
- [ ] Prepare sample manifests (train and val)
- [ ] Compute `BAND_MEAN` and `BAND_STD` from training data
- [ ] Train using `train_supervision.py`
- [ ] Validate on common subsets (RGB only, RGB+NIR, full superset)
- [ ] Run inference on test images and large rasters

---

## 21. Notes for future upgrades

Once the baseline works, consider:

* subset-aware sampling probabilities (oversample sparse subsets)
* separate metrics per subset family
* confidence maps and uncertainty estimates
* `rasterio`-based huge-image inference with preserved CRS and transform
* overlap-tile inference with blending if border artifacts appear
* stronger initialization strategies for non-RGB channels

Get the baseline working before optimizing. Let the ablations decide what to improve next.
