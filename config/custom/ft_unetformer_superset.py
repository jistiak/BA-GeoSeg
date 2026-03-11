import os
from torch.utils.data import DataLoader
import torch
from catalyst.contrib.nn import Lookahead
from catalyst import utils
from torch_optimizer import RAdam
import albumentations as albu

from geoseg.losses import JointLoss, SoftCrossEntropyLoss, DiceLoss
from geoseg.datasets.band_superset_dataset import BandSupersetSegDataset
from geoseg.models.FTUNetFormerSuperset import ft_unetformer_superset

# -----------------------------
# Basic settings
# -----------------------------
max_epoch = 80
train_batch_size = 4
val_batch_size = 4
num_classes = 6
ignore_index = 255
classes = ["cls0", "cls1", "cls2", "cls3", "cls4", "cls5"]

gpus = [0]
save_top_k = 1
save_last = True
monitor = "val_mIoU"
monitor_mode = "max"
weights_name = "ft_unetformer_superset"
weights_path = "model_weights/mydataset"
test_weights_name = weights_name
log_name = "mydataset/ft_unetformer_superset"
pretrained_ckpt_path = None
use_aux_loss = False

# -----------------------------
# Bands
# -----------------------------
SUP_BANDS = ["B", "G", "R", "RE1", "RE2", "NIR", "DSM"]
NUM_SPECTRAL_BANDS = len(SUP_BANDS)
USE_PRESENCE_PLANES = True
MODEL_IN_CHANS = NUM_SPECTRAL_BANDS * 2 if USE_PRESENCE_PLANES else NUM_SPECTRAL_BANDS

BAND_MEAN = [1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 12.0]  # example
BAND_STD  = [300.0, 320.0, 350.0, 370.0, 380.0, 400.0,  4.0]        # example

# -----------------------------
# Model
# -----------------------------
net = ft_unetformer_superset(
    pretrained=True,
    num_classes=num_classes,
    freeze_stages=-1,
    decoder_channels=256,
    in_chans=MODEL_IN_CHANS,
    weight_path="pretrain_weights/stseg_base.pth",
)

# -----------------------------
# Loss
# -----------------------------
loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(mode="multiclass", ignore_index=ignore_index),
    1.0,
    1.0,
)

# -----------------------------
# Augmentations
# Only geometric transforms here.
# No RGB-only color jitter clownery.
# -----------------------------
train_transform = albu.Compose([
    albu.RandomRotate90(p=0.5),
    albu.HorizontalFlip(p=0.5),
    albu.VerticalFlip(p=0.5),
])

val_transform = albu.Compose([])

# -----------------------------
# Build sample lists
# Replace these with your manifest loader.
# -----------------------------
def load_samples(split):
    # You implement this from CSV/JSON.
    raise NotImplementedError

train_samples = load_samples("train")
val_samples = load_samples("val")

train_dataset = BandSupersetSegDataset(
    samples=train_samples,
    superset_bands=SUP_BANDS,
    num_classes=num_classes,
    transform=train_transform,
    train=True,
    band_dropout_prob=0.35,
    band_dropout_mode="per_band",
    min_keep_bands=1,
    use_presence_planes=USE_PRESENCE_PLANES,
    band_mean=BAND_MEAN,
    band_std=BAND_STD,
    ignore_index=ignore_index,
)

val_dataset = BandSupersetSegDataset(
    samples=val_samples,
    superset_bands=SUP_BANDS,
    num_classes=num_classes,
    transform=val_transform,
    train=False,
    band_dropout_prob=0.0,
    min_keep_bands=1,
    use_presence_planes=USE_PRESENCE_PLANES,
    band_mean=BAND_MEAN,
    band_std=BAND_STD,
    ignore_index=ignore_index,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=val_batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
)

# -----------------------------
# Optimizer / scheduler
# -----------------------------
base_optimizer = RAdam(utils.process_model_params(net, layerwise_params={"backbone.*": dict(lr=6e-5)}), lr=6e-4, weight_decay=1e-4)
optimizer = Lookahead(base_optimizer)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)
