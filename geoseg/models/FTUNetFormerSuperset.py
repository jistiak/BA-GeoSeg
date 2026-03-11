import torch
import torch.nn as nn

# Reuse the existing heavy machinery from GeoSeg
from geoseg.models.FTUNetFormer import SwinTransformer, Decoder


class FTUNetFormerSuperset(nn.Module):
    def __init__(
        self,
        decode_channels=256,
        dropout=0.2,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        freeze_stages=-1,
        window_size=8,
        num_classes=6,
        in_chans=14,  # e.g. 7 spectral + 7 presence planes
    ):
        super().__init__()
        self.backbone = SwinTransformer(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            frozen_stages=freeze_stages,
            in_chans=in_chans,
        )
        encoder_channels = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        self.decoder = Decoder(
            encoder_channels,
            decode_channels,
            dropout,
            window_size,
            num_classes
        )

    def forward(self, x):
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        x = self.decoder(res1, res2, res3, res4, h, w)
        return x


def _inflate_first_conv_rgb_to_superset(weight_3ch: torch.Tensor, in_chans_new: int) -> torch.Tensor:
    """
    weight_3ch: [out_c, 3, kh, kw]
    returns:    [out_c, in_chans_new, kh, kw]
    """
    out_c, old_c, kh, kw = weight_3ch.shape
    assert old_c == 3, f"Expected 3 input channels in pretrained conv, got {old_c}"

    new_w = torch.zeros((out_c, in_chans_new, kh, kw), dtype=weight_3ch.dtype)

    # Copy RGB into the first three channels
    copy_c = min(3, in_chans_new)
    new_w[:, :copy_c] = weight_3ch[:, :copy_c]

    # Fill remaining channels with mean RGB weights
    if in_chans_new > 3:
        mean_rgb = weight_3ch.mean(dim=1, keepdim=True)  # [out_c,1,kh,kw]
        new_w[:, 3:] = mean_rgb.repeat(1, in_chans_new - 3, 1, 1)

    return new_w


def load_pretrained_safely(model, weight_path, in_chans):
    ckpt = torch.load(weight_path, map_location="cpu")
    old_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model_dict = model.state_dict()

    patched = {}
    skipped = []

    for k, v in old_dict.items():
        if k not in model_dict:
            continue

        # Special handling for the first conv / patch embed
        if k == "backbone.patch_embed.proj.weight":
            if v.shape != model_dict[k].shape:
                # Adapt 3ch pretrained weights to new channel count
                if v.ndim == 4 and v.shape[1] == 3:
                    patched[k] = _inflate_first_conv_rgb_to_superset(v, in_chans)
                else:
                    skipped.append((k, tuple(v.shape), tuple(model_dict[k].shape)))
            else:
                patched[k] = v
            continue

        # Normal compatible keys
        if v.shape == model_dict[k].shape:
            patched[k] = v
        else:
            skipped.append((k, tuple(v.shape), tuple(model_dict[k].shape)))

    model_dict.update(patched)
    model.load_state_dict(model_dict, strict=True)

    if skipped:
        print("Skipped incompatible pretrained keys:")
        for s in skipped:
            print("  ", s)

    return model


def ft_unetformer_superset(
    pretrained=True,
    num_classes=6,
    freeze_stages=-1,
    decoder_channels=256,
    in_chans=14,
    weight_path="pretrain_weights/stseg_base.pth",
):
    model = FTUNetFormerSuperset(
        num_classes=num_classes,
        freeze_stages=freeze_stages,
        decode_channels=decoder_channels,
        in_chans=in_chans,
    )

    if pretrained and weight_path is not None:
        model = load_pretrained_safely(model, weight_path, in_chans)

    return model
