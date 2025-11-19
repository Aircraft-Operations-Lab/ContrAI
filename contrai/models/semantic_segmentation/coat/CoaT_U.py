# This script is modified from https://github.com/DrHB/2nd-place-contrails

"""
CoaT-U: CoaT Encoder + U-Net Decoder for image segmentation.

This script defines a segmentation model combining CoaT transformer encoders with
FPN and U-Net style upsampling modules. Includes pixel-shuffle ICNR upsampling,
custom LayerNorm for 2D feature maps, and intermediate feature propagation.

Modules
-------
- FPN : Feature Pyramid Network used to combine multi-scale features.
- LayerNorm2d : Custom normalization adapted for 2D CNN activations.
- PixelShuffle_ICNR : Upsampling with ICNR initialization for artifact reduction.
- UnetBlock : U-Net skip connection decoding blocks.
- UpBlock : Final upsampling block to reconstruct segmentation mask.
- CoaT_U : Main segmentation model combining all above components.

Dependencies
------------
- PyTorch
- CoaT backbones defined in `coat.py`
- Model weight management from `weight_loader.py`
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .coat import CoaT, coat_lite_mini, coat_small, coat_lite_medium
from pathlib import Path
from .weight_loader import ensure_local_weight

pretrained_path = ensure_local_weight("pretrained")


class FPN(nn.Module):
    """
    Feature Pyramid Network module.

    Parameters
    ----------
    input_channels : list[int]
        Input channels for each FPN feature level.
    output_channels : list[int]
        Output channels after FPN processing for each level.

    Notes
    -----
    Uses resizing + concatenation to fuse hierarchical features into a single tensor.
    """

    def __init__(self, input_channels: list, output_channels: list):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch * 2, kernel_size=3, padding=1),
                nn.Tanh(),
                LayerNorm2d(out_ch * 2),
                nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1)
            )
            for in_ch, out_ch in zip(input_channels, output_channels)
        ])

    def forward(self, xs: list, last_layer):
        """
        Forward pass through FPN.

        Parameters
        ----------
        xs : list[Tensor]
            List of multi-scale features.
        last_layer : Tensor
            Final decoded feature from U-Net.

        Returns
        -------
        Tensor
            Concatenated multi-scale fused output.
        """
        hcs = [
            F.interpolate(c(x), scale_factor=2 ** (len(self.convs) - i), mode='bilinear')
            for i, (c, x) in enumerate(zip(self.convs, xs))
        ]
        hcs.append(last_layer)
        return torch.cat(hcs, dim=1)


class LayerNorm2d(nn.Module):
    """
    LayerNorm applied across channel dimension for 2D CNN features.

    Parameters
    ----------
    num_channels : int
        Number of feature channels.
    eps : float
        Numerical stability term.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize features channel-wise.

        Returns
        -------
        Tensor
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def icnr_init(x, scale=2, init=nn.init.kaiming_normal_):
    """
    ICNR initialization for PixelShuffle to reduce checkerboard artifacts.

    Reference: "Checkerboard artifact free sub-pixel convolution"
    """
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale ** 2))
    k = init(x.new_zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1).repeat(1, 1, scale ** 2)
    return k.contiguous().view([nf, ni, h, w]).transpose(0, 1)


class PixelShuffle_ICNR(nn.Sequential):
    """
    PixelShuffle upsampling block with ICNR initialization.

    Parameters
    ----------
    ni : int
        Input channels
    nf : int
        Output channels before PixelShuffle
    scale : int
        Upsampling factor
    blur : bool
        Whether to apply blur to suppress artifacts
    """

    def __init__(self, ni, nf=None, scale=2, blur=True):
        super().__init__()
        nf = ni if nf is None else nf
        layers = [
            nn.Conv2d(ni, nf * (scale ** 2), 1),
            LayerNorm2d(nf * (scale ** 2)),
            nn.GELU(),
            nn.PixelShuffle(scale)
        ]

        # ICNR init
        layers[0].weight.data.copy_(icnr_init(layers[0].weight.data))

        if blur:
            layers += [
                nn.ReplicationPad2d((1, 0, 1, 0)),
                nn.AvgPool2d(2, stride=1)
            ]

        super().__init__(*layers)


class UnetBlock(nn.Module):
    """
    U-Net upsampling block with skip connection.

    Parameters
    ----------
    up_in_c : int
        Channels coming from decoder.
    x_in_c : int
        Channels from corresponding encoder feature map.
    nf : int
        Output channels.
    """

    def __init__(self, up_in_c: int, x_in_c: int, nf: int = None, blur: bool = False, **kwargs):
        super().__init__()
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c // 2, blur=blur, **kwargs)
        self.bn = LayerNorm2d(x_in_c)

        ni = up_in_c // 2 + x_in_c
        nf = nf if nf is not None else max(up_in_c // 2, 32)

        self.conv1 = nn.Sequential(nn.Conv2d(ni, nf, 3, padding=1), nn.GELU())
        self.conv2 = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1), nn.GELU())
        self.relu = nn.GELU()

    def forward(self, up_in: torch.Tensor, left_in: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining decoder + skip features.

        Returns
        -------
        Tensor
        """
        up_out = self.shuf(up_in)
        cat_x = self.relu(torch.cat([up_out, self.bn(left_in)], dim=1))
        return self.conv2(self.conv1(cat_x))


class UpBlock(nn.Module):
    """
    Final upsampling block for segmentation output.

    Parameters
    ----------
    up_in_c : int
        Input decoder channels.
    nf : int
        Output channels.
    """

    def __init__(self, up_in_c: int, nf: int = None, blur: bool = True, **kwargs):
        super().__init__()
        ni = up_in_c // 4
        self.shuf = PixelShuffle_ICNR(up_in_c, ni, blur=blur, **kwargs)

        nf = nf if nf is not None else max(up_in_c // 4, 16)
        self.conv = nn.Sequential(
            nn.Conv2d(ni, ni, 3, padding=1),
            LayerNorm2d(ni) if ni >= 16 else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(ni, nf, 1)
        )

    def forward(self, up_in: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.conv(self.shuf(up_in))


class CoaT_U(nn.Module):
    """
    CoaT-U segmentation model.

    Parameters
    ----------
    pre : str or None
        Path to pretrained weights.
    arch : {'mini','small','medium'}
        Selection of CoaT encoder variant.
    num_classes : int
        Number of segmentation classes.
    ps : float
        Dropout probability.

    Returns
    -------
    Tensor
        Segmentation prediction.
    """

    def __init__(self, pre=pretrained_path, arch='small', num_classes=1, ps=0.1, **kwargs):
        super().__init__()

        # Encoder backbone initialization
        if arch == 'mini':
            self.enc = coat_lite_mini(return_interm_layers=True)
            nc = [64, 128, 320, 512]
        elif arch == 'small':
            self.enc = coat_small(return_interm_layers=True)
            nc = [152, 320, 320, 320]
        elif arch == 'medium':
            self.enc = coat_lite_medium(return_interm_layers=True)
            nc = [128, 256, 320, 512]
        else:
            raise ValueError("Unknown model architecture")

        # Optional pretrained weight loading
        if pre is not None:
            sd = torch.load(pre)
            self.enc.load_state_dict(sd, strict=False)

        # Decoder with U-Net blocks
        self.dec4 = UnetBlock(nc[-1], nc[-2], 384)
        self.dec3 = UnetBlock(384, nc[-3], 192)
        self.dec2 = UnetBlock(192, nc[-4], 96)

        self.fpn = FPN([nc[-1], 384, 192], [32] * 3)
        self.drop = nn.Dropout2d(ps)

        self.final_conv = nn.Sequential(
            UpBlock(96 + 32 * 3, num_classes, blur=True)
        )

        self.up_result = 1  # Output upsampling factor

    def forward(self, x):
        """
        Complete forward pass: Encoder → Decoder → FPN → Segmentation

        Parameters
        ----------
        x : Tensor
            Input image (B, C, H, W)

        Returns
        -------
        Tensor
            Output mask prediction
        """
        if len(x.shape) == 5:  # Handle 5D input
            x = x[:, :, 4]

        x = F.interpolate(x, scale_factor=2, mode='bicubic').clip(0, 1)
        encs = self.enc(x)
        encs = [encs[k] for k in encs]  # Convert dict to ordered list

        dec4 = encs[-1]
        dec3 = self.dec4(dec4, encs[-2])
        dec2 = self.dec3(dec3, encs[-3])
        dec1 = self.dec2(dec2, encs[-4])

        x = self.fpn([dec4, dec3, dec2], dec1)
        x = self.final_conv(self.drop(x))

        if self.up_result != 0:
            x = F.interpolate(x, scale_factor=self.up_result, mode='bilinear')

        return x
