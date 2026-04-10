import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.layers import Norm
from monai.networks.nets import UNet

class DentalFusionNetwork(nn.Module):
  def __init__(self, out_shape=(128, 128, 64)):
    super().__init__()
    self.out_shape = out_shape

    # Panorammic Encoder (MONAI UNet)
    self.pan_encoder = UNet(
        spatial_dims = 2,
        in_channels = 1,
        out_channels = 128,
        channels = (32, 64, 128),
        strides = (2, 2),
        num_res_units = 2,
        norm = Norm.BATCH,
    )

    # CBCT Encoder (MONAI UNet)
    self.cbct_encoder = UNet(
        spatial_dims = 3,
        in_channels = 1,
        out_channels = 128,
        channels = (16, 32, 64),
        strides = (2, 2),
        num_res_units = 2,
        norm = Norm.BATCH,
    )

    # Soft Tissue Encoder
    self.soft_encoder = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((8, 8))
    )

    # Fusion Network
    self.fusion = nn.Sequential(
        nn.Linear(128*64 + 128*64 + 64*64, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
    )

    # Reconstruction
    self.reconstruct = nn.Sequential(
        nn.Linear(256, np.prod(out_shape)),
        nn.Sigmoid()
    )

  def forward(self, pan, cbct, soft):
    pan_f = self.pan_encoder(pan)
    pan_f = F.adaptive_avg_pool2d(pan_f, (8, 8)).view(pan.size(0), -1)

    cbct_f = self.cbct_encoder(cbct)
    cbct_f = F.adaptive_avg_pool3d(cbct_f, (4, 4, 4)).view(cbct.size(0), -1)

    soft_f = self.soft_encoder(soft).view(soft.size(0), -1)

    # FUSE
    combined = torch.cat([pan_f, cbct_f, soft_f], dim=1)
    features = self.fusion(combined)

    # Reconstruct
    output = self.reconstruct(features)
    output = output.view(-1, 1, *self.out_shape)

    return output, features

if __name__ == "__main__":
  print("\n" + "=" * 80)
  print("Building MONAI Model")
  print("=" * 80)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = DentalFusionNetwork(out_shape=(128, 128, 64)).to(device)

  total_params = sum(p.numel() for p in model.parameters())
  print(f"Total parameters: {total_params:,}")
  print(f"Model size: {total_params * 4 / (1024**2):.2f} MB")