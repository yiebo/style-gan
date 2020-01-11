import torch.nn as nn


class D_Block(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.layer = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, 3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.LeakyReLU(0.2),
      nn.MaxPool2d(2, 2)
    )
  def forward(self, x):
    x = self.layer(x)
    return x

class Discriminator(nn.Module):
  def __init__(self, in_channels, latent):
    super().__init__()
    layers = []
    for channels in [64, 128, 256, 512, 1028]:
      layers.append(D_Block(in_channels, channels))
      in_channels = channels
    self.layers = nn.Sequential(*layers)

    self.adversarial = nn.Sequential(
      nn.Conv2d(1028, 512, 3, padding=1),
      nn.BatchNorm2d(512),
      nn.LeakyReLU(0.2),
      nn.Conv2d(512, 1, 1)
    )

    self.cls = nn.Conv2d(1028, latent, 1)

  def forward(self, x):
    x = self.layers(x)
    x_adv = self.adversarial(x)
    x_cls = self.cls(x)
    return x_adv, x_cls
