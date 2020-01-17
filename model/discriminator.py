import torch.nn as nn
from ops import Conv2dEqualized, LinearEqualized


class Block(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.layers = nn.Sequential(
      Conv2dEqualized(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(0.2),
      Conv2dEqualized(in_channels, out_channels, 3, padding=1),
      nn.AvgPool2d(2, 2),
      nn.LeakyReLU(0.2)
    )
  def forward(self, x):
    x = self.layers(x)
    return x


class FinalBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    
    self.lrelu = nn.LeakyReLU(0.2)

    self.conv = nn.Conv2d(in_channels, 512, 3, padding=1)

    self.dense = nn.Sequential(
      LinearEqualized(4 * 4 * 512, 512),
      nn.LeakyReLU(0.2),
      LinearEqualized(512, out_channels)
    )
  def forward(self, x):
    x = self.conv(x)
    x = self.lrelu(x)
    x = x.view(x.shape[0], -1)
    x = self.dense(x)
    return x


class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()

    self.blocks = nn.ModuleList([
      FinalBlock(512, 1),
      Block(512, 512),
      Block(512, 512),
      Block(512, 512),
      Block(256, 512),
      Block(128, 256),
    ])
    
    self.from_rgb = nn.ModuleList([
      Conv2dEqualized(3, 512, kernel_size=1, stride=1),
      Conv2dEqualized(3, 512, kernel_size=1, stride=1),
      Conv2dEqualized(3, 512, kernel_size=1, stride=1),
      Conv2dEqualized(3, 512, kernel_size=1, stride=1),
      Conv2dEqualized(3, 256, kernel_size=1, stride=1),
      Conv2dEqualized(3, 128, kernel_size=1, stride=1)
    ])

  def forward(self, x, depth, alpha):
    if depth > 0:
      # added block
      x_ = self.from_rgb[depth](x)
      x_ = self.blocks[depth](x_)

      x = nn.functional.avg_pool2d(x, 2)
      x = self.from_rgb[depth - 1](x)

      x = alpha * x_ + (1.0 - alpha) * x
      
      for block in self.blocks[depth-1::-1]:
        x = block(x)

    else:
      x = self.from_rgb[0](x)
      x = self.blocks[0](x)
    
    return x
