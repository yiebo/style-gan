import torch.nn as nn


class Block(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(0.2),
      nn.Conv2d(in_channels, out_channels, 3, padding=1),
      nn.LeakyReLU(0.2),
      nn.AvgPool2d(2, 2)
    )
  def forward(self, x):
    x = self.layers(x)
    return x

class FinalBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    
    self.lrelu = nn.LeakyReLU(0.2)

    self.conv = nn.Conv2d(512, 512, 3, padding=1)

    self.dense = nn.Sequential(
      nn.Linear(4 * 4 * 512, 512),
      nn.LeakyReLU(0.2),
      nn.Linear(512, 1)
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
      Block(512, 512),
      Block(512, 512),
      Block(512, 512),
      Block(256, 512),
      Block(128, 256),
    ])
    
    self.from_rgb = nn.ModuleList([
      nn.Conv2d(3, 512, kernel_size=1, stride=1),
      nn.Conv2d(3, 512, kernel_size=1, stride=1),
      nn.Conv2d(3, 512, kernel_size=1, stride=1),
      nn.Conv2d(3, 256, kernel_size=1, stride=1),
      nn.Conv2d(3, 128, kernel_size=1, stride=1)
    ])

  def forward(self, x, depth, alpha):
    if depth > 0:
      # added block
      x_ = self.from_rgb[depth](x)
      x_ = self.blocks[depth](x_)

      x = self.from_rgb[depth - 1](x)
      x = alpha * x_ + (1 - alpha) * x
      
      for block in self.blocks[depth-1::-1]:
        x = block(x)

    else:
      x = self.from_rgb[0](x)
      x = self.blocks[0](x)
    
    return x
