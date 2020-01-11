import torch
import torch.nn as nn
from ops import StyleMod, SubPixelConv, Conv2d_AdaIn


class U_BlockLatent(nn.Module):
  def __init__(self, in_channels, out_channels, latent):
    super().__init__()
    self.relu = nn.ReLU()
    self.conv0 = Conv2d_AdaIn(latent, in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    self.conv1 = Conv2d_AdaIn(latent, out_channels, out_channels, kernel_size=3, stride=1, padding=1)

  def forward(self, x, latent):
    x = self.conv0(x, latent)
    x = self.relu(x)
    
    x = self.conv1(x, latent)
    x = self.relu(x)
    return x


class Block(nn.Module):
  def __init__(self, in_channels, out_channels, latent):
    super().__init__()
    self.relu = nn.ReLU()
    
    self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    self.style_mod0 = StyleMod(out_channels, latent)
    self.instance_norm0 = nn.InstanceNorm2d(out_channels)

    self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    self.style_mod1 = StyleMod(out_channels, latent)
    self.instance_norm1 = nn.InstanceNorm2d(out_channels)

  def forward(self, x, latent):
    x = self.conv0(x)
    x = self.style_mod0(x, latent)
    x = self.instance_norm0(x)
    x = self.relu(x)
    
    x = self.conv1(x)
    x = self.style_mod1(x, latent)
    x = self.instance_norm1(x)
    x = self.relu(x)
    return x


class GeneratorMapping(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    layers = [
      nn.Linear(in_channels, out_channels),
      nn.LeakyReLU(0.2)
      ]
    for _ in range(4):
      layers.append(nn.Linear(out_channels, out_channels))
      layers.append(nn.LeakyReLU(0.2))
  
    self.mapping = nn.Sequential(*layers)

  def forward(self, x):
    return self.mapping(x)

class GeneratorSynth(nn.Module):
  def __init__(self, in_channels, out_channels, latent):
    super().__init__()
    # self.init_layer = nn.Sequential(
    #   nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
    #   nn.InstanceNorm2d(out_channels),
    #   nn.ReLU()
    # )

    self.init_block = None
    self.blocks = nn.ModuleList(
      [
        Block(512, 512, latent),
        Block(512, 512, latent),
        Block(512, 512, latent),
        Block(512, 256, latent),
        Block(256, 128, latent)
      ]
    )
    
    self.to_rgb = nn.Sequential(
      nn.Conv2d(512, 3, kernel_size=1, stride=1),
      nn.Conv2d(512, 3, kernel_size=1, stride=1),
      nn.Conv2d(512, 3, kernel_size=1, stride=1),
      nn.Conv2d(256, 3, kernel_size=1, stride=1),
      nn.Conv2d(128, 3, kernel_size=1, stride=1)
    )

  def forward(self, x, latent, depth, alpha):
    
    if depth > 0:
      for idx in range(depth-1):
        x = self.blocks[idx](x, latent[idx])

      x_ = self.to_rgb[depth-1](x)
      x_ = nn.functional.interpolate(x_, scale_factor=2, mode='bilinear')

      x = self.blocks[depth](x, latent[depth])
      x = self.to_rgb[depth](x)

      x = alpha * x + (1 - alpha) * x_
    else:
      x = self.blocks[depth](x, latent[depth])
      x = self.to_rgb[depth](x)

    return x

class Generator(nn.Module):
  def __init__(self, in_channels, out_channels, latent_in, latent_out):
    super().__init__()
    self.generator_mapping = GeneratorMapping(latent_in, latent_out)
    self.generator_synth = GeneratorSynth(in_channels, out_channels, latent_out)

  def forward(self, x, latent):
    style = self.generator_mapping(latent)
    x = self.u_net(x, style)
    return x
