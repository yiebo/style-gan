import torch
import torch.nn as nn
from ops import StyleMod, SubPixelConv, Conv2d_AdaIn


class Block(nn.Module):
  def __init__(self, in_channels, out_channels, latent):
    super().__init__()
    self.relu = nn.ReLU()
    
    self.conv0 = SubPixelConv(in_channels, out_channels, scale=2, kernel_size=3, stride=1, padding=1, bias=False)
    self.style_mod0 = StyleMod(out_channels, latent)
    self.instance_norm0 = nn.InstanceNorm2d(out_channels)

    self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    self.style_mod1 = StyleMod(out_channels, latent)
    self.instance_norm1 = nn.InstanceNorm2d(out_channels)

  def forward(self, x, latent):
    x = self.conv0(x)
    x = self.style_mod0(x, latent[:,:,0])
    x = self.instance_norm0(x)
    x = self.relu(x)
    
    x = self.conv1(x)
    x = self.style_mod1(x, latent[:,:,1])
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
    # pixelnorm
    x = x * torch.rsqrt(torch.mean(x.pow(2), dim=1, keepdim=True) + 1e-8)

    x = self.mapping(x)
    x = x.expand(2).expand(3).expand(-1, -1, 5, 2)
    return x

class GeneratorSynth(nn.Module):
  def __init__(self, latent):
    super().__init__()
    self.relu = nn.ReLU()

    self.init_block = nn.Parameter(torch.ones(1, 512, 4, 4))
    self.init_block_bias = nn.Parameter(torch.ones(1, 512, 1, 1))

    self.blocks = nn.ModuleList([
      Block(512, 512, latent),
      Block(512, 512, latent),
      Block(512, 512, latent),
      Block(512, 256, latent),
      Block(256, 128, latent)
    ])
    
    self.to_rgb = nn.ModuleList([
      nn.Conv2d(512, 3, kernel_size=1, stride=1),
      nn.Conv2d(512, 3, kernel_size=1, stride=1),
      nn.Conv2d(512, 3, kernel_size=1, stride=1),
      nn.Conv2d(256, 3, kernel_size=1, stride=1),
      nn.Conv2d(128, 3, kernel_size=1, stride=1)
    ])

  def forward(self, latent, depth, alpha):
    x = self.init_block.expand(latent.shape[0], -1, -1, -1) + self.init_block_bias

    if depth > 0:
      for idx in range(depth - 1):
        x = self.blocks[idx](x, latent[:, :, idx])

      x_ = self.to_rgb[depth - 1](x)
      x_ = nn.functional.interpolate(x_, scale_factor=2, mode='bilinear')

      # added block
      x = self.blocks[depth](x, latent[:, :, depth])
      x = self.to_rgb[depth](x)

      x = alpha * x + (1 - alpha) * x_
    else:
      x = self.blocks[0](x, latent[:, :, 0])
      x = self.to_rgb[0](x)
      
    x = self.relu(x)

    return x

class Generator(nn.Module):
  def __init__(self, latent_in, latent_out):
    super().__init__()
    self.generator_mapping = GeneratorMapping(latent_in, latent_out)
    self.generator_synth = GeneratorSynth(latent_out)

  def forward(self, latent, depth=5, alpha=1):
    # style [B, C, block, layer]
    style = self.generator_mapping(latent)

    if torch.randn(1) < 0.9:
      style_ = torch.randn_like(latent).to('cuda:0')
      style_ = self.generator_mapping(style_)

      layer_idx = torch.arange(depth * 2).view(1, 1, 5, 2)
      cutoff = torch.randint(depth)
      style = torch.where(layer_idx < cutoff, style, style_)

    x = self.generator_synth(style, depth, alpha)
    
    return x
