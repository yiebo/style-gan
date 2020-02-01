import torch
import torch.nn as nn
from ops import StyleMod, Conv2dEqualized, LinearEqualized, Noise


class Block(nn.Module):
  def __init__(self, in_channels, out_channels, latent, upscale=True):
    super().__init__()
    self.lrelu = nn.LeakyReLU(0.2)
    if upscale:
      self.conv0 = nn.Sequential(
          Conv2dEqualized(in_channels, out_channels * 2 ** 2, kernel_size=3, stride=1, padding=1, bias=False),
          nn.PixelShuffle(2))
    else:
      self.conv0 = Conv2dEqualized(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    self.noise0 = Noise(out_channels)
    self.style_mod0 = StyleMod(out_channels, latent)
    self.instance_norm0 = nn.InstanceNorm2d(out_channels)

    self.conv1 = Conv2dEqualized(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    self.noise1 = Noise(out_channels)
    self.style_mod1 = StyleMod(out_channels, latent)
    self.instance_norm1 = nn.InstanceNorm2d(out_channels)

  def forward(self, x, latent):
    x = self.conv0(x)
    x = self.noise0(x)
    x = self.lrelu(x)
    x = self.instance_norm0(x)
    x = self.style_mod0(x, latent[:, :, 0])

    x = self.conv1(x)
    x = self.noise1(x)
    x = self.lrelu(x)
    x = self.instance_norm1(x)
    x = self.style_mod1(x, latent[:, :, 1])
    return x

class BlockInit(nn.Module):
  def __init__(self, in_channels, out_channels, latent):
    super().__init__()
    self.lrelu = nn.LeakyReLU(0.2)

    self.noise0 = Noise(in_channels)
    self.style_mod0 = StyleMod(in_channels, latent)
    self.instance_norm0 = nn.InstanceNorm2d(in_channels)

    self.conv1 = Conv2dEqualized(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    self.noise1 = Noise(out_channels)
    self.style_mod1 = StyleMod(out_channels, latent)
    self.instance_norm1 = nn.InstanceNorm2d(out_channels)

  def forward(self, x, latent):
    x = self.noise0(x)
    x = self.lrelu(x)
    x = self.instance_norm0(x)
    x = self.style_mod0(x, latent[:, :, 0])

    x = self.conv1(x)
    x = self.noise1(x)
    x = self.lrelu(x)
    x = self.instance_norm1(x)
    x = self.style_mod1(x, latent[:, :, 1])
    return x

class GeneratorMapping(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    layers = [
        LinearEqualized(in_channels, out_channels),
        nn.LeakyReLU(0.2)
    ]
    for _ in range(3):
      layers.append(LinearEqualized(out_channels, out_channels))
      layers.append(nn.LeakyReLU(0.2))

    self.mapping = nn.Sequential(*layers)

  def forward(self, x):
    # pixelnorm
    x = x * torch.rsqrt(torch.mean(x.pow(2), dim=1, keepdim=True) + 1e-8)

    x = self.mapping(x)
    x = x.unsqueeze(2).unsqueeze(3).expand(-1, -1, 5, 2)
    return x


class GeneratorSynth(nn.Module):
  def __init__(self, latent):
    super().__init__()

    self.init_block = nn.Parameter(torch.ones(1, 512, 4, 4))
    self.init_block_bias = nn.Parameter(torch.ones(1, 512, 1, 1))

    self.blocks = nn.ModuleList([
        BlockInit(512, 512, latent),
        Block(512, 512, latent),
        Block(512, 512, latent),
        Block(512, 512, latent),
        Block(512, 256, latent),
        Block(256, 128, latent)
    ])

    self.to_rgb = nn.ModuleList([
        Conv2dEqualized(512, 3, kernel_size=1, stride=1),
        Conv2dEqualized(512, 3, kernel_size=1, stride=1),
        Conv2dEqualized(512, 3, kernel_size=1, stride=1),
        Conv2dEqualized(512, 3, kernel_size=1, stride=1),
        Conv2dEqualized(256, 3, kernel_size=1, stride=1),
        Conv2dEqualized(128, 3, kernel_size=1, stride=1)
    ])

    # self.upsample = torch.nn.Upsample(scale_factor=2)

  def forward(self, latent, depth, alpha):
    x = self.init_block.expand(latent.shape[0], -1, -1, -1) + self.init_block_bias

    if depth > 0 or alpha < 1.0:
      for idx in range(depth):
        x = self.blocks[idx](x, latent[:, :, idx])

      x_ = self.to_rgb[depth - 1](x)
      x_ = nn.functional.interpolate(x_, scale_factor=2, mode='nearest')

      # added block
      x = self.blocks[depth](x, latent[:, :, depth])
      x = self.to_rgb[depth](x)

      x = alpha * x + (1.0 - alpha) * x_

    else:
      for idx in range(depth + 1):
        x = self.blocks[idx](x, latent[:, :, idx])
      x = self.to_rgb[depth](x)

    return x


class Generator(nn.Module):
  def __init__(self, latent_in, latent_out):
    super().__init__()
    self.generator_mapping = GeneratorMapping(latent_in, latent_out)
    self.generator_synth = GeneratorSynth(latent_out)

  def forward(self, latent, depth, alpha, mix=True):
    # style [B, C, block, layer]
    style = self.generator_mapping(latent)

    if torch.rand(1) < 0.9 and mix:
      style_ = torch.randn_like(latent).to(latent.device)
      style_ = self.generator_mapping(style_)

      layer_idx = torch.arange(5 * 2).view(1, 1, 5, 2).to(latent.device)
      cutoff = torch.randint((depth + 1) * 2, [1]).to(latent.device)
      style = torch.where(layer_idx < cutoff, style, style_)

    x = self.generator_synth(style, depth, alpha)

    return x
