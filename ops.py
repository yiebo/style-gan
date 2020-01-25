import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleMod(nn.Module):
  def __init__(self, channels, latent):
    super().__init__()
    self.style_scale = LinearEqualized(latent, channels, gain=1.0)
    self.style_bias = LinearEqualized(latent, channels, gain=1.0)
    torch.nn.init.ones_(self.style_scale.bias)

  def forward(self, x, latent):
    style_scale = self.style_scale(latent).unsqueeze(2).unsqueeze(3)
    style_bias = self.style_bias(latent).unsqueeze(2).unsqueeze(3)
    x = x * style_scale + style_bias
    return x

class Conv2d_AdaIn(nn.Conv2d):
  def __init__(self, latent_channels, in_channels, out_channels, kernel_size, stride=1,
               padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
    super(Conv2d_AdaIn, self).__init__(in_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias, padding_mode)
    self.style_scale = nn.Linear(latent_channels, out_channels)

  def forward(self, x, latent):
    batch_size = x.shape[0]
    # modulate
    # weight = self.weight  #[1,O,I,K,K]

    # weight *= self.style_scale(latent).unsqueeze(2).unsqueeze(3).unsqueeze(4) #[B,O,I,K,K]
    style = self.style_scale(latent).unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [B,O,I,K,K]

    weight = (self.weight + 1.) * style

    # demod
    style_std = torch.rsqrt(torch.sum(weight.pow(2), dim=[2, 3, 4], keepdim=True) + 1e-8)  # [B,O,1,1,1]
    weight = weight * style_std  # [B,O,I,K,K]

    weight = weight.view(-1, weight.shape[2], weight.shape[3], weight.shape[4])  # [BO,I,K,K]
    x = x.view(1, -1, x.shape[2], x.shape[3])  # [1,BC,H,W]

    x = F.conv2d(x, weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=batch_size)
    # print(x.shape)
    x = x.view(-1, self.weight.shape[0], x.shape[2], x.shape[3])

    return x + self.bias.unsqueeze(1).unsqueeze(2)


class SubPixelConv(nn.Module):
  def __init__(self, in_channels, out_channels, scale=2, kernel_size=3, stride=1,
               padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros'):
    # self, in_channels, out_channels, scale, kernel_size=3, stride=1, padding=1):
    super().__init__()
    out_channels = out_channels * scale ** 2
    self.sub_pixel_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                        padding, dilation, groups, bias, padding_mode),
                                        nn.PixelShuffle(scale))

  def forward(self, x):
    x = self.sub_pixel_conv(x)
    return x


class Conv2dEqualized(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
               padding=0, bias=True, gain=2 ** .5):
    super().__init__()
    self.w_scale = gain / np.sqrt(in_channels * kernel_size ** 2)
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

    if bias:
      self.bias = torch.nn.Parameter(torch.zeros(out_channels))
    else:
      self.bias = None

  def forward(self, x):
    weight = self.w_scale * self.weight
    bias = self.bias
    if bias is not None:
      bias = self.w_scale * bias
    return F.conv2d(x, weight, bias=bias, stride=self.stride, padding=self.padding)

class LinearEqualized(nn.Module):
  def __init__(self, in_channels, out_channels, bias=True, gain=2 ** .5):
    super().__init__()

    self.w_scale = gain / np.sqrt(in_channels)
    self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels))

    if bias:
      self.bias = torch.nn.Parameter(torch.zeros(out_channels))
    else:
      self.bias = None

  def forward(self, x):
    weight = self.w_scale * self.weight
    bias = self.bias
    if bias is not None:
      bias = self.w_scale * bias
    return F.linear(x, weight, bias)
