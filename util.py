import torch
import torch.nn as nn
from apex import amp

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
      nn.init.normal_(m.weight.data, 0.0, 0.02)
      
  elif classname.find('BatchNorm') != -1:
      nn.init.normal_(m.weight.data, 1.0, 0.02)
      nn.init.constant_(m.bias.data, 0)


def gradient_penalty_R1(d_output, x, gamma=10):
    d_output = d_output.sum()
    gradients = torch.autograd.grad(d_output, x, create_graph=True)[0]
    gradients = gradients.flatten(1)
    gp = gamma * torch.sum(gradients ** 2, 1).mean()
    return gp
