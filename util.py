import torch
import torch.nn as nn

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
      nn.init.normal_(m.weight.data, 0.0, 0.02)
      
  elif classname.find('BatchNorm') != -1:
      nn.init.normal_(m.weight.data, 1.0, 0.02)
      nn.init.constant_(m.bias.data, 0)


def gradient_penalty_R1(x, disc, **kwargs):
    x = torch.autograd.Variable(x, requires_grad=True)
    d_output = disc(x, **kwargs)
    gradients = torch.autograd.grad(d_output, x, grad_outputs=torch.ones_like(d_output).to(x.device),
                                    create_graph=True)[0]
    gradients = gradients.flatten(1)
    gp = 10 * torch.sum(gradients ** 2, 1)
    return gp
