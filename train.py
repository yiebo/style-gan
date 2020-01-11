from tqdm import tqdm
import glob
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torch.utils import tensorboard
from model.discriminator import Discriminator
from model.generator import Generator
from dataset import Dataset
from util import gradient_penalty_R1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
  transforms.RandomAffine(degrees=0, translate=[.05, .05], scale=[.97, 1.03]),
  transforms.CenterCrop([178, 178]),
  transforms.Resize([208, 208]),
  transforms.RandomHorizontalFlip(0.5),
  transforms.ToTensor()
])

dataset = Dataset('../DATASETS/celebA/data.txt',
                  '../DATASETS/celebA/img_align_celeba', transform)
dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0)

writer = tensorboard.SummaryWriter(log_dir='logs')

############

class_shape = len(dataset.labels)
latent_shape = class_shape + 128

generator = Generator(3, 3, latent_shape, 256).to(device)
discriminator = Discriminator(3, class_shape).to(device)

############

g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))
gamma = 0.99 ** (1./1000)
g_scheduler = torch.optim.lr_scheduler.ExponentialLR(g_optimizer, gamma=gamma)
d_scheduler = torch.optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=gamma)

############

bce_loss = torch.nn.BCEWithLogitsLoss()
l1_loss = torch.nn.L1Loss()
softplus = torch.nn.Softplus()
upsample = torch.nn.UpsamplingNearest2d(size=[20, 208])


global_idx = 0
mean_losses = np.zeros(8)
for epoch in range(10):
    # For each batch in the dataloader
    for idx, (x, cls_real) in enumerate(tqdm(dataloader)):
      # flip for now, shuffle(?) for larger batches
      cls_real_in = cls_real + 0.01 * torch.randn_like(cls_real)
      cls_fake = cls_real.flip(0)
      cls_fake_in = cls_fake + 0.01 * torch.randn_like(cls_real)

      cls_real_d = cls_real_in - cls_fake_in
      cls_fake_d = cls_fake_in - cls_real_in

      latent_random = torch.randn([cls_real.size()[0], 128])
      latent_real_d = torch.cat([cls_real_d, latent_random], 1)
      latent_fake_d = torch.cat([cls_fake_d, -latent_random], 1)

      ############
      x = x.to(device)
      latent_fake_d = latent_fake_d.to(device)
      latent_real_d = latent_real_d.to(device)
      cls_real_in = cls_real_in.to(device)
      cls_fake_in = cls_fake_in.to(device)
      ############
      y = generator(x, latent_fake_d)
      x_ = generator(y, latent_real_d)
      d_fake, cls_fake_ = discriminator(y)
      cls_fake_ = torch.mean(cls_fake_, dim=[2, 3])
      loss_g_cls = bce_loss(cls_fake_, cls_fake_in)

      # generator loss
      loss_cycle = 10 * l1_loss(x_, x)
      d_fake = torch.mean(d_fake)
      loss_g_x = softplus(-d_fake)
      loss_g = loss_g_x + loss_g_cls + loss_cycle

      g_optimizer.zero_grad()
      loss_g.backward()
      g_optimizer.step()
      g_scheduler.step()
#################################

      x.requires_grad = True
      d_real, cls_real_ = discriminator(x)
      d_fake, cls_fake_ = discriminator(y.detach())

      d_real = torch.mean(d_real)
      d_fake = torch.mean(d_fake)

      cls_real_ = torch.mean(cls_real_, dim=[2, 3])


      # class losses
      loss_d_cls = bce_loss(cls_real_, cls_real_in)

      # discriminator loss
      gp = gradient_penalty_R1(d_real, x)
      
      loss_d_x = softplus(d_fake) + softplus(-d_real)
      loss_d = loss_d_x + loss_d_cls + gp

      d_optimizer.zero_grad()
      loss_d.backward()
      d_optimizer.step()
      d_scheduler.step()

      mean_losses += [
        loss_d_cls.item(),
        loss_g_cls.item(), 
        loss_d_x.item(),
        loss_g_x.item(),
        loss_d.item(),
        loss_g.item(),
        loss_cycle.item(),
        gp.item()
        ]
      global_idx += 1

      if idx % 100 == 0:
        if idx == 0:
          writer.add_graph(generator, (x, latent_fake_d))
          writer.add_graph(discriminator, x)
        if idx % 2000 == 0:
          saves = glob.glob('logs/*.pt')
          if len(saves) == 10:
            saves.sort(key=os.path.getmtime)
            os.remove(saves[0])
          
          torch.save({
            'global_idx': global_idx,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
            }, f'logs/model_{global_idx}.pt')

        mean_losses /= 50
        writer.add_scalar('cls/d', mean_losses[0], global_idx)
        writer.add_scalar('cls/g', mean_losses[1], global_idx)
        writer.add_scalar('loss/d', mean_losses[2], global_idx)
        writer.add_scalar('loss/g', mean_losses[3], global_idx)
        writer.add_scalar('loss/total/d', mean_losses[4], global_idx)
        writer.add_scalar('loss/total/g', mean_losses[5], global_idx)
        writer.add_scalar('loss/cycle_loss', mean_losses[6], global_idx)
        writer.add_scalar('loss/gp', mean_losses[7], global_idx)
        mean_losses = np.zeros(8)


        x = x[0:2].clamp(min=0., max=1.)
        y = y[0:2].clamp(min=0., max=1.)
        x_ = x_[0:2].clamp(min=0., max=1.)
        cls_real = upsample(cls_real[0:2].unsqueeze(1).unsqueeze(1)).repeat_interleave(3, 1).to(device)
        cls_fake = upsample(cls_fake[0:2].unsqueeze(1).unsqueeze(1)).repeat_interleave(3, 1).to(device)

        x = torch.cat([x, cls_real], 2)
        y = torch.cat([y, cls_fake], 2)
        x = torch.cat([x, y, x_], 2)

        writer.add_images('img', x, global_idx)
        writer.add_scalar('misc/lr', g_scheduler.get_lr()[0], global_idx)
        writer.flush()

        del x, y, x_

