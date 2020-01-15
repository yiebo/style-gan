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
  transforms.CenterCrop([178, 178]),
  transforms.Resize([128, 128]),
  transforms.RandomHorizontalFlip(0.5),
  transforms.ToTensor()
])

dataset = Dataset('../DATASETS/celebA/data.txt',
                  '../DATASETS/celebA/img_align_celeba', transform)
# dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0)

writer = tensorboard.SummaryWriter(log_dir='logs')

############

generator = Generator(512, 512).to(device)
discriminator = Discriminator().to(device)

############

g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))
gamma = 0.99 ** (1./1000)
g_scheduler = torch.optim.lr_scheduler.ExponentialLR(g_optimizer, gamma=gamma)
d_scheduler = torch.optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=gamma)

############

softplus = torch.nn.Softplus()
upsample = torch.nn.UpsamplingNearest2d(size=[128, 128])


global_idx = 0
mean_losses = np.zeros(3)
batch_sizes = [128, 64, 32, 16, 8]
# epoch_sizes = [16, 8, 4, 2, 1]
epoch_sizes = [4, 4, 4, 8, 16]
for depth, (batch_size, epoch_size) in enumerate(tqdm(zip(batch_sizes, epoch_sizes), total=len(epoch_sizes))):
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
  data_size = len(dataloader)
  alpha_total = (epoch_size * data_size) // 2
  for epoch in tqdm(range(epoch_size), leave=False):
    # For each batch in the dataloader
    for idx, x in enumerate(tqdm(dataloader, leave=False)):
      alpha = min((epoch * data_size + idx) / alpha_total, 1.0)
      # alpha = 0
      latent_random = torch.randn([batch_size, 512]).to(device)
      x = x.to(device)
      ############
      y, y_old, y_added = generator(latent_random, depth=depth, alpha=alpha)
      d_fake = discriminator(y, depth=depth, alpha=alpha)

      # generator loss
      d_fake = torch.mean(d_fake)
      loss_g = softplus(-d_fake)

      g_optimizer.zero_grad()
      loss_g.backward()
      g_optimizer.step()
      g_scheduler.step()
#################################

      x.requires_grad = True
      x_ = torch.nn.functional.adaptive_avg_pool2d(x, 4 * 2 ** (depth + 1))
      d_real = discriminator(x_, depth=depth, alpha=alpha)
      d_fake = discriminator(y.detach(), depth=depth, alpha=alpha)
      
      # discriminator loss
      d_fake = torch.mean(d_fake)
      d_real = torch.mean(d_real)
      gp = gradient_penalty_R1(d_real, x_)
      
      loss_d = softplus(d_fake) + softplus(-d_real) + gp

      d_optimizer.zero_grad()
      loss_d.backward()
      d_optimizer.step()
      d_scheduler.step()

      mean_losses += [
        loss_d.item(),
        loss_g.item(),
        gp.item()
        ]
      global_idx += 1

      if idx % 100 == 0:
        # if idx == 0:
          # writer.add_graph(generator, latent_random)
          # writer.add_graph(discriminator, x)
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

        mean_losses /= 100
        writer.add_scalar('loss/d', mean_losses[0], global_idx)
        writer.add_scalar('loss/g', mean_losses[1], global_idx)
        writer.add_scalar('loss/gp', mean_losses[2], global_idx)
        writer.add_scalar('misc/alpha', alpha, global_idx)
        mean_losses = np.zeros(3)


        x_ = x_[0:4].clamp(min=0., max=1.)
        x_ = upsample(x_)
        writer.add_images('img_', x_, global_idx)
        
        y = y[0:4].clamp(min=0., max=1.)
        y = upsample(y)
        writer.add_images('img', y, global_idx)
        writer.add_scalar('misc/lr', g_scheduler.get_lr()[0], global_idx)

        if depth > 0:
          y = y_old[0:4].clamp(min=0., max=1.)
          y = y_added[0:4].clamp(min=0., max=1.)
          y_old = upsample(y_old)
          y_addedy = upsample(y_added)
          writer.add_images('img_old', y_old, global_idx)
          writer.add_images('img_added', y_added, global_idx)



        writer.flush()

        del x, y

