from tqdm import tqdm
import glob
import os
import numpy as np
from prefetch_generator import BackgroundGenerator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
logs_idx = len(glob.glob('logs/*'))
writer = tensorboard.SummaryWriter(log_dir=f'logs/{logs_idx}')


generator = Generator(512, 512).to(device)
discriminator = Discriminator().to(device)


g_optimizer = torch.optim.Adam([{'params': generator.generator_mapping.parameters(), 'lr': 0.001 * 0.01},
                                {'params': generator.generator_synth.parameters()}], 
                                lr=0.001, betas=(0., 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001, betas=(0., 0.99))

############

global_idx = 0
summ_counter = 0
mean_losses = np.zeros(5)
batch_sizes = [256, 128, 64, 32, 16, 8]
epoch_sizes = [2, 4, 4, 8, 8, 16]
latent_const = torch.from_numpy(np.load('randn.npy')).float().to(device)

for depth, (batch_size, epoch_size) in enumerate(tqdm(zip(batch_sizes, epoch_sizes), total=len(epoch_sizes))):
  
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
  data_size = len(dataloader)
  alpha_total = (epoch_size * data_size) // 2

  # prefetch
  dataloader = BackgroundGenerator(dataloader)

  for epoch in tqdm(range(epoch_size), leave=False):
    # For each batch in the dataloader
    for idx, x in enumerate(tqdm(dataloader, leave=False)):
      alpha = min((epoch * data_size + idx) / alpha_total, 1.0)
      latent_random = torch.randn([batch_size, 512]).to(device)
      x = x.to(device)
      #################################
      y = generator(latent_random, depth=depth, alpha=alpha)
      d_fake = discriminator(y, depth=depth, alpha=alpha)

      loss_g = F.softplus(input, beta=1, threshold=20)(-d_fake).mean()

      g_optimizer.zero_grad()
      loss_g.backward()
      g_optimizer.step()
      #################################

      x_ = torch.nn.functional.adaptive_avg_pool2d(x, 4 * 2 ** depth)
      d_real = discriminator(x_, depth=depth, alpha=alpha)
      d_fake = discriminator(y.detach(), depth=depth, alpha=alpha)
      
      # discriminator loss
      d_fake = F.softplus(d_fake).mean()
      d_real = F.softplus(-d_real).mean()
      gp = gradient_penalty_R1(x_, disc=discriminator, depth=depth, alpha=alpha).mean()
      
      loss_d = d_fake + d_real + gp

      d_optimizer.zero_grad()
      loss_d.backward()
      d_optimizer.step()

      mean_losses += [
        loss_d.item(),
        d_real.item(),
        d_fake.item(),
        loss_g.item(),
        gp.item()
        ]
      global_idx += 1
      summ_counter += 1

      if global_idx % 10 == 0 or idx == 0:
        if idx == 0 and epoch == 0:
          writer.add_graph(generator, [latent_const, depth, alpha])
          writer.add_graph(discriminator,  [x, depth, alpha])
        if idx == data_size - 1:
          saves = glob.glob(f'logs/{logs_idx}/*.pt')
          if len(saves) == 10:
            saves.sort(key=os.path.getmtime)
            os.remove(saves[0])
          if epoch - 1 == epoch_size:
            depth += 1
            epoch = -1
          torch.save({
            'depth': depth,
            'epoch': epoch + 1,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
            }, f'logs/{logs_idx}/model_{depth}_{epoch}.pt')

        mean_losses /= summ_counter
        writer.add_scalar('loss/d', mean_losses[0], global_idx)
        writer.add_scalar('loss/d/real', mean_losses[1], global_idx)
        writer.add_scalar('loss/d/fake', mean_losses[2], global_idx)
        writer.add_scalar('loss/g', mean_losses[3], global_idx)
        writer.add_scalar('loss/gp', mean_losses[4], global_idx)
        writer.add_scalar('misc/alpha', alpha, global_idx)
        mean_losses = np.zeros(5)
        summ_counter = 0

        x_ = x_[:8].clamp(min=0., max=1.)
        writer.add_images('img_', x_, global_idx)
        
        y = y[:8].clamp(min=0., max=1.)
        writer.add_images('img/random', y, global_idx)

        with torch.no_grad():
          y = generator(latent_const, depth=depth, alpha=alpha, mix=False)
        y = y.clamp(min=0., max=1.)
        writer.add_images('img/const', y, global_idx)

        writer.flush()
    

