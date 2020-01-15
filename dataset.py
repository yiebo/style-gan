from PIL import Image
import csv
import torch
from torch.utils import data


class Dataset(data.Dataset):
  def __init__(self, file_path, root_dir, transform):
    with open(file_path) as file:
      self.data = list(csv.reader(file, delimiter=" "))[1:]
    self.root_dir = root_dir
    self.transform = transform

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    item = self.data[idx]

    x = Image.open(f'{self.root_dir}/{item[0]}')
    x = self.transform(x)
    
    return x