from PIL import Image
import csv
import torch
from torch.utils import data


class Dataset(data.Dataset):
  def __init__(self, file_path, root_dir, transform):
    with open(f'{file_path}') as file:
      self.data = list(csv.reader(file, delimiter=" "))[1:]
    self.root_dir = root_dir
    self.transform = transform
    self.labels = [21, 40, 9, 12, 10, 32, 22]
    # ['Male','Young','Black_Hair','Brown_Hair','Blond_Hair','Smiling','Mouth_Slightly_Open']

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    item = self.data[idx]
    y = [int(item[idx]) for idx in self.labels]
    y = torch.FloatTensor(y)

    x = Image.open(f'{self.root_dir}/{item[0]}')
    x = self.transform(x)
    
    return x, y