import torch
import os
import numpy as np
import torch.nn as nn
import torchaudio





class DataLoader(torch.utils.data.Dataset):

  def __init__(self, root_dir):
    self.root_dir = root_dir
    self.data_dir = []
    self.get_video_dir()

  def __len__(self):
    return len(self.data_dir)

  def get_video_dir(self):
    for folder in os.listdir(self.root_dir):
      if folder == ".DS_Store":
        continue
      folder_path = os.path.join(self.root_dir, folder)
      try:
        for file in os.listdir(folder_path):
          if file == ".DS_Store":
            continue
          file_path = os.path.join(folder_path, file)
          self.data_dir.append(file_path)
      except Exception as e:
        raise e

  def __getitem__(self, idx):
    file_path = self.data_dir[idx]
    curr_cls = [0,0]
    if "males" in file_path:
      curr_cls = torch.tensor([0.,1.])
    else:
      curr_cls = torch.tensor([1.,0.])
    audio, sr = torchaudio.load(file_path)
    spectrogram = torchaudio.transforms.MelSpectrogram()(audio)
    spectrogram = spectrogram[0,:128,:300]
    return spectrogram, curr_cls
