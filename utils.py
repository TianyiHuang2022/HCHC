from __future__ import division, print_function
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat



def load_fashion(path='./data/fashion.npz'):
    data = np.load(path)
    x = torch.Tensor(data["arr_0"] / 255)
    y = data["arr_1"]
    return x, y



class fashionDataset(Dataset):

    def __init__(self):
        self.x, self.y = load_fashion()
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))





