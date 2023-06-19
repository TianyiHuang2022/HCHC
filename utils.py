from __future__ import division, print_function
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat

def maxmin_norm(Data):
    n_dim = Data.shape[1]
    max_D = np.max(Data, axis=0)
    min_D = np.min(Data, axis=0)
    temp = max_D - min_D
    for i in range(n_dim):
        if temp[i] != 0:
            Data[:, i] = (Data[:, i] - min_D[i]) / temp[i]
    return Data



def load_mnist(path='./data/mnist.npz'):
    f = np.load(path)

    x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f[
        'x_test'], f['y_test']
    f.close()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test)).astype(np.int32)
    x = x.reshape((x.shape[0], -1)).astype(np.float32)
    x = np.divide(x, 255.)
    x = maxmin_norm(x)
    print('MNIST samples', x.shape)
    return x, y



def load_fashion(path='./data/fashion.npz'):
    data = np.load(path)
    x = torch.Tensor(data["arr_0"] / 255)
    y = data["arr_1"]
    return x, y

class MnistDataset(Dataset):

    def __init__(self):
        self.x, self.y = load_mnist()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))


class fashionDataset(Dataset):

    def __init__(self):
        self.x, self.y = load_fashion()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))









