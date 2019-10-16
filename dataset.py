from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
import numpy as np
import torch
import pickle
from numpy.core import multiarray

class MultiMNIST_Dataset(Dataset):

    def __init__(self, X, y, transform=None):
    
        self.mlb = MultiLabelBinarizer()
        self.transform = transform

        self.X = X
        self.y = self.mlb.fit_transform(y).astype(np.float32)

    def __getitem__(self, index):
        img = self.X[index]
        if self.transform is not None:
            img = self.transform(img)
        
        label = torch.from_numpy(self.y[index])
        return img, label

    def __len__(self):
        return len(self.X)
        


def timestep_Multiminst(address1,address2):
    file1 = open(address1, "rb")
    file2 = open(address2, "rb")
    train = pickle.load(file1,encoding='latin1')
    test=pickle.load(file2,encoding='latin1')
    return np.array((train['imgs'])),np.array((test['imgs'])) #[10 60000 50 50],[10 10000 50 50]
