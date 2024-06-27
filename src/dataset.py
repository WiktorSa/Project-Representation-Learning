import torch
from torch.utils.data import Dataset

class UnlabelledDataset(Dataset):
    def __init__(self, images):
        self.images = torch.from_numpy(images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]
    
class LabelledDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]