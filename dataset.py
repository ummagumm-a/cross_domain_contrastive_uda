from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from pathlib import Path
import torchvision.transforms as T
from random import shuffle

class UDADataset(Dataset):
    def __init__(self, items, labels):
        assert labels is None or len(items) == len(labels)
        
        self.items = items
        self.labels = labels
        
    def update_labels(self, new_labels):
        assert self.labels is None or len(new_labels) == len(self.labels)
        self.labels = new_labels

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, i):
        if self.labels is None:
            return self.items[i], -1
        else:
            return self.items[i], self.labels[i].astype(np.int64)
        
        
        
def second_last_dir(path):
    return os.path.split(os.path.split(path)[0])[1]

def office_label_mapping():
    images_dir = os.path.join(os.getcwd(), 'datasets', 'office31', 'amazon', 'images')
    images = Path(images_dir).rglob('*.jpg')
    images = list(map(str, images))
    labels = list(map(second_last_dir, images))
    mapping = { label: i for i, label in enumerate(np.unique(labels)) }
    return mapping

class OfficeDataset(Dataset):
    def __init__(self, name, transform=None):
        
        images_dir = os.path.join(os.getcwd(), 'datasets', 'office31', name, 'images')
        images = Path(images_dir).rglob('*.jpg')
        images = list(map(str, images))
        label_mapping = office_label_mapping()
        self.class_names = list(label_mapping.keys())
        labels = list(map(second_last_dir, images))
        labels = list(map(lambda x: label_mapping[x], labels))
        self.pairs = list(zip(images, labels))
        shuffle(self.pairs)
        
        if transform is None:
            transform = T.Compose([T.ToTensor(), T.Resize((300, 300))])
            
        self.transform = transform
        
    def get_samples(self):
        return list(map(lambda x: x[0], self.pairs))
    
    def get_labels(self):
        return list(map(lambda x: x[1], self.pairs))
    
    def get_class_names(self):
        return self.class_names
    
    def update_labels(self, new_labels):
        assert len(new_labels) == len(self.pairs)
        
        for i in range(len(self.pairs)):
            self.pairs[i] = (self.pairs[i][0], new_labels[i])
    
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, i):
        pair = self.pairs[i]
        with Image.open(pair[0]) as img:
            img = self.transform(img)
        return img, pair[1]
        