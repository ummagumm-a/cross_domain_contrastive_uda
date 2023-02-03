from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from pathlib import Path
import torchvision.transforms as T
from random import shuffle
import torch
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def second_last_dir(path):
    return path.split(os.sep)[-2]#os.path.split(os.path.split(path)[0])[1]


def office_label_mapping(datadir):
    """
    Makes a mapping from class name to a number.
    
    :param datadir: directory where the data is stored, e.g. 'datasets/OFFICE31/amazon/'
    """
    labels = sorted(os.listdir(datadir))
    mapping = { label: i for i, label in enumerate(labels) }
    return mapping


class UDADataset(Dataset):
    def __init__(self, images_dir, transform=None):
        # load all images
        images = list(Path(images_dir).rglob('*.jpg')) + list(Path(images_dir).rglob('*.png'))
        # convert paths to strings
        images = list(map(str, images))
        shuffle(images)
#        images = random.shuffle(images)
        # dict with class-number pairs
        label_mapping = office_label_mapping(images_dir)
        # extract class names
        self.class_names = sorted(list(label_mapping.keys()))
        # labels of samples
        labels = list(map(second_last_dir, images))
        # convert them to numbers
        labels = list(map(lambda x: label_mapping[x], labels))
        
        self.pairs = list(zip(images, labels))
        
        self.labels = labels
        self.real_labels = self.labels.copy()
        
        if transform is None:
            logger.info("No transform passed. Use the default transform")
            transform = T.Compose([
                    T.Resize((300, 300)), 
                    T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
            transform = T.Compose([T.ToTensor(), T.Resize((300, 300))])
            
        self.transform = transform
        
    def get_samples(self):
        return list(map(lambda x: x[0], self.pairs))
    
    def get_labels(self):
        return self.labels
    
    def get_real_labels(self):
        return self.real_labels
    
    def get_class_names(self):
        return self.class_names
    
    def update_labels(self, new_labels):
        assert len(new_labels) == len(self.pairs)
        
        for i in range(len(self.pairs)):
            self.pairs[i] = (self.pairs[i][0], new_labels[i])
        self.labels = new_labels
    
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, i):
        pair = self.pairs[i]
        with Image.open(pair[0]) as img:
            # For 1-channel images in visda
            if img.mode == "L":
                img = img.convert(mode='RGB')
            img = self.transform(img)


        return img, torch.tensor(pair[1], dtype=torch.long), \
               torch.tensor(self.real_labels[i], dtype=torch.long)
    
class RemoveMismatchedAdapter(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.matched_pairs_idx = list(range(len(dataset)))
        
    def reset(self):
        self.matched_pairs_idx = list(range(len(self.dataset)))
        
    def get_class_names(self):
        return self.dataset.get_class_names()
      
    def get_labels(self):
        labels = [self.dataset.get_labels()[i] for i in self.matched_pairs_idx]
        return labels
    
    def get_real_labels(self):
        real_labels = [self.dataset.get_real_labels()[i] for i in self.matched_pairs_idx]
        return real_labels
    
    def update_labels(self, new_labels):
        self.dataset.update_labels(new_labels)
        
        self.matched_pairs_idx = []
        for i in range(len(self.dataset)):
            if self.dataset.real_labels[i] == self.dataset.labels[i]:
                self.matched_pairs_idx.append(i)
    
    def __len__(self):
        return len(self.matched_pairs_idx)
    
    def __getitem__(self, i):
        i = self.matched_pairs_idx[i]
        
        return self.dataset[i]
