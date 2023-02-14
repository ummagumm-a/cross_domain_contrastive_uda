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
        self.images = list(map(str, images))
        shuffle(self.images)
        # dict with class-number pairs
        label_mapping = office_label_mapping(images_dir)
        # extract class names
        self.class_names = sorted(list(label_mapping.keys()))
        # labels of samples
        labels = list(map(second_last_dir, self.images))
        # convert them to numbers
        self.labels = np.array(list(map(lambda x: label_mapping[x], labels)))
        
        self.real_labels = self.labels.copy()
        
        if transform is None:
            logger.info("No transform passed. Use the default transform")
            transform = T.Compose([
                    T.Resize((300, 300)), 
                    T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
            
        self.transform = transform
        
    def get_labels(self):
        return self.labels
    
    def get_real_labels(self):
        return self.real_labels
    
    def get_class_names(self):
        return self.class_names
   
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, i):
        img, label, real_label = self.images[i], self.labels[i], self.real_labels[i]
        with Image.open(img) as img:
            # For 1-channel images in visda
            if img.mode == "L":
                img = img.convert(mode='RGB')
            img = self.transform(img)


        return img, torch.tensor(label, dtype=torch.long), \
               torch.tensor(real_label, dtype=torch.long)
    
