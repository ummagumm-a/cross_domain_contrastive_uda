from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from pathlib import Path
import torchvision.transforms as T
from random import shuffle
import torch
import logging
from sklearn.model_selection import StratifiedKFold

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

def list_data(images_dir):
    images = list(Path(images_dir).rglob('*.jpg')) + list(Path(images_dir).rglob('*.png'))
    # convert paths to strings
    images = list(map(str, images))
    images = sorted(images)
    labels = list(map(second_last_dir, images))

    return np.array(images), np.array(labels)

def make_office_datasets_kfold(transform, n_splits, random_state=0):
    datasets = [
            os.path.join('datasets', 'OFFICE31', 'amazon'),
            os.path.join('datasets', 'OFFICE31', 'webcam'),
            os.path.join('datasets', 'OFFICE31', 'dslr'),
            ]

    # (images, labels) pairs for each dataset
    data = list(map(list_data, datasets))

    # 'n_splits' train-test splits for each dataset
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    amazon_splits = list(skf.split(*data[0]))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + 1)
    webcam_splits = list(skf.split(*data[1]))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + 2)
    dslr_splits = list(skf.split(*data[2]))

    for i in range(n_splits):
        amazon_train_dataset = UDADataset(
            datasets[0],
            transform=transform,
            indices=amazon_splits[i][0]
        )
        amazon_val_dataset = UDADataset(
            datasets[0],
            transform=transform,
            indices=amazon_splits[i][1]
        )

        webcam_train_dataset = UDADataset(
            datasets[1],
            transform=transform,
            indices=webcam_splits[i][0]
        )
        webcam_val_dataset = UDADataset(
            datasets[1],
            transform=transform,
            indices=webcam_splits[i][1]
        )

        dslr_train_dataset = UDADataset(
            datasets[2],
            transform=transform,
            indices=dslr_splits[i][0]
        )
        dslr_val_dataset = UDADataset(
            datasets[2],
            transform=transform,
            indices=dslr_splits[i][1]
        )

        yield (amazon_train_dataset, amazon_val_dataset), \
              (webcam_train_dataset, webcam_val_dataset), \
              (dslr_train_dataset, dslr_val_dataset)

def get_visda_datasets(transform):
    visda_source_train_dataset = UDADataset(
        os.path.join('datasets', 'visda', 'source_train'),
        transform=transform
    )
    visda_source_val_dataset = UDADataset(
        os.path.join('datasets', 'visda', 'source_val'),
        transform=transform
    )

    visda_target_train_dataset = UDADataset(
        os.path.join('datasets', 'visda', 'target_train'),
        transform=transform
    )
    visda_target_val_dataset = UDADataset(
        os.path.join('datasets', 'visda', 'target_val'),
        transform=transform
    )

    return (visda_source_train_dataset, visda_source_val_dataset), \
           (visda_target_train_dataset, visda_target_val_dataset)


class UDADataset(Dataset):
    def __init__(self, images_dir, transform=None, indices=None):
        # dict with class-number pairs
        label_mapping = office_label_mapping(images_dir)
        # extract class names
        self.class_names = sorted(list(label_mapping.keys()))

        # load image filenames and their classes
        self.images, self.labels = list_data(images_dir)
        if indices is not None:
            self.images, self.labels = self.images[indices], self.labels[indices]

        # Shuffle them
        im_la = list(zip(self.images, self.labels))
        shuffle(im_la)
        self.images, self.labels = zip(*im_la)

        # convert them to numbers
        self.labels = np.array(list(map(lambda x: label_mapping[x], self.labels)))
        
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
    
