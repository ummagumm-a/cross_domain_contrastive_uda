from pathlib import Path
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil


RANDOM_SEED = 42

def split_dataset(dataset_path: str,
                  test_size: float = 0.25) -> None:

    """
    Splits dataset in Office-31 format into train and test.
    Does it by splitting each class separately, 
    ensuring that each class is proportionally represented in train and in test.
    Assumes that all images are either jpg or png.

    :param dataset_path: path to dataset; should be relative to 'data_dir'. Example: OFFICE31/amazon/
    :param test_size: fraction of the dataset which will go into test.
    """
    
    # Create directories for splits
    train_dir = dataset_path.rstrip(os.sep) + '_train' + os.sep
    val_dir = dataset_path.rstrip(os.sep) + '_val' + os.sep
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    contents = os.listdir(dataset_path)
    contents = list(map(lambda x: os.path.join(dataset_path, x), contents))
    classes = list(filter(os.path.isdir, contents))

    # To ensure that each domain-class pair is split in an even proportion,
    # i.e. there is always test_size% of samples in val for each class.
    for class_dir in classes:

        jpgs = os.listdir(class_dir)
        train_jpgs, val_jpgs = train_test_split(jpgs, test_size=0.25, random_state=41)

        train_class_dir = os.path.join(train_dir, os.path.basename(class_dir))
        os.makedirs(train_class_dir, exist_ok=True)
        for train_jpg in train_jpgs:
            shutil.copy(os.path.join(class_dir, train_jpg),
                        train_class_dir)

        val_class_dir = os.path.join(val_dir, os.path.basename(class_dir))
        os.makedirs(val_class_dir, exist_ok=True)
        for val_jpg in val_jpgs:
            shutil.copy(os.path.join(class_dir, val_jpg), 
                        val_class_dir)

            
if __name__ == "__main__":
    split_dataset(dataset_path=sys.argv[1])
