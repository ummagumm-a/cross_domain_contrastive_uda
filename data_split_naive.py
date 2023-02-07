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
    train_dir = dataset_path.rstrip(os.sep) + '_train_naive' + os.sep
    val_dir = dataset_path.rstrip(os.sep) + '_val_naive' + os.sep
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    jpgs = list(Path(dataset_path).rglob('*.jpg')) + list(Path(dataset_path).rglob('*.png'))
    train_jpgs, val_jpgs = train_test_split(jpgs, test_size=0.25, random_state=41)

    for train_jpg in train_jpgs:
        train_jpg_cls_name = os.sep.join(str(train_jpg).split(os.sep)[-2:])
#        print(train_jpg, os.path.join(train_dir, train_jpg_cls_name))
        dest = os.path.join(train_dir, train_jpg_cls_name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(train_jpg, dest)

    for val_jpg in val_jpgs:
        val_jpg_cls_name = os.sep.join(str(val_jpg).split(os.sep)[-2:])
#        print(val_jpg, os.path.join(val_dir, val_jpg_cls_name))
        dest = os.path.join(val_dir, val_jpg_cls_name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(val_jpg, dest)

            
if __name__ == "__main__":
    split_dataset(dataset_path=sys.argv[1])
