from pathlib import Path
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil


RANDOM_SEED = 42

def split_office31(data_dir='datasets',
                   office_dir='OFFICE31',
                   test_size=0.25):
    
    office_dir = os.path.join(data_dir, office_dir)
    # Create directories for splits
    office_dir_bigger = os.path.join(data_dir, 'OFFICE31_bigger')
    office_dir_smaller = os.path.join(data_dir, 'OFFICE31_smaller')
    os.makedirs(office_dir_bigger, exist_ok=True)
    os.makedirs(office_dir_smaller, exist_ok=True)

    # List all jpgs in Office 31 dataset
    jpg_paths = list(Path(office_dir).rglob('*.jpg'))
    jpg_paths = list(map(lambda x: str(x).split(os.sep)[2:], jpg_paths))
    jpg_paths = pd.DataFrame(jpg_paths)
    # Domain-class groupings
    domain_classes = jpg_paths.groupby(by=[0, 1]).size()
    
    # To ensure that each domain-class pair is split in an even proportion,
    # i.e. there is always test_size% of samples in val for each class.
    for i in domain_classes.keys():
        subdirs = os.sep.join(i)
        path = os.path.join(office_dir, subdirs)
        dest_train_path = os.path.join(office_dir_bigger, subdirs)
        dest_val_path = os.path.join(office_dir_smaller, subdirs)
        os.makedirs(path, exist_ok=True)
        os.makedirs(dest_train_path, exist_ok=True)
        os.makedirs(dest_val_path, exist_ok=True)

        jpgs = os.listdir(path)
        train_jpgs, val_jpgs = train_test_split(jpgs, test_size=test_size, random_state=RANDOM_SEED)

        for train_jpg in train_jpgs:
            shutil.copy(os.path.join(path, train_jpg), dest_train_path)

        for val_jpg in val_jpgs:
            shutil.copy(os.path.join(path, val_jpg), dest_val_path)

            
if __name__ == "__main__":
    split_office31(sys.argv[1])
