import torchvision.transforms as T
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from resnetdsbn import resnet50dsbn
from utils import FeatureNormL2
from dataset import UDADataset, RemoveMismatchedAdapter
from model import UDAModel
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor, StochasticWeightAveraging
import torch
import logging
from torchvision.models import ResNet50_Weights
import os

def baseline_setting():
    return dict(
        pretrain_num_epochs = 0,
        explicit_negative_sampling_threshold = 0.0,
        negative_sampling = None,
        tau = 1.0,
        )

def no_adaptation_setting():
    return dict(
        pretrain_num_epochs = 10000,
        explicit_negative_sampling_threshold = 0.0,
        negative_sampling = None,
        tau = 1.0,
        )

def negative_sampling_setting():
    return dict(
        pretrain_num_epochs = 0,
        explicit_negative_sampling_threshold = 0.5,
        negative_sampling = 'hard',
        tau = 1.0,
        )

def pretrain_on_source_setting():
    return dict(
        pretrain_num_epochs = 100,
        explicit_negative_sampling_threshold = 0.0,
        negative_sampling = None,
        tau = 1.0,
        )

def train(source_dataset, target_dataset, num_classes, settings, version, total_epochs=500, remove_mismatched=False):
    # Define the backbone
    resnet = resnet50dsbn(pretrained=True, in_features=256)
    resnet.fc2 = FeatureNormL2()
    resnet.out_features = 256
    classification_head = nn.Linear(resnet.out_features, num_classes, bias=False)

    if remove_mismatched:
        target_dataset = tuple(map(RemoveMismatchedAdapter, target_dataset))

    model = UDAModel(resnet, classification_head, num_classes, 
                     source_dataset, target_dataset, 
                     total_epochs=total_epochs, batch_size=64,
                     num_workers=1, **settings,
                     remove_mismatched=remove_mismatched,
                     class_names=source_dataset[0].get_class_names())

    tb_logger = TensorBoardLogger('lightning_logs', '23_final', version=version)

    # add checkpointing to amazon-webcam dataset
    if 'aw' in version:
        model_ckpt = dict(save_last=True, save_top_k=1, every_n_epochs=333)
    else:
        model_ckpt = dict(save_last=False, save_top_k=0, every_n_epochs=None)

    trainer = pl.Trainer(accelerator='gpu', devices=[4],
                         max_epochs=total_epochs,
                         logger=tb_logger,
                         # track_grad_norm=2, 
                         gradient_clip_val=1.5,
                         log_every_n_steps=16,
                         multiple_trainloader_mode='max_size_cycle',
                         callbacks=[
                             ModelCheckpoint(monitor='source_val_loss', 
                                             auto_insert_metric_name=True,
                                             **model_ckpt,
                                             ),
                         ]
                        )

#     ckpt_path = 'lightning_logs/final/shit_check/checkpoints/last.ckpt'
#     checkpoint = torch.load(ckpt_path, map_location='cpu')
#     global_step_offset = checkpoint["global_step"]
#     trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset
#     del checkpoint
#     trainer.fit(model, ckpt_path=ckpt_path)
    trainer.fit(model)
    

def make_datasets(transform):
    amazon_train_dataset = UDADataset(
        os.path.join('datasets', 'OFFICE31', 'amazon_train'), 
        transform=transform
    )
    amazon_val_dataset = UDADataset(
        os.path.join('datasets', 'OFFICE31', 'amazon_val'), 
        transform=transform
    )
    webcam_train_dataset = UDADataset(
        os.path.join('datasets', 'OFFICE31', 'webcam_train'), 
        transform=transform
    )
    webcam_val_dataset = UDADataset(
        os.path.join('datasets', 'OFFICE31', 'webcam_val'), 
        transform=transform
    )
    dslr_train_dataset = UDADataset(
        os.path.join('datasets', 'OFFICE31', 'dslr_train'), 
        transform=transform
    )
    dslr_val_dataset = UDADataset(
        os.path.join('datasets', 'OFFICE31', 'dslr_val'),
        transform=transform
    )
    
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
    
    return (amazon_train_dataset, amazon_val_dataset), \
           (webcam_train_dataset, webcam_val_dataset), \
           (dslr_train_dataset, dslr_val_dataset), \
           (visda_source_train_dataset, visda_source_val_dataset), \
           (visda_target_train_dataset, visda_target_val_dataset)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
#    pl.seed_everything(41)
    num_simulations = 1

#    transform = T.Compose([
#        T.Resize((300, 300)), 
#        T.ToTensor(),
##         T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#        ])
    
    # transforms that are suitable for the pretrained model
    transform = ResNet50_Weights.DEFAULT.transforms()
    amazon_dataset, webcam_dataset, dslr_dataset, visda_source, visda_target = make_datasets(transform)

    dataset_pairs = [
        ('aw', 31, amazon_dataset, webcam_dataset),
        ('ad', 31, amazon_dataset, dslr_dataset),
        ('wa', 31, webcam_dataset, amazon_dataset),
        ('wd', 31, webcam_dataset, dslr_dataset),
        ('da', 31, dslr_dataset, amazon_dataset),
        ('dw', 31, dslr_dataset, webcam_dataset),
        ('visda', 12, visda_source, visda_target),
        ]

    training_modes = [
#            ('baseline', baseline_setting), 
#            ('negative_sampling', negative_sampling_setting), 
#            ('pretrain', pretrain_on_source_setting),
            ('no_adaptation', no_adaptation_setting)

       ]
    
    # Run training for several simulations to obtain more reliable results
    for i in range(num_simulations):
        # For each source-target pair
        for name, num_classes, source, target in dataset_pairs:
            # defines which mode of training to use
            for setting_name, training_mode in training_modes:
                settings = training_mode()
                version = f'{name}, {setting_name}, simulation_{i}'

                train(source, target, num_classes, settings, version)
                if name == 'aw' and setting_name == 'baseline':
                    version = f'{name}, remove_mismatched, simulation_{i}'
                    train(source, target, num_classes, settings, version, remove_mismatched=True)


#    ckpt_path = 'lightning_logs/lightning_logs/baseline/checkpoints/last-v2.ckpt'
#    checkpoint = torch.load(ckpt_path, map_location='cpu')
#    global_step_offset = checkpoint["global_step"]
#    trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset
#    del checkpoint
#    trainer.fit(model, ckpt_path=ckpt_path)
