import torchvision.transforms as T
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from resnetdsbn import resnet50dsbn, resnet101dsbn
from utils import FeatureNormL2
from dataset import UDADataset
from model import UDAModel
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor, StochasticWeightAveraging
import torch
import logging
from torchvision.models import ResNet50_Weights
import os
import copy

def default_office_params():
    params = {}
    params['batch_size'] = 84
    params['tau'] = 1.0
    params['b'] = 0.75

    params['explicit_negative_sampling_threshold'] = 0.0
    params['negative_sampling'] = None
    params['pretrain_num_epochs'] = 0

    return params

def default_visda_params():
    params = {}
    params['batch_size'] = 64
    params['tau'] = 0.05
    params['b'] = 2.25

    params['explicit_negative_sampling_threshold'] = 0.0
    params['negative_sampling'] = None
    params['pretrain_num_epochs'] = 0

    return params

def baseline_setting(default_params):
    return default_params

def baseline005_setting(default_params):
    default_params['tau'] = 0.05

    return default_params

def no_adaptation_setting(default_params):
    default_params['pretrain_num_epochs'] = 10000

    return default_params

def negative_sampling_setting(default_params):
    default_params['negative_sampling'] = 'hard'
    default_params['explicit_negative_sampling_threshold'] = 0.5

    return default_params

def random_negative_sampling_setting(default_params):
    default_params['negative_sampling'] = 'random'
    default_params['explicit_negative_sampling_threshold'] = 0.5

    return default_params

def pretrain_on_source_setting(default_params):
    default_params['pretrain_num_epochs'] = 100

    return default_params


def train(resnet, source_dataset, target_dataset, num_classes, settings, version, device, total_epochs=500):
    # Define the backbone
    classification_head = nn.Linear(resnet.in_features, num_classes, bias=False)

    model = UDAModel(resnet, classification_head, num_classes, 
                     source_dataset, target_dataset,
                     total_epochs=total_epochs, 
                     num_workers=6, **settings, grad_clip=1.5,
                     class_names=source_dataset[0].get_class_names())

    tb_logger = TensorBoardLogger('lightning_logs', 'shit', version=version)

    # add checkpointing to amazon-webcam dataset
    if 'aw' in version:
        model_ckpt = dict(save_last=True, save_top_k=1)
    else:
        model_ckpt = dict(save_last=False, save_top_k=0)

    trainer = pl.Trainer(accelerator='gpu', devices=[device],
                         max_epochs=total_epochs,
                         logger=tb_logger,
                         log_every_n_steps=13,
                         gradient_clip_val=None if settings['track_grad_norm'] else 1.5,
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
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
#    pl.seed_everything(41)
    num_simulations = 1

    # transforms that are suitable for the pretrained model
    transform = ResNet50_Weights.DEFAULT.transforms()
    amazon_dataset, webcam_dataset, dslr_dataset, visda_source, visda_target = make_datasets(transform)

    cp = lambda x: tuple(map(lambda y: copy.deepcopy(y), x))

    dataset_pairs = [
        ('aw', 31, cp(amazon_dataset), cp(webcam_dataset)),
#        ('ad', 31, cp(amazon_dataset), cp(dslr_dataset)),
#        ('wa', 31, cp(webcam_dataset), cp(amazon_dataset)),
#        ('wd', 31, cp(webcam_dataset), cp(dslr_dataset)),
#        ('da', 31, cp(dslr_dataset), cp(amazon_dataset)),
#        ('dw', 31, cp(dslr_dataset), cp(webcam_dataset)),
#        ('visda', 12, cp(visda_source), cp(visda_target)),
        ]

    track_grad_norm = False

    training_modes = [
            ('baseline_lmbda_0.1', baseline_setting, 2), 
#            ('negative_sampling', negative_sampling_setting, 1), 
#            ('pretrain', pretrain_on_source_setting, 5),
#            ('no_adaptation', no_adaptation_setting, 3),
#            ('random_sampling', random_negative_sampling_setting, 3),
       ]

    
    # Run training for several simulations to obtain more reliable results
    for i in range(num_simulations):
        # For each source-target pair
        for name, num_classes, source, target in dataset_pairs:
            # defines which mode of training to use
            for setting_name, training_mode, device in training_modes:
                if name == 'visda':
                    resnet = resnet101dsbn(pretrained=True, in_features=256)
                    default_settings = default_visda_params()
                else:
                    resnet = resnet50dsbn(pretrained=True, in_features=256)
                    default_settings = default_office_params()

                settings = training_mode(default_settings)
#                settings['remove_mismatched'] = True
#                settings['contrastive_only'] = True
#                settings['tau'] = 0.05
                settings['lmbda'] = 0.1
                settings['track_grad_norm'] = True
                logger.info(f"Training settings: \n{settings}")

                resnet.fc2 = FeatureNormL2()
                version = f'{name}, {setting_name}, simulation_{i}'

                train(resnet, source, target, num_classes, settings, version, device)
#                if name == 'aw' and setting_name == 'baseline':
#                    version = f'{name}, remove_mismatched, simulation_{i}'
#                    train(source, target, num_classes, settings, version, device, remove_mismatched=True)


#    ckpt_path = 'lightning_logs/lightning_logs/baseline/checkpoints/last-v2.ckpt'
#    checkpoint = torch.load(ckpt_path, map_location='cpu')
#    global_step_offset = checkpoint["global_step"]
#    trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset
#    del checkpoint
#    trainer.fit(model, ckpt_path=ckpt_path)
