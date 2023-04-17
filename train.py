import torchvision.transforms as T
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from resnetdsbn import resnet50dsbn, resnet101dsbn
from utils import FeatureNormL2
from dataset import UDADataset, make_office_datasets_kfold, get_visda_datasets
from model import UDAModel
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor, StochasticWeightAveraging
import torch
torch.backends.cudnn.benchmark = True
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

def remove_mismatched_setting(default_params):
    default_params['remove_mismatched'] = True

    return default_params

def contrastive_only_setting(default_params):
    default_params['contrastive_only'] = True

    return default_params

def smaller_lmbda_setting(default_params):
    default_params['lmbda'] = 0.1

    return default_params

def smaller_tau_setting(default_params):
    default_params['tau'] = 0.05

    return default_params

def greater_tau_setting(default_params):
    # TODO: change to 1.0
    default_params['tau'] = 1.0

    return default_params

def smaller_tau_lmbda_setting(default_params):
    default_params['tau'] = 0.05
    default_params['lmbda'] = 0.1

    return default_params

def train(resnet, source_dataset, target_dataset, num_classes, settings, version, device, model_ckpt, total_epochs=500, folder='shit'):

    logger.info(f"Training settings: \n{settings}")
    # Define the backbone
    classification_head = nn.Linear(resnet.in_features, num_classes, bias=False)

    model = UDAModel(resnet, classification_head, num_classes, 
                     source_dataset, target_dataset,
                     total_epochs=total_epochs, 
                     num_workers=6, **settings, grad_clip=1.5,
                     class_names=source_dataset[0].get_class_names())

    tb_logger = TensorBoardLogger('lightning_logs', folder, version=version)


    grad_norm = 1.5
    if 'track_grad_norm' in settings and settings['track_grad_norm']:
        grad_norm = None

    trainer = pl.Trainer(accelerator='gpu', devices=[device],
                         max_epochs=total_epochs,
                         logger=tb_logger,
                         log_every_n_steps=13,
                         gradient_clip_val=grad_norm,
                         multiple_trainloader_mode='max_size_cycle',
                         callbacks=[
                             ModelCheckpoint(monitor='source_val_loss', 
                                             auto_insert_metric_name=True,
                                             **model_ckpt,
                                             ),
                         ]
                        )

    trainer.fit(model)


def train_single_fold(fold_num, office_datasets=None, visda_datasets=None):
    if office_datasets is not None:
        amazon_dataset, webcam_dataset, dslr_dataset = datasets

    if visda_datasets is not None:
        visda_source, visda_target = visda_datasets

    cp = lambda x: tuple(map(lambda y: copy.deepcopy(y), x))

    dataset_pairs = [
        ('aw', 31, cp(amazon_dataset), cp(webcam_dataset)),
        ('ad', 31, cp(amazon_dataset), cp(dslr_dataset)),
        ('wa', 31, cp(webcam_dataset), cp(amazon_dataset)),
        ('wd', 31, cp(webcam_dataset), cp(dslr_dataset)),
        ('da', 31, cp(dslr_dataset), cp(amazon_dataset)),
        ('dw', 31, cp(dslr_dataset), cp(webcam_dataset)),
#        ('visda', 12, cp(visda_source), cp(visda_target)),
        ]

    training_modes = [
            ('no_adaptation', no_adaptation_setting, 7),
            ('baseline', baseline_setting, 6), 
            ('pretrain', pretrain_on_source_setting, 5),
            ('negative_sampling', negative_sampling_setting, 4), 
            ('random_sampling', random_negative_sampling_setting, 3),
            ('smaller_tau_lmbda_baseline', lambda x: smaller_tau_lmbda_setting(baseline_setting(x)), 2),

#            ('smaller_lmbda_baseline', lambda x: smaller_lmbda_setting(baseline_setting(x)), 3),
#            ('smaller_tau_baseline', lambda x: smaller_tau_setting(baseline_setting(x)), 7),
#            
#            ('smaller_lmbda_neg_sampl', lambda x: smaller_lmbda_setting(negative_sampling_setting(x)), 1),
#            ('smaller_tau_lmbda_neg_sampl', lambda x: smaller_tau_lmbda_setting(negative_sampling_setting(x)), 1),
#            
#            ('smaller_lmbda_pretrain', lambda x: smaller_lmbda_setting(pretrain_on_source_setting(x)), 1),
#
#            ('remove_mismatched', lambda x: remove_mismatched_setting(baseline_setting(x)), 4),
#            ('remove_mismatched_smaller_lmbda', lambda x: remove_mismatched_setting(smaller_lmbda_setting(baseline_setting(x))), 4),
#            ('contrastive_only', lambda x: contrastive_only_setting(baseline_setting(x)), 5),
#            ('greater_tau_neg_sampl', lambda x: greater_tau_setting(negative_sampling_setting(x)), 6),
#            ('greater_tau_baseline', lambda x: greater_tau_setting(baseline_setting(x)), 4),
       ]


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

            resnet.fc2 = FeatureNormL2()
            settings = training_mode(default_settings)
#            if fold_num == 0:
#                settings['track_grad_norm'] = True

            # Default
            version = f'{name}, {setting_name}, fold_{fold_num}'

            # add checkpointing to amazon-webcam dataset
            if False and fold_num == 1 and 'aw' in version and not ('pretrain' in version or 'random_sampling' in version or 'no_adaptation' in version):
                model_ckpt = dict(save_last=True, save_top_k=4, every_n_epochs=100)
            else:
                model_ckpt = dict(save_last=False, save_top_k=0)

            train(resnet, source, target, num_classes, settings, version, device, model_ckpt=model_ckpt, folder='final_kfold_run')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
#    pl.seed_everything(41)

    # transforms that are suitable for the pretrained model
    transform = ResNet50_Weights.DEFAULT.transforms()

    # For all splits
    for i, datasets in enumerate(make_office_datasets_kfold(transform, n_splits=4)):
        train_single_fold(i, office_datasets=datasets)

#    transform = ResNet101_Weights.DEFAULT.transforms()
#    train_single_fold(1, visda_datasets=get_visda_datasets(transform))

