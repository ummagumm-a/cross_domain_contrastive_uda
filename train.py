import torchvision.transforms as T
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from resnetdsbn import resnet50dsbn
from utils import FeatureNormL2
from dataset import OfficeDataset, RemoveMismatchedAdapter
from model import UDAModel
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor, StochasticWeightAveraging
import torch
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    pl.seed_everything(41)

    total_epochs = 2500
    pretrain_epochs = 0
    transform = T.Compose([T.Resize((300, 300)), T.ToTensor()])
    amazon_dataset = OfficeDataset('amazon', transform=transform)
    webcam_dataset = OfficeDataset('webcam', transform=transform)
    num_classes = 31
    negative_sampling_threshold = 0.5
    # for Office
    # b = 0.75
    # for Visda
    # b = 2.25

    resnet = resnet50dsbn(pretrained=True)
    resnet.fc2 = FeatureNormL2()
    resnet.out_features = 256
    classification_head = nn.Linear(resnet.out_features, num_classes)
    
    model = UDAModel(resnet, classification_head, num_classes, 
                     amazon_dataset, webcam_dataset, 
                     total_epochs=total_epochs, batch_size=64,
                     num_workers=1, pretrain_num_epochs=pretrain_epochs,
                     explicit_negative_sampling_threshold=negative_sampling_threshold,
                     class_names=amazon_dataset.get_class_names())

    tb_logger = TensorBoardLogger('lightning_logs', version=f'ns_analisys_{negative_sampling_threshold}')

    trainer = pl.Trainer(accelerator='gpu', devices=[4],
                         max_epochs=total_epochs,
                         logger=tb_logger,
                         track_grad_norm=2, 
                         gradient_clip_val=1.5,
                         log_every_n_steps=16,
                         multiple_trainloader_mode='max_size_cycle',
                         callbacks=[
                             ModelCheckpoint(monitor='source_val_loss', 
                                             save_last=True, 
                                             save_top_k=1,
                                             auto_insert_metric_name=True,
                                             ),
                         ]
                        )

#    ckpt_path = 'lightning_logs/lightning_logs/negative_sampling/checkpoints/last.ckpt'
#    checkpoint = torch.load(ckpt_path, map_location='cpu')
#    global_step_offset = checkpoint["global_step"]
#    trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset
#    del checkpoint
#    trainer.fit(model, ckpt_path=ckpt_path)
    trainer.fit(model)
