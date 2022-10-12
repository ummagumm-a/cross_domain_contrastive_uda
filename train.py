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
#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    pl.seed_everything(41)

    total_epochs = 1200
    pretrain_epochs = 500
    transform = T.Compose([T.Resize((300, 300)), T.ToTensor()])
    amazon_dataset = OfficeDataset('amazon', transform=transform)
    webcam_dataset = OfficeDataset('webcam', transform=transform)
    num_classes = 31
    # for Office
    # b = 0.75
    # for Visda
    # b = 2.25

    resnet = resnet50dsbn(pretrained=True)
    resnet.fc2 = FeatureNormL2()
    # resnet.fc2 = nn.Identity()
    resnet.out_features = 256
    classification_head = nn.Linear(resnet.out_features, num_classes)

    # model = UDAModel(resnet, 3, 
    #                  (source_df.to_numpy()[:, :-1], source_df.to_numpy()[:, -1]), 
    #                  target_df.to_numpy()[:, :-1], total_epochs=total_epochs)
    model = UDAModel(resnet, classification_head, num_classes, 
                     amazon_dataset, webcam_dataset, 
                     total_epochs=total_epochs, batch_size=64,
                     num_workers=1, pretrain_num_epochs=pretrain_epochs,
                     class_names=amazon_dataset.get_class_names())
    # model.setup('fit')
    tb_logger = TensorBoardLogger('lightning_logs', version='baseline_source_pretrain')

    trainer = pl.Trainer(accelerator='gpu', devices=[7], #strategy='ddp',
                         max_epochs=total_epochs, #logger=False,
                         logger=tb_logger,
    #                      track_grad_norm=2, 
                         gradient_clip_val=1.5,
                         log_every_n_steps=10, #deterministic=True,
                         multiple_trainloader_mode='max_size_cycle',
    #                      limit_train_batches=5, limit_val_batches=5,
    #                     num_sanity_val_steps=0, #precision=16,
    #                      profiler='advanced',
                         callbacks=[
#                             EarlyStopping(monitor='source_val_loss', 
#                                           mode='min',
#                                           patience=50,
#                                          ),
                             ModelCheckpoint(monitor='source_val_loss', 
                                             save_last=True, 
                                             save_top_k=1,
                                             auto_insert_metric_name=True,
                                             ),
    #                          StochasticWeightAveraging(swa_lrs=1e-2),
                         ]
                        )

    ckpt_path = 'lightning_logs/lightning_logs/baseline_source_pretrain/checkpoints/epoch=508-step=8144.ckpt'
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    global_step_offset = checkpoint["global_step"]
    trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset
    del checkpoint
    trainer.fit(model)#, train_dataloaders=model.train_dataloader(), val_dataloaders=model.val_dataloader())
