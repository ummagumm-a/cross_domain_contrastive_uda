import torchvision.transforms as T
from resnetdsbn import resnet50dsbn
from utils import FeatureNormL2
from dataset import OfficeDataset
from model import UDAModel
from pytorch_lightning import Trainer, seed_everything

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor, StochasticWeightAveraging

if __name__ == '__main__':
    total_epochs = 300
    transform = T.Compose([T.Resize((300, 300)), T.ToTensor()])
    amazon_dataset = OfficeDataset('amazon', transform=transform)
    webcam_dataset = OfficeDataset('webcam', transform=transform)
    # for Office
    # b = 0.75
    # for Visda
    # b = 2.25

    resnet = resnet50dsbn(pretrained=True)
    resnet.fc2 = FeatureNormL2()
    resnet.out_features = 256

    # model = UDAModel(resnet, 3, 
    #                  (source_df.to_numpy()[:, :-1], source_df.to_numpy()[:, -1]), 
    #                  target_df.to_numpy()[:, :-1], total_epochs=total_epochs)
    model = UDAModel(resnet, 31, 
                     amazon_dataset, webcam_dataset, 
                     total_epochs=total_epochs, batch_size=64,
                     class_names=amazon_dataset.get_class_names())
    # model.setup('fit')

    seed_everything(41, workers=True)
    trainer = Trainer(accelerator='gpu', devices=[6], #strategy='ddp',
                      max_epochs=total_epochs, #logger=False,
                      track_grad_norm=2, gradient_clip_val=1.5,
                      log_every_n_steps=10, #deterministic=True,
    #                      limit_train_batches=5, limit_val_batches=5,
    #                     num_sanity_val_steps=0, #precision=16,
    #                      profiler='advanced',
                      callbacks=[
                          EarlyStopping(monitor='val_loss', 
                                        mode='min',
                                        patience=50,
                                       ),
                          ModelCheckpoint(monitor='val_loss', 
                                          save_last=False, 
                                          save_top_k=1,
                                          auto_insert_metric_name=True,
                                         ),
    #                          StochasticWeightAveraging(swa_lrs=1e-2),
                      ])

    trainer.fit(model)