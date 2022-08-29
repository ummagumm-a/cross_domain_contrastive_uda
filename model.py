import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
import torchmetrics
# from info_nce import InfoNCE
import numpy as np
# from kmeans_pytorch import kmeans
from sklearn.cluster import MiniBatchKMeans
import os

import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

class UDAModel(pl.LightningModule):
    def __init__(self, feature_extractor, n_classes, 
                       source_dataset, target_dataset, 
                       tau=1., b=0.75, test_size=0.3, lmbda=1.4,
                       batch_size=64, num_workers=48,
                       total_epochs=None, class_names=None):
        super().__init__()
        self.n_classes = n_classes
        # The last layer should be smth giving (N, out_features)
        self.feature_extractor = feature_extractor
        # What should it be? How much layers?
        self.classification_head = nn.Sequential(
            nn.Linear(feature_extractor.out_features, n_classes),
            nn.Softmax(dim=1)
        )
        
        self.classification_loss = nn.CrossEntropyLoss()
        
        self.tau = tau
        self.b = b
        self.lmbda = lmbda
        self.test_size = test_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.total_epochs = self.check_total_epochs(total_epochs)
        self.class_names = self.check_class_names(class_names)
        
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        
        self.accuracy_metric = torchmetrics.Accuracy(num_classes=n_classes, average=None)
        self.precision_metric = torchmetrics.Precision(num_classes=n_classes, average=None)
        self.recall_metric = torchmetrics.Recall(num_classes=n_classes, average=None)
        
        self.clusterizer = MiniBatchKMeans(n_clusters=self.n_classes, 
                                           batch_size=self.batch_size,
                                           n_init=1)
        
    def check_class_names(self, class_names):
        if class_names is None:
            class_names = np.arange(self.n_classes)
            
        return class_names
    
    def check_total_epochs(self, total_epochs):
        if total_epochs is None:
            raise Exception("total_epochs should be not 'None'")
        elif not isinstance(total_epochs, int):
            raise Exception(f"total_epochs should be int, but is '{type(total_epochs)}'")
        else:
            return total_epochs
        
    def __call__(self, input, domain_label):
        features = self.feature_extractor(input, domain_label)
        
        return self.classification_head(features)
    
    def configure_optimizers(self):
        sgd = torch.optim.SGD([
            { 'params': self.feature_extractor.parameters(), 'lr': 1e-3 },
            { 'params': self.classification_head.parameters(), 'lr': 1e-2 },
        ], momentum=0.9)
        
        def mult_factor(epoch):
            p = epoch / self.total_epochs
            
            return (1 + 10 * p) ** (-self.b)
        
        scheduler = LambdaLR(sgd, mult_factor)
        
        return {
            "optimizer": sgd,
            "lr_scheduler": { "scheduler": scheduler },
        }
    
    def _label_count(self, y):
        y = torch.concat((y, torch.arange(self.n_classes, device=self.device)))
        _, label_count_ = y.unique(dim=0, return_counts=True)

        return label_count_ - 1
    
    def calculate_class_centers(self):
        dataloader = self.dataloader_source_for_centroids()

        accum = torch.zeros((self.n_classes, self.feature_extractor.out_features), device=self.device)
        labels_count = torch.zeros(self.n_classes, dtype=torch.long, device=self.device)

        # TODO: may result in overflow. Rewrite with running mean.
        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            features = self.feature_extractor(x, 0)
            label_count_ = self._label_count(y)
            y = y.view(y.size(0), 1).expand(-1, features.size(1))

            batch_class_means = torch.zeros_like(accum, dtype=torch.float32, device=self.device)\
                                     .scatter_add_(0, y, features)
            accum += batch_class_means
            labels_count += label_count_

        initial_centers = accum / labels_count.float().unsqueeze(1)

        self.clusterizer.set_params(init=initial_centers.cpu())
        
    def fit_clusterizer(self):
        dataloader = self.dataloader_target_for_clustering()

        for x, _ in dataloader:
            x = x.to(self.device)
            features = self.feature_extractor(x, 1)
            features = features.cpu().numpy()
            self.clusterizer.partial_fit(features)

        return self.clusterizer.cluster_centers_
    
    def assign_labels(self):
        dataloader = self.dataloader_target_for_clustering()

        collected_labels = []
        for x, _ in dataloader:
            x = x.to(self.device)
            features = self.feature_extractor(x, 1)
            features = features.cpu().numpy()

            pred = self.clusterizer.predict(features)
            collected_labels.append(pred)

        return np.hstack(collected_labels)
        
    def on_train_epoch_start(self):
        self.feature_extractor.eval()
        with torch.no_grad():
            self.calculate_class_centers()
            self.fit_clusterizer()
            assigned_labels = self.assign_labels()
            self.assigned_labels = assigned_labels
            self.target_dataset.update_labels(assigned_labels)
            self.log('unique labels', len(np.unique(assigned_labels)))
            
        self.feature_extractor.train()
    
    def classification_step(self, batch):
        source_x, source_y = batch
        
        pred = self(source_x, 0)
        classification_loss = self.classification_loss(pred, source_y)
        
        return classification_loss
    
    def get_same_class(self, batch, cls):
        x, y = batch
        
        return x[y == cls]
    
    def contrastive_step(self, anchors_batch, other_batch):
        other_x, other_y = other_batch
        
        def helper(anchor_item, other_items):
            return torch.exp(other_items @ anchor_item / self.tau)
        
        contrastive_loss = 0
        for x, y in zip(*anchors_batch):
            same_class = self.get_same_class(other_batch, y)
            if len(same_class) == 0:
                continue
                
            contrastive_loss -= torch.mean(
                torch.log(
                    helper(x, same_class) / torch.sum(helper(x, other_x))
                )
            )
            
        return contrastive_loss
    
    def split_batch_in_two(self, batch):
        (x, y) = batch
        half_batch_len = len(x) // 2
        x1 = x[:half_batch_len]
        y1 = y[:half_batch_len]
        x2 = x[half_batch_len:]
        y2 = y[half_batch_len:]
        
        assert abs(len(x1) - len(x2)) <= 1, f'{abs(len(x1) - len(x2))}'
        
        return (x1, y1), (x2, y2)
    
    def training_step(self, batch):
#         logger.debug('start of training_step')
        source_for_classification, (source_x, source_y) = self.split_batch_in_two(batch['source'])
        classification_loss = self.classification_step(source_for_classification)
        
        target_x, target_y = batch['target']
        
#         logger.debug(f'training_step batch_size: {len(source_for_classification[0])}, {len(source_x)}, {len(target_x)}')
        target_features = self.feature_extractor(target_x, 1)
        source_features = self.feature_extractor(source_x, 0)
        contrastive_loss = self.contrastive_step((target_features, target_y), (source_features, source_y)) \
                         + self.contrastive_step((source_features, source_y), (target_features, target_y))
        
        train_loss = classification_loss + self.lmbda * contrastive_loss
        
        self.log_dict({
            "classification_loss": classification_loss,
            "contrastive_loss": contrastive_loss,
            "train_loss": train_loss
        })
        
#         logger.debug('end of training_step')
        
        return train_loss
    
    def val_metrics(self, pred, y):
        # TODO: for optimization may need to first calculate tp, tn, fp, fn
        accuracy = self.accuracy_metric(pred, y)
        precision = self.precision_metric(pred, y)
        recall = self.recall_metric(pred, y)
        
        log_dict = {}
        for i, class_name in enumerate(self.class_names):
            log_dict[f'accuracy_{class_name}'] = accuracy[i]
            log_dict[f'precision_{class_name}'] = precision[i]
            log_dict[f'recall_{class_name}'] = recall[i]
            
        return log_dict
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
#         logger.debug(f'validation_step batch_size: {len(x)}')
        
        pred = self(x, 0)
        loss = self.classification_loss(pred, y)
        
        self.log("val_loss", loss)
        self.log_dict(self.val_metrics(pred, y))
        
        return loss
        
    def setup(self, stage):
        if stage == 'fit':
#             self.source_dataset = UDADataset(self.source_dataset[0], self.source_dataset[1])
#             self.target_dataset = UDADataset(self.target_dataset, None)

            l = len(self.source_dataset)
            test_len = int(self.test_size * l)
            lens = (l - test_len, test_len)
            
            self.train_source_dataset, self.val_source_dataset = \
                random_split(self.source_dataset, lens)
            
    def dataloader_target_for_clustering(self):
        return DataLoader(self.target_dataset,
                          batch_size=self.batch_size,
                          pin_memory=True,
                          shuffle=False,
                          num_workers=self.num_workers)
    
    def dataloader_source_for_centroids(self):
        return DataLoader(self.train_source_dataset,
                          batch_size=self.batch_size,
                          pin_memory=True,
                          shuffle=True,
                          num_workers=self.num_workers)
    
    def train_dataloader(self):
        dataloaders = {}
        dataloaders['source'] = \
                            DataLoader(self.train_source_dataset,
                                       batch_size=self.batch_size * 2,
                                       pin_memory=True,
                                       shuffle=True,
                                       num_workers=self.num_workers * 2)
        dataloaders['target'] = \
                            DataLoader(self.target_dataset,
                                       batch_size=self.batch_size,
                                       pin_memory=True,
                                       shuffle=True,
                                       num_workers=self.num_workers)
        
        return dataloaders
    
    def val_dataloader(self):
        dataloader = DataLoader(self.val_source_dataset,
                                batch_size=self.batch_size,
                                pin_memory=True,
                                shuffle=False,
                                num_workers=self.num_workers)
        
        return dataloader
