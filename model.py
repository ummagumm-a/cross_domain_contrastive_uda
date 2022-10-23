import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
import torchmetrics
# from info_nce import InfoNCE
import numpy as np
# from kmeans_pytorch import kmeans
from sklearn.cluster import KMeans
import os
import plotly.express as px

import logging
logger = logging.getLogger(__name__)


class UDAModel(pl.LightningModule):
    def __init__(self, feature_extractor, classification_head, n_classes, 
                       source_dataset, target_dataset, 
                       tau=1., b=0.75, test_size=0.3, lmbda=1.4, explicit_negative_sampling_threshold=0.5,
                       batch_size=64, num_workers=48, pretrain_num_epochs=0,
                       total_epochs=None, class_names=None):
        super().__init__()
        self.n_classes = n_classes
        # The last layer should be smth giving (N, out_features)
        self.feature_extractor = feature_extractor
        self.classification_head = classification_head
        self.classification_loss = nn.CrossEntropyLoss()
        
        self.tau = tau
        self.explicit_negative_sampling_threshold = explicit_negative_sampling_threshold
        self.b = b
        self.lmbda = lmbda
        self.test_size = test_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pretrain_num_epochs = pretrain_num_epochs 
        
        self.total_epochs = self.check_total_epochs(total_epochs)
        self.class_names = self.check_class_names(class_names)
        
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        
        self.accuracy_metric = torchmetrics.Accuracy(num_classes=n_classes, average=None)
        self.precision_metric = torchmetrics.Precision(num_classes=n_classes, average=None)
        self.recall_metric = torchmetrics.Recall(num_classes=n_classes, average=None)
        
        self.clusterizer = KMeans(n_clusters=self.n_classes, n_init=1)
        
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
        for x, y, _ in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            features = self.feature_extractor(x, 0)
            
            label_count_ = self._label_count(y)
            labels_count += label_count_
            
            y = y.view(y.size(0), 1).expand(-1, features.size(1))

            batch_class_means = torch.zeros_like(accum, dtype=torch.float32, device=self.device)\
                                     .scatter_add_(0, y, features)
            accum += batch_class_means
            

        initial_centers = accum / labels_count.float().unsqueeze(1)

        self.clusterizer.set_params(init=initial_centers.cpu())

    def fit_clusterizer(self):
        dataloader = self.dataloader_target_for_clustering()
        all_features = []
        for x, _, _ in dataloader:
            x = x.to(self.device)
            features = self.feature_extractor(x, 1)
            features = features.cpu().numpy()
            all_features.append(features)
        all_features = np.vstack(all_features)
        
        self.clusterizer.fit(all_features)
    
    def assign_labels(self):
        dataloader = self.dataloader_target_for_clustering()

        collected_labels = []
        for x, _, _ in dataloader:
            x = x.to(self.device)
            features = self.feature_extractor(x, 1)
            features = features.cpu().numpy()

            pred = self.clusterizer.predict(features)
            collected_labels.append(pred)

        return np.hstack(collected_labels)
    
    def visualize_pseudo_labeling(self):
        real_labels, labels = [], []
        for i in range(len(self.target_dataset)):
            real_labels.append(self.target_dataset.get_real_labels()[i])
            labels.append(self.target_dataset.get_labels()[i])

        real_labels, labels = np.array(real_labels), np.array(labels)
        pairs = np.vstack((real_labels, labels))
        unique_pairs, counts = np.unique(pairs, axis=1, return_counts=True)
        fig = px.scatter_3d(x=unique_pairs[0, :], y=unique_pairs[1, :], z=counts)
        fig.show()
        
    def class_analysis(self, dataset):
        labels = dataset.get_labels()
        labels = np.array(labels)
        real_labels = dataset.get_real_labels()
        real_labels = np.array(real_labels)

        # find unassigned labels
        unassigned_labels = set(range(31)).difference(np.unique(labels).tolist())
        self.logger.experiment.add_text('unassigned labels', 
                                        str(unassigned_labels), self.current_epoch)

        # for each class find the distribution of assigned classes
        label_stats = {}
        for i in range(31):
            ilabels = labels[real_labels == i]
            self.logger.experiment.add_histogram(f'class {i} assigned to:', 
                                                 ilabels, self.current_epoch,
                                                 bins=31)
            
    def on_train_epoch_start(self):
        self.feature_extractor.eval()
        with torch.no_grad():
            self.calculate_class_centers()
            self.fit_clusterizer()
            assigned_labels = self.assign_labels()
            self.target_dataset.update_labels(assigned_labels)
#            self.visualize_pseudo_labeling()
            self.class_analysis(self.target_dataset)
            self.log('unique labels', len(np.unique(self.target_dataset.get_labels())))
            
        self.feature_extractor.train()
    
    def classification_step(self, batch):
        source_x, source_y, _ = batch
        
        pred = self(source_x, 0)
        classification_loss = self.classification_loss(pred, source_y)
        
        return classification_loss
    
    def get_same_class(self, batch, cls):
        x, y, _ = batch
        
        return x[y == cls]
    
    def analyze_negative_samples(self, anchor, negatives, positives, anchor_type):
        if anchor_type == 'source':
            # since anchor is from source, its class is unambiguous
            # Therefore, we'll check correctness of pseudo-labeling in target batch
            # For that we'll use actual labels of targets.
            x, y, _ = anchor
            neg_x, _, neg_y_real = negatives
            pos_x, _, pos_y_real = positives
                
            false_negatives_sims = neg_x[neg_y_real == y]
            if len(false_negatives_sims) != 0:
                self.logger.experiment.add_histogram('source_anchor, false negative sims', 
                                                     false_negatives_sims, 
                                                     self.global_step)
                self.logger.experiment.add_histogram(f'source_anchor, false negative sims, anchor {y}', 
                                                     false_negatives_sims, 
                                                     self.global_step)
            
            true_negatives_sims = neg_x[neg_y_real != y]
            if len(true_negatives_sims) != 0:
                self.logger.experiment.add_histogram('source_anchor, true negative sims', 
                                                     true_negatives_sims, 
                                                     self.global_step)
                self.logger.experiment.add_histogram(f'source_anchor, true negative sims, anchor {y}', 
                                                     true_negatives_sims, 
                                                     self.global_step)
                
            false_positives_sims = pos_x[pos_y_real != y]
            if len(false_positives_sims) != 0:
                self.logger.experiment.add_histogram('source_anchor, false positive sims', 
                                                     false_positives_sims, 
                                                     self.global_step)
                self.logger.experiment.add_histogram(f'source_anchor, false positive sims, anchor {y}', 
                                                     false_positives_sims, 
                                                     self.global_step)
            
            true_positives_sims = pos_x[pos_y_real == y]
            if len(true_positives_sims) != 0:
                self.logger.experiment.add_histogram('source_anchor, true positive sims', 
                                                     true_positives_sims, 
                                                     self.global_step)
                self.logger.experiment.add_histogram(f'source_anchor, true positive sims, anchor {y}', 
                                                     true_positives_sims, 
                                                     self.global_step)
            
            
        elif anchor_type == 'target':
            # The anchor is from target, therefore its label may be incorrect.
            # Having a source batch, we'll check how much samples are of the same actual class
            # as target anchor
            x, _, y_real = anchor
            neg_x, neg_y, _ = negatives
            pos_x, pos_y, _ = positives
            
            false_negatives_sims = neg_x[neg_y == y_real]
            if len(false_negatives_sims) != 0:
                self.logger.experiment.add_histogram('target_anchor, false negative sims', 
                                                     false_negatives_sims, 
                                                     self.global_step)
                self.logger.experiment.add_histogram(f'target_anchor, false negative sims, anchor {y_real}', 
                                                     false_negatives_sims, 
                                                     self.global_step)
            
            true_negatives_sims = neg_x[neg_y != y_real]
            if len(true_negatives_sims) != 0:
                self.logger.experiment.add_histogram('target_anchor, true negative sims', 
                                                     true_negatives_sims, 
                                                     self.global_step)
                self.logger.experiment.add_histogram(f'target_anchor, true negative sims, anchor {y_real}', 
                                                     true_negatives_sims, 
                                                     self.global_step)
                
            false_positives_sims = pos_x[pos_y != y_real]
            if len(false_positives_sims) != 0:
                self.logger.experiment.add_histogram('target_anchor, false positive sims', 
                                                     false_positives_sims, 
                                                     self.global_step)
                self.logger.experiment.add_histogram(f'target_anchor, false positive sims, anchor {y_real}', 
                                                     false_positives_sims, 
                                                     self.global_step)
            
            true_positives_sims = pos_x[pos_y == y_real]
            if len(true_positives_sims) != 0:
                self.logger.experiment.add_histogram('target_anchor, true positive sims', 
                                                     true_positives_sims, 
                                                     self.global_step)
                self.logger.experiment.add_histogram(f'target_anchor, true positive sims, anchor {y_real}', 
                                                     true_positives_sims, 
                                                     self.global_step)
                
        else:
            raise Exception(f"Wrong anchor type: {anchor_type}")
    
    def contrastive_step(self, anchors_batch, other_batch, anchor_type):
        other_x, other_y, other_y_real = other_batch
        
        contrastive_loss = 0
        for x, y, y_real in zip(*anchors_batch):
            same_class_indices = other_batch[1] == y
            if not same_class_indices.any():
                continue

            positives = other_batch[0][same_class_indices]
            negatives = other_batch[0][~same_class_indices]

            positives_sims = positives @ x
            positives_exp = torch.exp(positives_sims / self.tau)
            negatives_sims = negatives @ x
            self.analyze_negative_samples((x, y, y_real), 
                                          (negatives_sims, other_y[~same_class_indices], other_y_real[~same_class_indices]),
                                          (positives_sims, other_y[same_class_indices], other_y_real[same_class_indices]),
                                          anchor_type)
            negatives_sims = negatives_sims[negatives_sims > self.explicit_negative_sampling_threshold]
            negatives_exp = torch.exp(negatives_sims / self.tau)

            logit = positives_exp / (negatives_exp.sum() + positives_exp.sum())
            log = torch.log(logit)
            sum_over_all_positives = torch.nanmean(log)
            
            if not sum_over_all_positives.isnan():
                contrastive_loss -= sum_over_all_positives
            
        return contrastive_loss
    
    def split_batch_in_two(self, batch):
        (x, y, y_real) = batch
        half_batch_len = len(x) // 2
        x1 = x[:half_batch_len]
        y1 = y[:half_batch_len]
        y_real1 = y_real[:half_batch_len]
        x2 = x[half_batch_len:]
        y2 = y[half_batch_len:]
        y_real2 = y_real[half_batch_len:]
        
        assert abs(len(x1) - len(x2)) <= 1, f'{abs(len(x1) - len(x2))}'
        
        return (x1, y1, y_real1), (x2, y2, y_real2)
    
    def training_step(self, batch):
        source_for_classification, (source_x, source_y, source_y_real) = self.split_batch_in_two(batch['source'])
        classification_loss = self.classification_step(source_for_classification)
        
        if self.pretrain_num_epochs <= self.current_epoch:
            target_x, target_y, target_y_real = batch['target']

            target_features = self.feature_extractor(target_x, 1)
            source_features = self.feature_extractor(source_x, 0)
            contrastive_target_anchor = self.contrastive_step(
                (target_features, target_y, target_y_real), 
                (source_features, source_y, source_y_real), 
                'target'
            )
            contrastive_source_anchor = self.contrastive_step(
                (source_features, source_y, source_y_real), 
                (target_features, target_y, target_y_real), 
                'source'
            )
            contrastive_loss = contrastive_target_anchor + contrastive_source_anchor

        else:
            contrastive_loss = -1
        train_loss = classification_loss + self.lmbda * contrastive_loss
        
        self.log_dict({
            "classification_loss": classification_loss,
            "contrastive_loss": contrastive_loss,
            "train_loss": train_loss
        }, on_epoch=True, on_step=False)
                
        return train_loss
    
    def val_metrics(self, pred, y, prefix):
        # TODO: for optimization may need to first calculate tp, tn, fp, fn
        accuracy = self.accuracy_metric(pred, y)
        precision = self.precision_metric(pred, y)
        recall = self.recall_metric(pred, y)
        
        log_dict = {}
        for i, class_name in enumerate(self.class_names):
            log_dict[f'{prefix}_accuracy_{class_name}'] = accuracy[i]
            log_dict[f'{prefix}_precision_{class_name}'] = precision[i]
            log_dict[f'{prefix}_recall_{class_name}'] = recall[i]

        log_dict[f'{prefix}_accuracy'] = torch.nanmean(accuracy)
        log_dict[f'{prefix}_precision'] = torch.nanmean(precision)
        log_dict[f'{prefix}_recall'] = torch.nanmean(recall)
            
        return log_dict

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y, y_real = batch
        if dataloader_idx == 0:
            pred = self(x, 0)
            loss = self.classification_loss(pred, y)

            self.log("source_val_loss", loss, on_epoch=True, on_step=False, add_dataloader_idx=False)
            self.log_dict(self.val_metrics(pred, y, 'source'), on_epoch=True, on_step=False, add_dataloader_idx=False)
        elif dataloader_idx == 1:
            pred = self(x, 0)
            loss = self.classification_loss(pred, y_real)

            self.log("target_val_loss", loss, on_epoch=True, on_step=False, add_dataloader_idx=False)
            self.log_dict(self.val_metrics(pred, y_real, 'target'), on_epoch=True, on_step=False, add_dataloader_idx=False)
        else:
            raise Exception(f'Weird dataloader_idx: {dataloader_idx}')
        
    def setup(self, stage):
        if stage == 'fit':
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
        dataloaders = []
        dataloaders.append(DataLoader(self.val_source_dataset,
                                      batch_size=self.batch_size,
                                      pin_memory=True,
                                      shuffle=False,
                                      num_workers=self.num_workers))
        
        dataloaders.append(DataLoader(self.target_dataset,
                                      batch_size=self.batch_size,
                                      pin_memory=True,
                                      shuffle=False,
                                      num_workers=self.num_workers))
        
        return dataloaders
