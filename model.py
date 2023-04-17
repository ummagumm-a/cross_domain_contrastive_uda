import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
import torchmetrics
import numpy as np
from sklearn.cluster import KMeans
import os
import plotly.express as px

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class UDAModel(pl.LightningModule):
    def __init__(self, feature_extractor, classification_head, n_classes, 
                       source_dataset, target_dataset, contrastive_only=False,
                       tau=1., b=0.75, lmbda=1.4, 
                       pseudo_filter_threshold=0.9, track_grad_norm=False,
                       explicit_negative_sampling_threshold=0.5, grad_clip=1.5,
                       batch_size=64, num_workers=48, pretrain_num_epochs=0,
                       negative_sampling=None, remove_mismatched=False,
                       total_epochs=None, class_names=None):
        super().__init__()
        self.save_hyperparameters(ignore=['feature_extractor', 'classification_head', 'source_dataset', 'target_dataset'])

        self.track_grad_norm = track_grad_norm
        self.automatic_optimization = not self.track_grad_norm
        self.grad_clip = grad_clip
        
        self.contrastive_only = contrastive_only
        assert not (contrastive_only and pretrain_num_epochs > 0)

        self.n_classes = n_classes
        # The last layer should be smth giving (N, out_features)
        self.feature_extractor = feature_extractor
        self.classification_head = classification_head
        self.classification_loss = nn.CrossEntropyLoss()
        
        self.tau = tau
        self.explicit_negative_sampling_threshold = explicit_negative_sampling_threshold
        self.pseudo_filter_threshold = pseudo_filter_threshold
        self.b = b
        self.lmbda = lmbda
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pretrain_num_epochs = pretrain_num_epochs
        self.negative_sampling = self.check_negative_sampling(negative_sampling)
        self.remove_mismatched = remove_mismatched
        
        self.total_epochs = self.check_total_epochs(total_epochs)
        self.class_names = self.check_class_names(class_names)
        
        self.train_source_dataset, self.val_source_dataset = source_dataset
        self.train_target_dataset, self.val_target_dataset = target_dataset
        self.valid_target_samples = list(range(len(self.train_target_dataset)))
        
        self.accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=n_classes, average=None)
        #self.precision_metric = torchmetrics.Precision(num_classes=n_classes, average=None)
        #self.recall_metric = torchmetrics.Recall(num_classes=n_classes, average=None)
        
    def check_class_names(self, class_names):
        if class_names is None:
            class_names = np.arange(self.n_classes)
            
        return class_names
    
    def check_negative_sampling(self, negative_sampling):
        """
        Checks correctness of 'negative_sampling' parameter.
        """
        
        negative_sampling_options = [None, 'soft', 'hard', 'random']
        if negative_sampling in negative_sampling_options:
            return negative_sampling
        else:
            raise Exception(f"Wrong negative sampling option {negative_sampling}. Available options are: {negative_sampling_options}.")
    
    def check_total_epochs(self, total_epochs):
        if total_epochs is None:
            raise Exception("total_epochs should be not 'None'")
        elif not isinstance(total_epochs, int):
            raise Exception(f"total_epochs should be int, but is '{type(total_epochs)}'")
        else:
            return total_epochs
        
    def __call__(self, inp, domain_label):
        features = self.feature_extractor(inp, domain_label)
        
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
    
    def calculate_class_centers(self):
        dataloader = self.dataloader_source_for_centroids()

        accum = torch.zeros((self.n_classes, self.feature_extractor.in_features), device=self.device)

        for x, y, y_real in dataloader:
            assert torch.all(y == y_real), 'y != y_real in calculate_class_centers'
            x = x.to(self.device)
            y = y.to(self.device)
            features = self.feature_extractor(x, 0)

            y = y.view(y.size(0), 1).expand(-1, features.size(1))

            batch_class_means = torch.zeros_like(accum, dtype=torch.float32, device=self.device)\
                                     .scatter_add_(0, y, features)
            accum += batch_class_means

            

        labels_count = torch.from_numpy(np.bincount(self.train_source_dataset.labels, minlength=self.n_classes))
        labels_count = labels_count.to(self.device)
        initial_centers = (accum / labels_count.float().unsqueeze(1)).cpu().numpy()
        initial_centers /= np.linalg.norm(initial_centers, axis=1)[:, np.newaxis]

        return initial_centers

    def get_target_features(self):
        dataloader = self.dataloader_target_for_clustering()
        all_features = []
        for x, _, _ in dataloader:
            x = x.to(self.device)
            features = self.feature_extractor(x, 1)
            features = features.cpu().numpy()
            all_features.append(features)
        all_features = np.vstack(all_features)

        return all_features

    def visualize_pseudo_labeling(self):
        real_labels = self.train_target_dataset.real_labels
        labels = self.train_target_dataset.labels

        pairs = np.vstack((real_labels, labels))
        unique_pairs, counts = np.unique(pairs, axis=1, return_counts=True)
        fig = px.scatter_3d(x=unique_pairs[0, :], y=unique_pairs[1, :], z=counts)
        fig.show()
        
    def class_analysis(self, dataset):
        labels = dataset.labels
        real_labels = dataset.real_labels
#        self.logger.experiment.add_scalar(f'Mislabeled fraction', np.mean(labels != real_labels), self.current_epoch)

        # find unassigned labels
        # unassigned_labels = set(range(31)).difference(np.unique(labels).tolist())
        # self.logger.experiment.add_text('unassigned labels', 
        #                                str(unassigned_labels), self.current_epoch)

        # for each class find the distribution of assigned classes
        # label_stats = {}
        # for i in range(31):
        #     ilabels = labels[real_labels == i]
        #     if len(ilabels) == 0:
        #         continue
        #     self.logger.experiment.add_histogram(f'class {i} assigned to:', 
        #                                          ilabels, self.current_epoch,
        #                                          bins=31)
        #     self.logger.experiment.add_scalar(f'Class {i} mislabeled fraction', (ilabels != i).sum() / len(ilabels), self.current_epoch)
            

    def filter_after_cluster(self, features, centers):
        # Normalize the centers so they are on the unit-sphere
        centers /= np.linalg.norm(centers, axis=1)[:, np.newaxis]
        # Calculate similarities between target features and cluster centers
        sims = features @ centers.T
        # Similarity with the closest cluster center
        max_sims = sims.max(axis=1)
        assert len(max_sims) == len(features)
        # Take only samples which are 'close enough' to class centroids
        mask = max_sims > self.pseudo_filter_threshold
#        self.logger.experiment.add_scalar(f'Close enough targets', np.sum(mask), self.current_epoch)

        # If the setting is to not allow mislabeled samples in training dataset
        if self.remove_mismatched:
            mismatch_mask = self.train_target_dataset.real_labels == self.train_target_dataset.labels
            
            mask &= mismatch_mask

        # Save indices of valid samples - they are 'close enough' (and optionally, are not mislabeled)
        # This list will be used in 'train_dataloader' function
        self.valid_target_samples = np.arange(len(self.train_target_dataset))[mask]
        np.random.shuffle(self.valid_target_samples)

#        self.logger.experiment.add_scalar(f'Remaining targets', np.sum(mask), self.current_epoch)

    def on_train_epoch_start(self):
#        if self.pretrain_num_epochs <= self.current_epoch:
        logger.info("Do pseudo-labeling")
        self.feature_extractor.eval()

        # Define a clusterizer
        clusterizer = KMeans(n_clusters=self.n_classes, n_init=1)
        with torch.no_grad():
            # Calculate class centroids on the source domain
            initial_centers = self.calculate_class_centers()
            # Initialize cluster positions
            clusterizer.set_params(init=initial_centers)
            # Collect features for target samples
            all_features = self.get_target_features()
            # Do clusterization
            clusterizer.fit(all_features)
            # Update target labels
            self.train_target_dataset.labels = clusterizer.labels_
            # Log stats about the results of pseudo-labeling
            self.class_analysis(self.train_target_dataset)

            # Filter out unwanted samples from training data. 
            self.filter_after_cluster(all_features, clusterizer.cluster_centers_)
#            self.log('unique labels', len(np.unique(self.train_target_dataset.labels)))

        self.feature_extractor.train()

    def classification_step(self, batch):
        x, y, y_real = batch
        assert torch.all(y == y_real), "y != y_real in classification step"
        
        features = self.feature_extractor(x, 0)
        if self.contrastive_only:
            # In contrastive only mode I want only contrastive loss to have impact on feature encoder weights.
            # Therefore, I detach 'features' from the computation graph, so classification loss impacts only classification head.
            logger.info("contrastive_only - Detach features")
            features = features.detach()

        pred = self.classification_head(features)

        probs = F.softmax(pred.detach(), dim=1)
        probs = probs.max(dim=1).values.mean().item()

#        self.log('mean_pred_prob', probs, on_epoch=True, on_step=False)

        class_loss = self.classification_loss(pred, y)
        
        return class_loss
    
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
                #self.logger.experiment.add_histogram('source_anchor, false negative sims', 
                #                                     false_negatives_sims, 
                #                                     self.global_step)
                self.logger.experiment.add_scalar('source_anchor, false negative sims, max', 
                                                  false_negatives_sims.max(), 
                                                  self.global_step)
                self.logger.experiment.add_scalar('source_anchor, false negative sims, mean', 
                                                  false_negatives_sims.mean(), 
                                                  self.global_step)
                #self.logger.experiment.add_histogram(f'source_anchor, false negative sims, anchor {y}', 
                #                                     false_negatives_sims, 
                #                                     self.global_step)
            
            true_negatives_sims = neg_x[neg_y_real != y]
            if len(true_negatives_sims) != 0:
                #self.logger.experiment.add_histogram('source_anchor, true negative sims', 
                #                                     true_negatives_sims, 
                #                                     self.global_step)
                self.logger.experiment.add_scalar('source_anchor, true negative sims, max', 
                                                  true_negatives_sims.max(), 
                                                  self.global_step)
                self.logger.experiment.add_scalar('source_anchor, true negative sims, mean', 
                                                  true_negatives_sims.mean(), 
                                                  self.global_step)
                #self.logger.experiment.add_histogram(f'source_anchor, true negative sims, anchor {y}', 
                #                                     true_negatives_sims, 
                #                                     self.global_step)
                
            false_positives_sims = pos_x[pos_y_real != y]
            if len(false_positives_sims) != 0:
                #self.logger.experiment.add_histogram('source_anchor, false positive sims', 
                #                                     false_positives_sims, 
                #                                     self.global_step)
                self.logger.experiment.add_scalar('source_anchor, false positive sims, max', 
                                                  false_positives_sims.max(), 
                                                  self.global_step)
                self.logger.experiment.add_scalar('source_anchor, false positive sims, mean', 
                                                  false_positives_sims.mean(), 
                                                  self.global_step)
                #self.logger.experiment.add_histogram(f'source_anchor, false positive sims, anchor {y}', 
                #                                     false_positives_sims, 
                #                                     self.global_step)
            
            true_positives_sims = pos_x[pos_y_real == y]
            if len(true_positives_sims) != 0:
                #self.logger.experiment.add_histogram('source_anchor, true positive sims', 
                #                                     true_positives_sims, 
                #                                     self.global_step)
                self.logger.experiment.add_scalar('source_anchor, true positive sims, max', 
                                                  true_positives_sims.max(), 
                                                  self.global_step)
                self.logger.experiment.add_scalar('source_anchor, true positive sims, mean', 
                                                  true_positives_sims.mean(), 
                                                  self.global_step)
                #self.logger.experiment.add_histogram(f'source_anchor, true positive sims, anchor {y}', 
                #                                     true_positives_sims, 
                #                                     self.global_step)
            
            
        elif anchor_type == 'target':
            # The anchor is from target, therefore its label may be incorrect.
            # Having a source batch, we'll check how much samples are of the same actual class
            # as target anchor
            x, _, y_real = anchor
            neg_x, neg_y, _ = negatives
            pos_x, pos_y, _ = positives
            
            false_negatives_sims = neg_x[neg_y == y_real]
            if len(false_negatives_sims) != 0:
                #self.logger.experiment.add_histogram('target_anchor, false negative sims', 
                #                                     false_negatives_sims, 
                #                                     self.global_step)
                self.logger.experiment.add_scalar('target_anchor, false negative sims, max', 
                                                  false_negatives_sims.max(), 
                                                  self.global_step)
                self.logger.experiment.add_scalar('target_anchor, false negative sims, mean', 
                                                  false_negatives_sims.mean(), 
                                                  self.global_step)
                #self.logger.experiment.add_histogram(f'target_anchor, false negative sims, anchor {y_real}', 
                #                                     false_negatives_sims, 
                #                                     self.global_step)
            
            true_negatives_sims = neg_x[neg_y != y_real]
            if len(true_negatives_sims) != 0:
                #self.logger.experiment.add_histogram('target_anchor, true negative sims', 
                #                                     true_negatives_sims, 
                #                                     self.global_step)
                self.logger.experiment.add_scalar('target_anchor, true negative sims, max', 
                                                  true_negatives_sims.max(), 
                                                  self.global_step)
                self.logger.experiment.add_scalar('target_anchor, true negative sims, mean', 
                                                  true_negatives_sims.mean(), 
                                                  self.global_step)
                #self.logger.experiment.add_histogram(f'target_anchor, true negative sims, anchor {y_real}', 
                #                                     true_negatives_sims, 
                #                                     self.global_step)
                
            false_positives_sims = pos_x[pos_y != y_real]
            if len(false_positives_sims) != 0:
                #self.logger.experiment.add_histogram('target_anchor, false positive sims', 
                #                                     false_positives_sims, 
                #                                     self.global_step)
                self.logger.experiment.add_scalar('target_anchor, false positive sims, max', 
                                                  false_positives_sims.max(), 
                                                  self.global_step)
                self.logger.experiment.add_scalar('target_anchor, false positive sims, mean', 
                                                  false_positives_sims.mean(), 
                                                  self.global_step)
                #self.logger.experiment.add_histogram(f'target_anchor, false positive sims, anchor {y_real}', 
                #                                     false_positives_sims, 
                #                                     self.global_step)
            
            true_positives_sims = pos_x[pos_y == y_real]
            if len(true_positives_sims) != 0:
                #self.logger.experiment.add_histogram('target_anchor, true positive sims', 
                #                                     true_positives_sims, 
                #                                     self.global_step)
                self.logger.experiment.add_scalar('target_anchor, true positive sims, max', 
                                                  true_positives_sims.max(), 
                                                  self.global_step)
                self.logger.experiment.add_scalar('target_anchor, true positive sims, mean', 
                                                  true_positives_sims.mean(), 
                                                  self.global_step)
                #self.logger.experiment.add_histogram(f'target_anchor, true positive sims, anchor {y_real}', 
                #                                     true_positives_sims, 
                #                                     self.global_step)
                
        else:
            raise Exception(f"Wrong anchor type: {anchor_type}")
            
    def _filter_negative_samples(self, negatives_sims):
        """
        Filter out negative samples according to negative sampling strategy and threshold.
        """
        
        if self.negative_sampling is None:
            return negatives_sims

        elif self.negative_sampling == 'soft':
            indices = negatives_sims < self.explicit_negative_sampling_threshold
            return negatives_sims[indices]

        elif self.negative_sampling == 'hard':
            indices = negatives_sims > self.explicit_negative_sampling_threshold
            return negatives_sims[indices]

        elif self.negative_sampling == 'random':
            num_negs = torch.sum(negatives_sims > self.explicit_negative_sampling_threshold)
            indices = torch.randperm(len(negatives_sims))[:num_negs]
            
            return negatives_sims[indices]
        else:
            raise Exception(f"Wrong negative sampling option {negative_sampling}. Available options are: {negative_sampling_options}.")
    
    def contrastive_step(self, anchors_batch, other_batch, anchor_type):
        if anchor_type == 'target':
            other_dataset, other_class_indices, other_index = self.train_source_dataset, self.source_class_indices, 0
        elif anchor_type == 'source':
            other_dataset, other_class_indices, other_index = self.train_target_dataset, self.target_class_indices, 1
        else:
            raise Exception("Wrong anchor_type in contrastive_step")

        other_feat, other_y, _ = other_batch
        anchor_feat, anchor_y, _ = anchors_batch

        contrastive_loss = 0
        for i in range(self.n_classes):
            anchor_same_class_indices = anchor_y == i
            feat_i = anchor_feat[anchor_same_class_indices]
        
        #for feat, y, y_real in zip(*anchors_batch):
            same_class_indices = other_y == i
            if not same_class_indices.any():
                continue

            positives = other_feat[same_class_indices]
            negatives = other_feat[~same_class_indices]

            # positives_sims = positives @ feat
            positives_sims = positives @ torch.t(feat_i)
            positives_exp = torch.exp(positives_sims / self.tau)
            # negatives_sims = negatives @ feat
            negatives_sims = negatives @ torch.t(feat_i)
#             self.analyze_negative_samples((feat, y, y_real), 
#                                           (negatives_sims, other_y[~same_class_indices], other_y_real[~same_class_indices]),
#                                           (positives_sims, other_y[same_class_indices], other_y_real[same_class_indices]),
#                                           anchor_type)
            
            # Explicit negative sampling
            negatives_sims = self._filter_negative_samples(negatives_sims)
            negatives_exp = torch.exp(negatives_sims / self.tau)

            denominator = negatives_exp.sum(0) + positives_exp.sum(0)
            assert len(denominator) == torch.sum(anchor_same_class_indices), f"{len(denominator)}  {len(anchor_same_class_indices)}"

            accum = 0
            for global_pos_x, global_pos_y, _ in self.dataloader_by_class(
                    other_dataset, other_class_indices[i]):
                assert torch.all(global_pos_y == i)
                global_pos_x = global_pos_x.to(self.device)
                global_pos_y = global_pos_y.to(self.device)

                global_pos_feat = self.feature_extractor(global_pos_x, other_index)
                global_pos_sims = global_pos_feat @ torch.t(feat_i)
                global_pos_exp = torch.exp(global_pos_sims / self.tau)

                logit = global_pos_exp / denominator.unsqueeze(0)
                accum += torch.log(logit).sum(0)

            assert len(accum) == torch.sum(anchor_same_class_indices), f"{len(accum)}  {len(anchor_same_class_indices)}"

            accum /= len(self.target_class_indices[i]) 
            
            if not accum.isnan().all():
                contrastive_loss += accum.nansum()
            
        return -contrastive_loss
    
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

    def compute_losses(self, batch):
        source_for_classification, (source_x, source_y, source_y_real) = self.split_batch_in_two(batch['source'])
        assert torch.all(source_y == source_y_real), "y != y_real in training step"

        classification_loss = self.classification_step(source_for_classification)
        
        if self.pretrain_num_epochs <= self.current_epoch:
            logger.info("Do contrastive step")
            target_x, target_y, target_y_real = batch['target']

            source_features = self.feature_extractor(source_x, 0)
            target_features = self.feature_extractor(target_x, 1)
            contrastive_target_anchor = self.contrastive_step(
                (target_features, target_y, target_y_real), 
                (source_features, source_y, source_y_real), 
                'target',
            )
            contrastive_source_anchor = self.contrastive_step(
                (source_features, source_y, source_y_real), 
                (target_features, target_y, target_y_real), 
                'source'
            )
            contrastive_loss = contrastive_target_anchor + contrastive_source_anchor

        else:
            logger.info("Don't do contrastive step")
            contrastive_loss = 0

        return classification_loss, contrastive_loss

    def extract_gradients(self):
        grads = []
        for i, (name, param) in enumerate(self.feature_extractor.named_parameters()):
            grads.append(param.grad)

        for i, (name, param) in enumerate(self.classification_head.named_parameters()):
            grads.append(param.grad)

        return grads

    def sum_gradients(self, grads1, grads2):
        grads = []
        for grad1, grad2 in zip(grads1, grads2):
            if grad1 is None and grad2 is None:
                grads.append(None)
            # why there can be a None value in one grad but not in the other?
            # These are the gradients of DSBN. Classification doesn't use target-specific BN, so no gradients will be there.
            elif grad1 is None or grad2 is None:
                if grad1 is not None:
                    grads.append(grad1)   
                elif grad2 is not None:
                    grads.append(grad2)   

            else:
                grads.append(grad1 + grad2)

        return grads

    def set_gradients(self, grads):
        for i, (param, grad) in enumerate(zip(self.feature_extractor.parameters(), grads)):
            param.grad = grad

        for param, grad in zip(self.classification_head.parameters(), grads[i + 1:]):
            param.grad = grad

    def log_grad_norm(self, grads, name):
        nonnan_grads = list(map(lambda x: x.flatten(), filter(lambda x: x is not None, grads)))

        grad_norm = torch.hstack(nonnan_grads).norm(2).detach()
        self.logger.experiment.add_scalar(f'{name} loss grad norm', grad_norm, self.global_step)

    def training_step(self, batch):
        classification_loss, contrastive_loss = self.compute_losses(batch)
        contrastive_loss *= self.lmbda
        train_loss = classification_loss + contrastive_loss

        if self.track_grad_norm:
            opt = self.optimizers()
            sch = self.lr_schedulers()

            # Calculate gradient norms for each loss separately
            opt.zero_grad()
            if float != type(contrastive_loss):
                self.manual_backward(contrastive_loss, retain_graph=True)
                with torch.no_grad():
                    grads2 = self.extract_gradients()
#                    self.log_grad_norm(grads2, "Contrastive")
#            else:
#                self.logger.experiment.add_scalar(f'Contrastive loss grad norm', 0, self.global_step)

            opt.zero_grad()
            self.manual_backward(classification_loss, retain_graph=True)
            with torch.no_grad():
                grads1 = self.extract_gradients()
#                self.log_grad_norm(grads1, "Classification")

    #        sum_grads = self.sum_gradients(grads1, grads2)
    #        self.log_grad_norm(grads2, "Total Test")

            # Calculate gradient on a combined loss
            opt.zero_grad()
            self.manual_backward(train_loss)
            with torch.no_grad():
                grads = self.extract_gradients()
#                self.log_grad_norm(grads, "Total")

            # Make a step
            self.clip_gradients(opt, gradient_clip_val=self.grad_clip, gradient_clip_algorithm="norm")
            opt.step()

            # Learning rate scheduler step
            if self.trainer.is_last_batch:
                sch.step()

#        self.log_dict({
#            "classification_loss": classification_loss,
#            "contrastive_loss": contrastive_loss,
#            "train_loss": train_loss
#        }, on_epoch=True, on_step=False)
                
        return train_loss
    
    def val_metrics(self, pred, y, prefix):
        # TODO: for optimization may need to first calculate tp, tn, fp, fn
        accuracy = self.accuracy_metric(pred, y)
        #precision = self.precision_metric(pred, y)
        #recall = self.recall_metric(pred, y)
        
        log_dict = {}
        # for i, class_name in enumerate(self.class_names):
        #     log_dict[f'{prefix}_accuracy_{class_name}'] = accuracy[i]
        #     log_dict[f'{prefix}_precision_{class_name}'] = precision[i]
        #     log_dict[f'{prefix}_recall_{class_name}'] = recall[i]

        log_dict[f'{prefix}_accuracy'] = torch.nanmean(accuracy)
        #log_dict[f'{prefix}_precision'] = torch.nanmean(precision)
        #log_dict[f'{prefix}_recall'] = torch.nanmean(recall)
            
        return log_dict

    def validation_step(self, batch, batch_idx, dataloader_idx):
        def val_helper(batch, domain_name, dsbn_index):
            x, y, y_real = batch
            assert torch.all(y == y_real), f"y != y_real in validation step ({dataloader_idx})"

            pred = self(x, dsbn_index)
            loss = self.classification_loss(pred, y)

#            self.log(f"{domain_name}_val_loss", loss, on_epoch=True, on_step=False, add_dataloader_idx=False)
            self.log_dict(self.val_metrics(pred, y, domain_name), on_epoch=True, on_step=False, add_dataloader_idx=False)
            
        # Evaluate source accuracy
        if dataloader_idx == 0:
            val_helper(batch, 'source', 0)

        # Evaluate target accuracy
        elif dataloader_idx == 1:
            # When the model pretrains or we study no_adaptation setting, 
            # weights of target-specific batch norms are not trained because no samples are passed through them
            # Also, it must be evident that 'no_adaptation' means no adaptation techniques at all, so we should remove DSBN factor from model evaluation.
            dsbn_index = int(self.pretrain_num_epochs <= self.current_epoch)
            logger.info(f"Use dsbn_index {dsbn_index} on epoch {self.current_epoch} because model is pretrained for {self.pretrain_num_epochs} epochs")

            val_helper(batch, 'target', dsbn_index)

        elif dataloader_idx not in [0, 1]:
            raise Exception(f'weird dataloader num: {dataloader_idx}')
        
    def dataloader_target_for_clustering(self):
        return DataLoader(self.train_target_dataset,
                          batch_size=self.batch_size,
                          pin_memory=True,
                          shuffle=False,
                          num_workers=self.num_workers)
    
    def dataloader_source_for_centroids(self):
        return DataLoader(self.train_source_dataset,
                          batch_size=self.batch_size,
                          pin_memory=True,
                          shuffle=False,
                          num_workers=self.num_workers)

    def dataloader_by_class(self, dataset, indices):
        np.random.shuffle(indices)

        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          pin_memory=True,
                          sampler=SubsetRandomSampler(indices),
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
                            DataLoader(self.train_target_dataset,
                                       batch_size=self.batch_size,
                                       pin_memory=True,
                                       sampler=SubsetRandomSampler(self.valid_target_samples),
                                       num_workers=self.num_workers)
        
        return dataloaders

    def val_dataloader(self):
        dataloaders = []
        dataloaders.append(DataLoader(self.val_source_dataset,
                                      batch_size=self.batch_size,
                                      pin_memory=True,
                                      shuffle=False,
                                      num_workers=self.num_workers))
        
        dataloaders.append(DataLoader(self.val_target_dataset,
                                      batch_size=self.batch_size,
                                      pin_memory=True,
                                      shuffle=False,
                                      num_workers=self.num_workers))
        
        return dataloaders
