import os
import ast
import torch
import numpy as np
from PIL import Image
from typing import Union, Dict, List, Tuple

from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from neural_net import *
from neural_net.sampler import ShuffleSampler
from neural_net.loss import GDiceLossV2
from neural_net.utils import *
from neural_net.transform import *
from neural_net.transf_learning import *

import segmentation_models_pytorch as smp
from neural_net.unet import UNet
from neural_net.attention_unet import AttentionUnet
from neural_net.canet_parts.networks.network import Comprehensive_Atten_Unet


import pytorch_lightning as pl

from config import *
import wandb

def c_binary_mean_iou(logits: torch.Tensor, targets: torch.Tensor, EPSILON = 1e-15) -> torch.Tensor:
    
    output = (logits > 0.5).int()
    intersection = (targets * output).sum()
    union = targets.sum() + output.sum() - intersection
    result = (intersection + EPSILON) / (union + EPSILON)

    return result

def max_min(t: torch.Tensor):
    t = t.view(t.size(0), -1)
    for i in range(t.shape[0]):
        t[i,:] = (t[i,:] - t[i,:].min()) / (t[i,:].max() - t[i,:].min())
    
    return t

    
class Satmodel(pl.LightningModule):
    def __init__(self, hparams, binary=True, discard_res=False):
        super().__init__()
        self.discard_res = discard_res
        self.hparams = hparams
        self.save_hyperparameters()
        self.key = hparams.key
        self.model = self.get_model()
        self.drop_last = False
                
        self.criterion = eval_object(hparams.criterion)
        self.binary = binary

        self.train_set = []
        self.validation_set = []
        self.test_set = []
        self.add_nbr = False

    def forward(self, batch):
        return self.model(batch)
    
    def get_model(self):
        model = eval_object(self.hparams.model)
        if type(model) == UNet:
            model.apply(initialize_weight)

    #         model = Comprehensive_Atten_Unet(in_ch=12, n_classes=1, im_size=(480, 480))
        return model
    

    def configure_optimizers(self):
        optimizer = eval_object(self.hparams.optimizer, params=self.model.parameters())
        self.optimizers = [optimizer]
        
        if self.hparams.get('scheduler'):
            scheduler = eval_object(self.hparams.scheduler, optimizer)
            
            if self.hparams.scheduler.name == optim.lr_scheduler.ReduceLROnPlateau:
                return {
                   'optimizer': optimizer,
                   'lr_scheduler': scheduler,
                   'monitor': 'val_loss'
               }
            return self.optimizers, [scheduler]
        return self.optimizers

    def setup(self, stage=0):
        ordered_keys = list(self.hparams.groups.keys())

        validation_fold_name = self.hparams.validation_dict[self.key]
        self.validation_set = self.hparams.groups[validation_fold_name]
        print(f'Test set is {self.key}, validation set is {validation_fold_name}. All the rest is training set.')

        for grp in self.hparams.groups:
            if grp == validation_fold_name or grp == self.key:
                continue
            else:
                self.train_set.extend(self.hparams.groups[grp])

        self.test_set = self.hparams.groups[self.key]
        print(f'Training set ({len(self.train_set)}): {str(self.train_set)}')
        print(f'Validation set ({len(self.validation_set)}): {str(self.validation_set)}')

    def train_dataloader(self) -> DataLoader:
        train_dataset = SatelliteDataset(folder_list=self.train_set,
                                         transform=self.hparams.train_transform,
                                         **self.hparams.dataset_specs)
        
        print(f'Train set dim: {len(train_dataset)}')
        train_sampler = ShuffleSampler(train_dataset, self.hparams["seed"])

        result = DataLoader(train_dataset, 
                            batch_size=self.hparams["batch_size"], 
                            sampler=train_sampler,
                            pin_memory=True,
                            drop_last=self.drop_last
                           )
        print(f"Train dataloader = {len(result)}")
                
        return result

    def val_dataloader(self):
        validation_dataset = SatelliteDataset(folder_list=self.validation_set,
                                              transform=self.hparams.test_transform,
                                              **self.hparams.dataset_specs)
                
        print(f'Validation set dim: {len(validation_dataset)}')
        validation_sampler = ShuffleSampler(validation_dataset, self.hparams["seed"])

        result = DataLoader(validation_dataset, 
                            batch_size=self.hparams["batch_size"],
                            sampler=validation_sampler,
                            pin_memory=True,
                            drop_last=self.drop_last
                           )
        print(f"Val dataloader = {len(result)}")
                
        return result

    # def test_dataloader(self):
    #     test_dataset = SatelliteDataset(self.config.master_folder, self.config.mask_intervals, self.config.mask_one_hot, self.config.height,
    #                                     self.config.width, self.config.product_list, self.config.mode, self.config.filter_validity_mask,
    #                                     self.config.test_transform, self.config.process_dict, self.config.satellite_csv_path,
    #                                     self.test_set,
    #                                     None, self.config.mask_filtering, only_burnt=True,
    #                                    )
    #
    #
    #     print('Test set dim: %d' % len(test_dataset))
    #     #         test_sampler = ShuffleSampler(test_dataset, config.seed)
    #
    #     return DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False, drop_last=self.drop_last)
    
    def training_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"]
        
        if self.binary: masks = (masks > .5).type_as(masks)
        logits = self.forward(images)
        loss = self.criterion(logits, masks)
        
        if self.binary:
            self.log('train_iou', binary_mean_iou(logits, masks))
        else:
            sq_err, counters = compute_squared_errors(logits, masks, len(self.mask_intervals))
            mse = np.true_divide(sq_err, counters, np.full(sq_err.shape, np.nan), where=counters != 0)
            self.log('train_rmse', np.sqrt(mse[-1]))
        

        self.log('lr', self._get_current_lr())
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_id):
        images = batch["image"]
        masks = batch["mask"]

        logits = self.forward(images)
        if self.binary: masks = (masks > .5).type_as(masks)
                                                     
        loss = self.criterion(logits, masks)
#         val_iou = binary_mean_iou(logits, masks)
        val_iou = binary_mean_iou(logits, masks)
            
        if not self.discard_res:
            self.log_images(images, logits, masks)

        self.log("val_loss", loss)
        self.log("val_iou", val_iou)
        
        return {'val_iou': val_iou}
    
    def log_images(self, images, logits, masks, log_dist=5):
        if self.binary:
            class_labels = {0: "background", 1: "fire"}

            logits_ = (torch.sigmoid(logits) > 0.5).cpu().detach().numpy().astype("float")
            masks_ = (masks > 0.5).cpu().detach().numpy().astype("float")
        else:
            class_labels = {0: "unburnt", 1: "1", 2: "2", 3: "3", 4: "4"}

            logits_ = logits.cpu().detach().numpy().astype("float")
            masks_ = masks.cpu().detach().numpy().astype("float")
#             result = np.zeros((mask.shape[0], mask.shape[1], len(self.mask_intervals)))
#             for idx in class_labels.keys():
#                 bin_mask = masks_ == float(idx)
#                 result[bin_mask, idx] = 1

        if self.trainer.current_epoch % log_dist == 0:
            for i in range(images.shape[0]):
                mask_img = wandb.Image(
                    images[i, [3,2,1], :, :]*2.5,
                    masks={
                        "predictions": {
                            "mask_data": logits_[i, 0, :, :],
                            "class_labels": class_labels,
                        },
                        "groud_truth": {
                            "mask_data": masks_[i, 0, :, :],
                            "class_labels": class_labels,
                        },
                    },
                )
                self.logger.experiment.log({"val_images": [mask_img]}, commit=False)

    def _get_current_lr(self):
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]  # type: ignore

        if torch.cuda.is_available(): return torch.Tensor([lr])[0].cuda()
        return torch.Tensor([lr])[0]

    def validation_epoch_end(self, outputs):
        self.log("epoch", self.trainer.current_epoch)
        avg_val_iou = find_average(outputs, "val_iou")

        self.log("val_iou", avg_val_iou)
        return

class Double_Satmodel(Satmodel):
    def __init__(self, hparams, binary=False, discard_res=False):
        super().__init__(hparams, binary=binary, discard_res=discard_res)
        
        if self.binary:
            print(f'    Iteration step 1/2 - Binary network training...  (test on {self.key} fold)\n')
        else:
            print(f'    Iteration step 2/2 - Regression network training...  (test on {self.key} fold)\n')
            self.criterion = eval_object(hparams.regr_criterion)
        self.set_model()
    
    def set_model(self):
        if self.binary:
            print("\n### Initializing model weights ###\n")
            self.model.apply(initialize_weight)
            self.model.unfreeze_binary_unet()
            self.model.freeze_regression_unet()
        else:
            self.model.freeze_binary_unet()
            self.model.unfreeze_regression_unet()
    
    def forward(self, batch):
        if self.binary:
            self.model.regression_unet.eval()
            self.model.freeze_regression_unet()
            return self.model(batch)[0]
        else:
            self.model.binary_unet.eval()
            self.model.freeze_binary_unet()
            return self.model(batch)[1]
#             mytype = torch.float32

    def configure_optimizers(self):
        if self.binary:
            optimizer = eval_object(self.hparams.optimizer, params=self.model.binary_unet.parameters())
        else:
            optimizer = eval_object(self.hparams.optimizer, params=self.model.parameters())
        self.optimizers = [optimizer]
        
        if self.hparams.get('scheduler'):
            scheduler = eval(self.hparams.scheduler.pop('name'))(optimizer, **self.hparams.scheduler)
            
            if self.hparams.scheduler.name == optim.lr_scheduler.ReduceLROnPlateau:
                return {
                   'optimizer': optimizer,
                   'lr_scheduler': scheduler,
                   'monitor': 'val_loss'
               }
            return self.optimizers, [scheduler]
        return self.optimizers
    
#     def training_step(self, batch, batch_idx):
#         images = batch["image"]
#         masks = batch["mask"]
        
#         if self.binary:
#             masks = (masks > .5).type_as(masks)
            
#             self.model.regression_unet.eval()
#             self.model.freeze_regression_unet()
#             logits = self.forward(images)[0]
#         else:
#             self.model.binary_unet.eval()
#             self.model.freeze_binary_unet()
#             logits = self.forward(images)[1]
        
#         loss = self.criterion(logits, masks)
#         train_iou = binary_mean_iou(logits, masks)

#         self.log('lr', self._get_current_lr())
#         self.log('loss', loss)
#         self.log('train_iou', train_iou)
#         return loss

#     def validation_step(self, batch, batch_id):
#         images = batch["image"]
#         masks = batch["mask"]
        
#         if self.binary:
#             masks = (masks > .5).type_as(masks)
            
#             self.model.regression_unet.eval()
#             self.model.freeze_regression_unet()
#             logits = self.forward(images)[0]
#         else:
#             self.model.binary_unet.eval()
#             self.model.freeze_binary_unet()
#             logits = self.forward(images)[1]
                                                             
#         loss = self.criterion(logits, masks)
# #         val_iou = binary_mean_iou(logits, masks)
#         val_iou = binary_mean_iou(logits, masks)
        
#         logits_ = (torch.sigmoid(logits) > 0.5).cpu().detach().numpy().astype("float")
#         masks_ = (masks > 0.5).cpu().detach().numpy().astype("float")
            
#         class_labels = {0: "background", 1: "fire"}
#         if not self.discard_res:
#             if self.trainer.current_epoch % 5 == 0:
#                 for i in range(images.shape[0]):
#                     mask_img = wandb.Image(
#                         images[i, [3,2,1], :, :]*2.5,
#                         masks={
#                             "predictions": {
#                                 "mask_data": logits_[i, 0, :, :],
#                                 "class_labels": class_labels,
#                             },
#                             "groud_truth": {
#                                 "mask_data": masks_[i, 0, :, :],
#                                 "class_labels": class_labels,
#                             },
#                         },
#                     )
#                     self.logger.experiment.log({"val_images": [mask_img]}, commit=False)


#         self.log("val_loss", loss)
#         self.log("val_iou", val_iou)
        
#         return {'val_iou': val_iou}


def compute_train_test_folders(master_folder, test_prefix, ignore_list=None):
    test_set = set()
    train_set = set()

    # creates a 'test_set' and 'train_set' lists with the folder names
    for dirname in os.listdir(master_folder):
        is_test = False
        if ignore_list is not None and dirname in ignore_list:
            continue
        for prefix in test_prefix:
            if dirname.startswith(prefix):
                is_test = True
                test_set.add(dirname)
                break

        if not is_test:
            train_set.add(dirname)

    return train_set, test_set