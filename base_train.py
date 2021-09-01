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
from neural_net.unet import UNet, ConcatenatedUNet
from neural_net.pspnet import PSPNet
from neural_net.nested_unet import NestedUNet, ConcatenatedNestedUNet
from neural_net.segnet import SegNet, ConcatenatedSegNet
from neural_net.sampler import ShuffleSampler
from neural_net.loss import GDiceLossV2
from neural_net.utils import *
from neural_net.transform import *
from neural_net.transf_learning import *

import segmentation_models_pytorch as smp
import pytorch_lightning as pl

from config import *
import wandb

def c_binary_mean_iou(logits: torch.Tensor, targets: torch.Tensor, EPSILON = 1e-15) -> torch.Tensor:
#     logits = max_min(logits)
#     targets = max_min(targets)
    
    output = (logits > 0.5).int()
    
#     targets = targets.squeeze()
#     output = output.squeeze()

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
    def __init__(self, model, criterion, hparams, binary=True, freeze_lim=None, discard_res=False):
        super().__init__()
        self.discard_res = discard_res
        self.hparams = hparams
        self.save_hyperparameters()
        self.key = hparams["key"]
        self.model = model
        self.drop_last = False
        
#         self.freeze_lim = freeze_lim
#         if self.freeze_lim:
#             self.__freeze_model()
                
        self.criterion = criterion
        self.binary = binary

        self.train_set = []
        self.validation_set = []
        self.test_set = []
        self.add_nbr = True
        
#     def __freeze_model(self):
#         """Freeze model layers."""
# #         _layers = list(backbone.children())
# #         self.model = torch.nn.Sequential(*_layers)
#         freeze(module=self.model, n=self.freeze_lim)


    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        optimizer = self.hparams["optimizer"](self.model.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["wd"])
        self.optimizers = [optimizer]
        
        if self.hparams["scheduler_tuple"] is not None:
            scheduler = self.hparams["scheduler_tuple"][0](optimizer, **(self.hparams["scheduler_tuple"][1]))
            
            if self.hparams["scheduler_tuple"][0] == optim.lr_scheduler.ReduceLROnPlateau:
                return {
                   'optimizer': optimizer,
                   'lr_scheduler': scheduler,
                   'monitor': 'val_loss'
               }
            return self.optimizers, [scheduler]
        return self.optimizers

    def setup(self, stage=0):
        ordered_keys = list(self.hparams["groups"].keys())

        validation_fold_name = self.hparams["validation_dict"][self.key]
        self.validation_set = self.hparams["groups"][validation_fold_name]
        print(f'Test set is {self.key}, validation set is {validation_fold_name}. All the rest is training set.')

        for grp in self.hparams["groups"]:
            if grp == validation_fold_name or grp == self.key:
                continue
            else:
                self.train_set.extend(self.hparams["groups"][grp])

        self.test_set = self.hparams["groups"][self.key]
        print('Training set (%d): %s' % (len(self.train_set), str(self.train_set)))
        print('Validation set (%d): %s' % (len(self.validation_set), str(self.validation_set)))

    def train_dataloader(self) -> DataLoader:
        train_dataset = SatelliteDataset(self.hparams["master_folder"], self.hparams["mask_intervals"], self.hparams["mask_one_hot"], self.hparams["height"],
                                         self.hparams["width"], self.hparams["product_list"], self.hparams["mode"], self.hparams["filter_validity_mask"],
                                         self.hparams["train_transform"], self.hparams["process_dict"], self.hparams["satellite_csv_path"],
                                         self.train_set,
                                         None, self.hparams["mask_filtering"], self.hparams["only_burnt"],
                                        )
        print('Train set dim: %d' % len(train_dataset))
        train_sampler = ShuffleSampler(train_dataset, self.hparams["seed"])

        result = DataLoader(train_dataset, 
                            batch_size=self.hparams["batch_size"], 
                            sampler=train_sampler,
                            pin_memory=True,
                            drop_last=self.drop_last
                           )
        print("Train dataloader = ", len(result))
        
#         if self.add_nbr:
#             for data in result:
#                 images = data['image']
#                 c7 = images[:, 7, :, :]
#                 c11 = images[:, 11, :, :]
#                 nbr = ((c7-c11) / (c7+c11)).unsqueeze(1)
#                 data['image'] = torch.cat([images, nbr],dim=1)
                
        return result

    def val_dataloader(self):
        validation_dataset = SatelliteDataset(self.hparams["master_folder"], self.hparams["mask_intervals"], self.hparams["mask_one_hot"],
                                              self.hparams["height"],
                                              self.hparams["width"], self.hparams["product_list"], self.hparams["mode"],
                                              self.hparams["filter_validity_mask"],
                                              self.hparams["test_transform"], self.hparams["process_dict"], self.hparams["satellite_csv_path"],
                                              self.validation_set,
                                              None, self.hparams["mask_filtering"], self.hparams["only_burnt"],
                                            )
                
        print('Validation set dim: %d' % len(validation_dataset))
        validation_sampler = ShuffleSampler(validation_dataset, self.hparams["seed"])

        result = DataLoader(validation_dataset, 
                            batch_size=self.hparams["batch_size"],
                            sampler=validation_sampler,
                            pin_memory=True,
                            drop_last=self.drop_last
                           )
        print("Val dataloader = ", len(result))
        
#         if self.add_nbr:
#             for data in result:
#                 images = data['image']
#                 c7 = images[:, 7, :, :]
#                 c11 = images[:, 11, :, :]
#                 nbr = ((c7-c11) / (c7+c11)).unsqueeze(1)
#                 data['image'] = torch.cat([images, nbr],dim=1)
                
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

#     def train(self, mode=True):
#         super(Satmodel, self).train(mode=mode)
#         if self.freeze_lim:
#             freeze(module=self.model,
#                    n=self.freeze_lim)
    
    def training_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"]

#         if self.binary: masks = torch.maximum(masks, torch.ones_like(masks))
        
        if self.binary: masks = (masks > .5).type_as(masks)
        logits = self.forward(images)
        loss = self.criterion(logits, masks)
        train_iou = binary_mean_iou(logits, masks)

        self.log('lr', self._get_current_lr())
        self.log('loss', loss)
        self.log('train_iou', train_iou)
        return loss

    def validation_step(self, batch, batch_id):
        images = batch["image"]
        masks = batch["mask"]

        logits = self.forward(images)
        if self.binary: masks = (masks > .5).type_as(masks)
                                                     
        loss = self.criterion(logits, masks)
#         val_iou = binary_mean_iou(logits, masks)
        val_iou = binary_mean_iou(logits, masks)
        
        logits_ = (torch.sigmoid(logits) > 0.5).cpu().detach().numpy().astype("float")
        masks_ = (masks > 0.5).cpu().detach().numpy().astype("float")
            
        class_labels = {0: "background", 1: "fire"}
        if not self.discard_res:
            if self.trainer.current_epoch % 5 == 0:
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


        self.log("val_loss", loss)
        self.log("val_iou", val_iou)
        
        return {'val_iou': val_iou}

    def _get_current_lr(self):
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]  # type: ignore

        if torch.cuda.is_available(): return torch.Tensor([lr])[0].cuda()
        return torch.Tensor([lr])[0]

    def validation_epoch_end(self, outputs):
        self.log("epoch", self.trainer.current_epoch)
        avg_val_iou = find_average(outputs, "val_iou")

        self.log("val_iou", avg_val_iou)
        return


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


def launch_training(result_path, criterion, nn_model='unet', config=None, key='blue', classification=False, wb=True, pretraining=False):
    print('cuda version detected: %s' % str(torch.version.cuda))
    print('cudnn backend %s' % str(torch.backends.cudnn.version()))

    master_folder = sentinel_hub_selected_dir
    
    ordered_keys = list(config.groups.keys())

    validation_fold_name = config.validation_dict[key]
    validation_set = config.groups[validation_fold_name]
    print(f'Test set is {key}, validation set is {validation_fold_name}. All the rest is training set.')

    train_set = []
    assert validation_fold_name in config.groups
    assert key in config.groups
    assert validation_fold_name != key
    assert isinstance(validation_fold_name, str) and isinstance(key, str)
    for grp in config.groups:
        if grp == validation_fold_name or grp == key:
            continue
        else:
            train_set.extend(config.groups[grp])

    test_set = config.groups[key]
    print('Training set (%d): %s' % (len(train_set), str(train_set)))
    print('Validation set (%d): %s' % (len(validation_set), str(validation_set)))
    print('Test set (%d): %s' % (len(test_set), str(test_set)))
    assert len(test_set) > 0

    train_dataset = SatelliteDataset(config.master_folder, config.mask_intervals, config.mask_one_hot, config.height,
                                     config.width, config.product_list, config.mode, config.filter_validity_mask,
                                     config.train_transform, config.process_dict, config.satellite_csv_path, train_set,
                                     None, config.mask_filtering, config.only_burnt,
                                     )
    validation_dataset = SatelliteDataset(config.master_folder, config.mask_intervals, config.mask_one_hot, config.height,
                                          config.width, config.product_list, config.mode, config.filter_validity_mask,
                                          config.test_transform, config.process_dict, config.satellite_csv_path, validation_set,
                                          None, config.mask_filtering, config.only_burnt,
                                          )
    test_dataset = SatelliteDataset(config.master_folder, config.mask_intervals, config.mask_one_hot, config.height,
                                    config.width, config.product_list, config.mode, config.filter_validity_mask,
                                    config.test_transform, config.process_dict, config.satellite_csv_path, test_set,
                                    None, config.mask_filtering, only_burnt=True,
                                    )

    print('Train set dim: %d' % len(train_dataset))
    print('Validation set dim: %d' % len(validation_dataset))
    print('Test set dim: %d' % len(test_dataset))

    train_sampler = ShuffleSampler(train_dataset, config.seed)
    validation_sampler = ShuffleSampler(validation_dataset, config.seed)
    #         test_sampler = ShuffleSampler(test_dataset, config.seed)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, drop_last=False)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, sampler=validation_sampler,
                                   drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)


    if nn_model == 'unet':
        model = UNet(config.n_channels, len(config.mask_intervals) if classification else 1, act='relu')
#         model = smp.Unet(encoder_name='resnet18', classes=1, in_channels=12, encoder_weights='imagenet')
    elif nn_model == 'pspnet':
        model = PSPNet(output_type='regr', backend='resnet18', input_channels=config.n_channels)
    elif nn_model == 'nestedunet':
        model = NestedUNet(output_type='regr', n_channels=config.n_channels, regr_range=4, act='relu')
    elif nn_model == 'segunet':
        model = SegNet(config.n_channels, len(mask_intervals) if classification else 1)
    else:
        raise ValueError('Invalid model name')

    # initialize_weight(model, seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print('%d gpus' % torch.cuda.device_count())
        model = nn.DataParallel(model)
    model = model.to(device)
    
    if pretraining:
        model.load_state_dict(torch.load('../data/pretrained_weights/unet_landcover.pt'))
        print(f'Freezing up to layer {pretraining}/5\n')
        freeze(module=model, n=pretraining)
    else:
        model.apply(initialize_weight)
    
    if wb: 
        wandb.init(config=config)
        wandb.watch(model)

    # criterion = GDiceLossV2(apply_nonlin=nn.Softmax(dim=1)).to(device)
    if torch.cuda.is_available():
        criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.wd)

    scheduler = None
    if config.scheduler_tuple is not None:
        scheduler = config.scheduler_tuple[0](optimizer, **(config.scheduler_tuple[1]))

    train(model, criterion, optimizer, train_loader, test_loader, config.epochs, device=device, squeeze=False, classification=classification, scheduler=scheduler, wb=wb)

    torch.save(model.state_dict(), result_path)
    return

def launch_concatenated_training(test_prefix: list, epochs, batch_size, lr, wd, result_path, criterion, mask_one_hot, product_list, mode, height, width, process_dict, filter_validity_mask, n_channels, train_transform, test_transform, seed, mask_intervals=None, scheduler_class=None, nn_model='concat_unet', scheduler_args=None, ignore_list=None, mask_filtering=False, only_burned=False):
    print('cuda version detected: %s' % str(torch.version.cuda))
    print('cudnn backend %s' % str(torch.backends.cudnn.version()))

    if bool(scheduler_class is None) ^ bool(scheduler_args is None):
        raise ValueError('Invalid combination of scheduler_class and scheduler_args arguments')

    result_basepath, _ = os.path.split(result_path)
    if not os.path.isdir(result_basepath):
        os.makedirs(result_basepath)
        if not os.path.isdir(result_basepath):
            raise RuntimeError('Unable to create folder %s' % result_basepath)

    master_folder = sentinel_hub_selected_dir
    if mask_intervals is None:
        mask_intervals = [(0, 32), (33, 96), (97, 160), (161, 224), (225, 255)]

    model_name = model_names[list(models.values()).index(nn_model)]
    print(f"\nTraining of {model_name} (output in '{result_path}')\n")
    train_set, test_set = compute_train_test_folders(master_folder, test_prefix, ignore_list)
    print('Training set (%d): %s' % (len(train_set), str(train_set)))
    print('Test set (%d): %s' % (len(test_set), str(test_set)))
    assert len(test_set) > 0

    train_dataset = SatelliteDataset(master_folder, mask_intervals, mask_one_hot, height, width, product_list, mode, filter_validity_mask, train_transform, process_dict, satellite_csv_path, folder_list=train_set, mask_filtering=mask_filtering, only_burnt=only_burned)
    test_dataset = SatelliteDataset(master_folder, mask_intervals, mask_one_hot, height, width, product_list, mode, filter_validity_mask, test_transform, process_dict, satellite_csv_path, folder_list=test_set, mask_filtering=mask_filtering, only_burnt=only_burned)

    train_sampler = ShuffleSampler(train_dataset, seed=seed)
    test_sampler = ShuffleSampler(test_dataset, seed=seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False, sampler=test_sampler)

    if nn_model == 'concat_unet':
        model = ConcatenatedUNet(n_channels, act='relu')
    elif nn_model == 'concat_nest_unet':
        model = ConcatenatedNestedUNet(n_channels, act='relu')
    elif nn_model == 'concat_segunet':
        model = ConcatenatedSegNet(n_channels)
    initialize_weight(model, seed=seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print('%d gpus' % torch.cuda.device_count())
        model = nn.DataParallel(model)
    model = model.to(device)

    if torch.cuda.is_available():
        for idx, crit in enumerate(criterion):
            crit = crit.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = None
    if scheduler_class is not None and scheduler_args is not None:
        scheduler = scheduler_class(optimizer, **scheduler_args)

    train_concatenated_model(model, criterion, optimizer, train_loader, test_loader, epochs, scheduler=scheduler)
    torch.save(model.state_dict(), result_path)
    # model.module.state_dict() --> model.state_dict()
    return
