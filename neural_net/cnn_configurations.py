import os

from config import *
from neural_net.cross_validator import ConcatenatedCrossValidator, CrossValidator, GradcamCrossValidator
from neural_net import *
from neural_net.transform import *
from neural_net.loss import IndexLoss, IoULoss, FuzzyIoULoss, GDiceLossV2, ComboLoss, softIoULoss, F1MSE
from neural_net.performance_storage import AccuracyBinStorage, AccuracyAllStorage, AccuracySingleRegrStorage

from neural_net.unet import ConcatenatedUNet, UNet, MixedNet
from neural_net.pspnet import PSPNet
from neural_net.nested_unet import NestedUNet, ConcatenatedNestedUNet
from neural_net.segnet import SegNet, ConcatenatedSegNet

from torch import nn, optim
from collections import OrderedDict
import pandas as pd
import albumentations as albu


class TrainingConfig:
    """
    Defines a training configuration for the model.
    """
    def __init__(self, key='blue', batch_size=8, n_channels=12, mask_one_hot=False, classification=False,
                 epochs=50, lr=1e-4, wd=0, product_list=['sentinel2'], mode='post',
                 mask_intervals=[(0, 36), (37, 96), (97, 160), (161, 224), (225, 255)],
                 height=480, width=480, filter_validity_mask=True, only_burnt=True, mask_filtering=False, imagenet=False, seed=47,
                 patience=5, tol=1e-2, optimizer=optim.Adam, scheduler_tuple= [optim.lr_scheduler.ReduceLROnPlateau, {'factor': 0.25, 'patience': 2}]):
                 
        # [optim.lr_scheduler.CosineAnnealingWarmRestarts, {'T_0': 10, 'T_mult': 2}]

        # Set hyperparameters
        self.classification = classification
        self.key = key
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.mask_one_hot = mask_one_hot
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.product_list = product_list
        self.mode = mode
        self.mask_intervals = mask_intervals
        self.height = height
        self.width = width
        self.filter_validity_mask = filter_validity_mask
        self.only_burnt = only_burnt
        self.mask_filtering = mask_filtering
        self.seed = seed
        self.patience = patience
        self.tol = tol
        self.optimizer = optimizer
        self.scheduler_tuple = scheduler_tuple
        self.binary = True

        # Input dataset
        self.master_folder = sentinel_hub_selected_dir
        self.satellite_csv_path = satellite_csv_path

        # Satellite image bandwitdths
        all_bands_selector = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.process_dict = {'sentinel2': all_bands_selector}

        # Test set fold (key) -> validation fold (value)
        self.validation_dict = {
            'blue': 'fucsia',
            'brown': 'fucsia',
            'fucsia': 'green',
            'green': 'fucsia',
            'orange': 'fucsia',
            'red': 'fucsia',
            'yellow': 'fucsia'
        }

        self.groups = self.read_groups()
        
        if imagenet:
            imgnet_mean = [1 for i in range(n_channels)]
            imgnet_std = [1 for i in range(n_channels)]
            imgnet_mean[1:4] = (0.406, 0.456, 0.485)  # rgb are 3,2,1
            imgnet_std[1:4] = (0.225, 0.224, 0.229)
            mn = imgnet_mean
            std = imgnet_std
        else:
            mn = (0.5,) * n_channels
            std = (0.5,) * n_channels
            
        # Dataset augmentation and normalization
        self.train_transform = transforms.Compose([
            RandomRotate(0.5, 50, seed=seed),
            RandomVerticalFlip(0.5, seed=seed),
            RandomHorizontalFlip(0.5, seed=seed),
            RandomShear(0.5, 20, seed=seed),
            ToTensor(round_mask=True),
#             Resize(800),
            Normalize(mn, std)
        ])
        self.test_transform = transforms.Compose([
            ToTensor(round_mask=True),
#             Resize(800),
            Normalize(mn, std)
        ])
    
    def update_transforms(self):
        print("Using pretraining for imagenet weights\n")
        
        imgnet_mean = [1 for i in range(self.n_channels)]
        imgnet_std = [1 for i in range(self.n_channels)]
        imgnet_mean[1:4] = (0.406, 0.456, 0.485)  # rgb are 3,2,1
        imgnet_std[1:4] = (0.225, 0.224, 0.229)
        mn = imgnet_mean
        std = imgnet_std
        
        self.train_transform = transforms.Compose([
            RandomRotate(0.5, 50, seed=self.seed),
            RandomVerticalFlip(0.5, seed=self.seed),
            RandomHorizontalFlip(0.5, seed=self.seed),
            RandomShear(0.5, 20, seed=self.seed),
            ToTensor(round_mask=True),
            Resize(800),
            Normalize(imgnet_mean, imgnet_std)
        ])
        self.test_transform = transforms.Compose([
            ToTensor(round_mask=True),
            Resize(800),
            Normalize(imgnet_mean, imgnet_std)
        ])
        
    def read_groups(self, verbose=False):
        """
        Read folds (i.e., colors) - for each fold get the corresponding input folders of Sentinel-2 dataset
        @return dictionary: key = fold color, value = list of dataset folders in this fold
        """
        groups = OrderedDict()
        df = pd.read_csv(satellite_folds_csv_path)
        grpby = df.groupby('fold')
        for grp in grpby:
            folder_list = grp[1]['folder'].tolist()

            if verbose==True:
                print('______________________________________')
                print(f'fold key: {grp[0]}')
                print(f'folders ({len(folder_list)}): {str(folder_list)}')
            groups[grp[0]] = folder_list
        return groups


class ConcatTypes:
    """
    Specifies the different Concatenated Unet types, based on loss functions.
    """
    BCE_MSE = 'BCE-MSE'
    DICE_MSE = 'DICE-MSE'
    SoftIoU_MSE = 'IoU-MSE'
    SoftIoU_SoftIoU = 'SoftIoU-SoftIoU'
    ComboBSM = 'Combo  BCE-softIoU_MSE'
    ComboBDM = 'Combo  BCE-DICE_MSE'
    F1MSE = 'BCE-F1*MSE'
    @staticmethod
    def get_loss_tuple(model_type, training_config):
        """
        Return loss tuple (firstUnet, secondUnet), given unet type
        @param model_type: select one among ConcatTypes (e.g. ConcatTypes.BCE_MSE)
        @param training_config: TrainingConfig object, with configuration
        """
        if model_type == ConcatTypes.BCE_MSE:
            loss_first_args = {'index': 0, 'gt_one_hot': training_config.mask_one_hot, 'loss': nn.BCEWithLogitsLoss()}
            loss_second_args = {'index': 1, 'gt_one_hot': training_config.mask_one_hot, 'loss': nn.MSELoss()}
        elif model_type == ConcatTypes.DICE_MSE:
            loss_first_args = {'index': 0, 'gt_one_hot': training_config.mask_one_hot, 'loss': GDiceLossV2()}
            loss_second_args = {'index': 1, 'gt_one_hot': training_config.mask_one_hot, 'loss': nn.MSELoss()}
        elif model_type == ConcatTypes.SoftIoU_MSE:
            loss_first_args = {'index': 0, 'gt_one_hot': training_config.mask_one_hot, 'loss': FuzzyIoULoss()}
            loss_second_args = {'index': 1, 'gt_one_hot': training_config.mask_one_hot, 'loss': nn.MSELoss()}
        elif model_type == ConcatTypes.SoftIoU_SoftIoU:
            loss_first_args = {'index': 0, 'gt_one_hot': training_config.mask_one_hot, 'loss': FuzzyIoULoss()}
            loss_second_args = {'index': 1, 'gt_one_hot': training_config.mask_one_hot, 'loss': softIoULoss()}
        elif model_type == ConcatTypes.ComboBSM:
            loss_first_args = {'index': 0, 'gt_one_hot': training_config.mask_one_hot, 'loss': ComboLoss(nn.BCEWithLogitsLoss(), FuzzyIoULoss())}
            loss_second_args = {'index': 1, 'gt_one_hot': training_config.mask_one_hot, 'loss': nn.MSELoss()}
        elif model_type == ConcatTypes.ComboBDM:
            loss_first_args = {'index': 0, 'gt_one_hot': training_config.mask_one_hot, 'loss': ComboLoss(nn.BCEWithLogitsLoss(), GDiceLossV2())}
            loss_second_args = {'index': 1, 'gt_one_hot': training_config.mask_one_hot, 'loss': nn.MSELoss()}
        elif model_type == ConcatTypes.F1MSE:
            loss_first_args = {'index': 0, 'gt_one_hot': training_config.mask_one_hot, 'loss': nn.BCEWithLogitsLoss()}
            loss_second_args = {'index': 1, 'gt_one_hot': training_config.mask_one_hot, 'loss': F1MSE()}

        return (IndexLoss, loss_first_args), (IndexLoss, loss_second_args)


class SingleTypes:
    """
    Specifies the different <Single>Net types, based on loss functions.
    """
    MSE = 'MSE'
    SoftIoU = 'Soft IoU'
    F1MSE = 'F1*MSE'
    @staticmethod
    def get_loss_tuple(model_type, training_config):
        """
        Return loss function, given <Single>net type
        @param model_type: select one among SingleTypes (e.g. SingleTypes.MSE)
        @param training_config: TrainingConfig object, with configuration
        """
        if model_type == SingleTypes.MSE:
            return nn.MSELoss, {}
        if model_type == SingleTypes.SoftIoU:
            return softIoULoss, {}
        elif model_type == SingleTypes.F1MSE:
            return F1MSE, {}


class GradcamTypes:
    """
    Specifies the different binary <Single>Net types, based on loss functions for gradcam analysis.
    """
    DICE = 'Dice loss'
    BCE = 'BCE loss'
    SoftIoU = 'Soft IoU'
    @staticmethod
    def get_loss_tuple(model_type, training_config):
        """
        Return loss function, given <Single>net type
        @param model_type: select one among SingleTypes (e.g. SingleTypes.MSE)
        @param training_config: TrainingConfig object, with configuration
        """
        if model_type == GradcamTypes.DICE:
            return GDiceLossV2, {}
        if model_type == GradcamTypes.BCE:
            return nn.BCEWithLogitsLoss, {}
        if model_type == GradcamTypes.SoftIoU:
            return FuzzyIoULoss, {}


def run_configuration(model_id, name, model_type, config, backbones=None, single_fold=False, gradcam=False):
    """
    Run training configuration for Concatenated Unet
    @param model_id: model id, used to run the correct cv (0: unet, 1: PSPNet)
    @param name: configuration name, used for output folders
    @param model_type: select one among <Net>Types (e.g., ConcatTypes.BCE_MSE)
    @param training_config: TrainingConfig object, with configuration
    """
    base_dir = pretrained_dir if single_fold else base_result_dir
    result_path = os.path.join(base_dir, name)
    print('------')
    print(f'Selected model: {model_names[model_id] +", "+ model_type}')
    print(f'Batch size: {config.batch_size}')
    print(f'Result path: {result_path}')
    print('------')

    # Select model:

    model_tuples = {
        0: (ConcatenatedUNet, {'n_channels': config.n_channels, 'act': 'relu'}),
        1: (PSPNet, {'output_type':'regr', 'n_channels': config.n_channels, 'regr_range':4, 'backend':'resnet18'}),
        2: (NestedUNet, {'output_type':'regr', 'n_channels': config.n_channels, 'regr_range':4, 'act': 'relu'}),
        3: (ConcatenatedNestedUNet, {'n_channels': config.n_channels, 'act': 'relu'}),
        4: (SegNet, {'n_channels': config.n_channels}),
        5: (ConcatenatedSegNet, {'n_channels': config.n_channels}),
        # 6: (MixedNet, {'n_channels': config.n_channels, 'bin_net': backbones[0], 'regr_net': backbones[1]}),
        7: (UNet, {'n_channels': config.n_channels, 'n_classes': 1, 'act': 'relu'}),
    }

    if gradcam:
        criterion_tuple = GradcamTypes.get_loss_tuple(model_type, config)  # Get loss function config
    else:
        if model_id in [0, 3, 5, 6]:
            criterion_tuple = ConcatTypes.get_loss_tuple(model_type, config)  # Get loss function config
        else:
            criterion_tuple = SingleTypes.get_loss_tuple(model_type, config)  # Get loss function config

    model_tuple = model_tuples[model_id]

    # Supervision during training: evaluate results after each epoch
    bin_perf_func = AccuracyBinStorage(config.mask_one_hot)
    regr_perf_func = AccuracyAllStorage(config.mask_one_hot)
    single_regr_perf_func = AccuracySingleRegrStorage()

    if gradcam:
        cv = GradcamCrossValidator(config.groups, model_tuple, criterion_tuple, config.train_transform, config.test_transform,
                                config.master_folder, config.satellite_csv_path, config.epochs, config.batch_size,
                                config.lr, config.wd, config.product_list, config.mode, config.process_dict,
                                config.mask_intervals, config.mask_one_hot, config.height, config.width,
                                config.filter_validity_mask, config.only_burnt, config.mask_filtering, config.seed, result_path,
                                config.scheduler_tuple, performance_eval_func=bin_perf_func,
                                squeeze_mask=False, early_stop=True, patience=config.patience, tol=config.tol,
                                validation_dict=config.validation_dict)
    else:
        if model_id in [0, 3, 5, 6]:
            cv = ConcatenatedCrossValidator(config.groups, model_tuple, criterion_tuple, config.train_transform, config.test_transform,
                                            config.master_folder, config.satellite_csv_path, config.epochs, config.batch_size,
                                            config.lr, config.wd, config.product_list, config.mode, config.process_dict,
                                            config.mask_intervals, config.mask_one_hot, config.height, config.width,
                                            config.filter_validity_mask, config.only_burnt, config.mask_filtering, config.seed, result_path,
                                            config.scheduler_tuple, performance_eval_func=bin_perf_func, second_eval_func=regr_perf_func,
                                            squeeze_mask=False, early_stop=True, patience=config.patience, tol=config.tol,
                                            validation_dict=config.validation_dict, single_fold=single_fold)
        elif model_id in [1, 2, 4, 7]:
            cv = CrossValidator(config.groups, model_tuple, criterion_tuple, config.train_transform, config.test_transform,
                                config.master_folder, config.satellite_csv_path, config.epochs, config.batch_size,
                                config.lr, config.wd, config.product_list, config.mode, config.process_dict,
                                config.mask_intervals, config.mask_one_hot, config.height, config.width,
                                config.filter_validity_mask, config.only_burnt, config.mask_filtering, config.seed, result_path,
                                config.scheduler_tuple, performance_eval_func=single_regr_perf_func,
                                squeeze_mask=False, early_stop=True, patience=config.patience, tol=config.tol,
                                is_regression=True, validation_dict=config.validation_dict
                                )


    cv.start()

