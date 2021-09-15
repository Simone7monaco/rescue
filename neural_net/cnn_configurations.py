import os
import yaml
import re
from pathlib import Path
from easydict import EasyDict as ed

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
import pandas as pd

validation_dict = {'purple': 'coral',
                   'coral': 'cyan',
                   'pink': 'coral',
                   'grey': 'coral',
                   'cyan': 'coral',
                   'lime': 'coral',
                   'magenta': 'coral'
                  }

def TrainingConfig(**args):
    with open(Path.cwd() / "configs/models.yaml", "r") as f:
        models = ed(yaml.load(f, Loader=yaml.SafeLoader))
        
    with open(Path.cwd() / "configs/losses.yaml", "r") as f:
        losses = ed(yaml.load(f, Loader=yaml.SafeLoader))
    
    mod_name = [f for f in models.keys() if args["model_name"].lower() == f.lower()]
    
    if (Path.cwd() / f"configs/{mod_name}.yaml").exists():
        with open(Path.cwd() / f"configs/{mod_name}.yaml", "r") as f:
            hparams = ed(yaml.load(f, Loader=yaml.SafeLoader))
    else:
        with open(Path.cwd() / "configs/UNet.yaml", "r") as f:
            hparams = ed(yaml.load(f, Loader=yaml.SafeLoader))
            hparams.model = models[mod_name[0]]
            
    if args["losses"]:
        loss_key = next((key for key in losses.config_names.keys() if args["losses"] in re.sub('[^A-Za-z0-9]+', '', key).lower()))
        hparams.criterion = losses.classification[losses.config_names[loss_key].first]
        hparams.regr_criterion = losses.regression[losses.config_names[loss_key].second]
        
    for k in args:
        if args[k] is None or str(k) in ['model_name', 'discard_images', 'encoder', 'losses']: continue
        found = False
        if k in hparams.keys():
            hparams[k] = args[k]
            found = True
        for k_n in hparams.keys():
            if type(hparams[k_n]) in [dict, ed] and k in hparams[k_n].keys():
                hparams[k_n][k] = args[k]
                found = True
        if not found: print(f"\nParameter `{k}` not found.\n")
    
    print(f"Selecting model {mod_name[0]} as backbone")
    
    hparams.dataset_specs.mask_intervals = [(0, 36), (37, 96), (97, 160), (161, 224), (225, 255)]
    
    hparams.groups = read_groups(hparams.fold_separation_csv)
    
    hparams.validation_dict = validation_dict
#     {
#         "blue": "fucsia",
#         "brown": "fucsia",
#         "fucsia": "green",
#         "green": "fucsia",
#         "orange": "fucsia",
#         "red": "fucsia",
#         "yellow": "fucsia",
#     }
    
    if "imagenet" in hparams.model.values():
        print("\n> Using imagenet preprocessing.")
        mn = [1 for i in range(hparams.model.n_channels)]
        std = [1 for i in range(hparams.model.n_channels)]
        mn[1:4] = (0.406, 0.456, 0.485)  # rgb are 3,2,1
        std[1:4] = (0.225, 0.224, 0.229)
    else:
        mn = (0.5,) * hparams.model.n_channels
        std = (0.5,) * hparams.model.n_channels
            
    # Dataset augmentation and normalization
    hparams.train_transform = transforms.Compose([
        RandomRotate(0.5, 50, seed=hparams.seed),
        RandomVerticalFlip(0.5, seed=hparams.seed),
        RandomHorizontalFlip(0.5, seed=hparams.seed),
        RandomShear(0.5, 20, seed=hparams.seed),
        ToTensor(round_mask=True),
#             Resize(800),
        Normalize(mn, std)
    ])
    hparams.test_transform = transforms.Compose([
        ToTensor(round_mask=True),
#             Resize(800),
        Normalize(mn, std)
    ])
    
    return hparams
    
def update_transforms():
    print("\nUsing pretraining for imagenet weights.\n")

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
        
def read_groups(satellite_folds, verbose=False):
    """
    Read folds (i.e., colors) - for each fold get the corresponding input folders of Sentinel-2 dataset
    @return dictionary: key = fold color, value = list of dataset folders in this fold
    """
    groups = {}
    df = pd.read_csv(satellite_folds)
    for key, grp in df.groupby('fold'):
        folder_list = grp['folder'].tolist()

        if verbose==True:
            print('______________________________________')
            print(f'fold key: {key}')
            print(f'folders ({len(folder_list)}): {str(folder_list)}')
        groups[key] = folder_list
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

