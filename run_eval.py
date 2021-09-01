from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from tqdm import tqdm
import json

from neural_net.unet import UNet, ConcatenatedUNet
from neural_net.sampler import ShuffleSampler
import pytorch_lightning as pl

from config import *
import wandb
import torch.nn.functional as F
from scipy.stats import logistic

from pathlib import Path
import os
import pickle
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

from torch import nn
from neural_net.attention_unet import AttentionUnet
from neural_net.canet_parts.networks.network import Comprehensive_Atten_Unet

from neural_net import SatelliteDataset, ProductProcessor
from neural_net.transform import *
from neural_net.unet import ConcatenatedUNet, UNet

import re
import ast
from sklearn.metrics import confusion_matrix, mean_squared_error
import seaborn as sns


results = True
attentions = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
thr = .5
print(f"Device : {device}")                    

def get_args():
    parser = argparse.ArgumentParser(description='Test of the U-Net usign Pytorch Lightining framework.')
    parser.add_argument('-w', '--weight_path', type=Path, help="Weight path", default=None)
#     parser.add_argument('--cv', nargs='?', default=False, const=True)
    parser.add_argument('--mode_both', nargs='?', default=False, const=True, help="Use both pre and post images for dataset")
    parser.add_argument('-m', '--model', type=str, default='unet', help="Select the model (unet or canet available). Default is unet.")
    parser.add_argument('-e', '--encoder', type=str, default=None, help="Select the model encoder (only available for smp models). Default is None.")
    parser.add_argument("-a", "--active_attention_layers", default=None, help="Attention layers list")
    
    args = parser.parse_args()
    return args


def rename_layers(state_dict, rename_in_layers):
    result = {}
    for key, value in state_dict.items():
        for key_r, value_r in rename_in_layers.items():
            key = re.sub(key_r, value_r, key)

        result[key] = value

    return result


def state_dict_from_disk(file_path, rename_in_layers=None):
    """Loads PyTorch checkpoint from disk, optionally renaming layer names.
    Args:
        file_path: path to the torch checkpoint.
        rename_in_layers: {from_name: to_name}
            ex: {"model.0.": "",
                 "model.": ""}
    Returns:
    """
#     checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)
    checkpoint = torch.load(file_path) if torch.cuda.is_available() else torch.load(file_path, map_location=torch.device('cpu'))

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if rename_in_layers is not None:
        state_dict = rename_layers(state_dict, rename_in_layers)

    return state_dict


def iou(a, b):
    res = (a & b).sum() / (a | b).sum()
    return res


validation_dict = {
            'blue': 'fucsia',
            'brown': 'fucsia',
            'fucsia': 'green',
            'green': 'fucsia',
            'orange': 'fucsia',
            'red': 'fucsia',
            'yellow': 'fucsia'
        }

def read_groups():
    """
    Read folds (i.e., colors) - for each fold get the corresponding input folders of Sentinel-2 dataset
    @return dictionary: key = fold color, value = list of dataset folders in this fold
    """
    groups = {}
    df = pd.read_csv(satellite_folds_csv_path)
    grpby = df.groupby('fold')
    for grp in grpby:
        folder_list = grp[1]['folder'].tolist()
        groups[grp[0]] = folder_list
    return groups

groups = read_groups()


def eval_key(key, thr:float =.5, show=False, batch_size=4):
    with torch.no_grad():
        t = []
        loader = DataLoader(datasets[key], 
                            batch_size=batch_size, 
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False
                           )
        for idx, data in tqdm(enumerate(loader), desc=f'{key}', total=len(loader)):
#             if idx>50: break
            d = norm(data)
            image, mask = d['image'].to(device), d['mask'].to(device)
            
#             image_np = data['image'].permute(1, 2, 0).squeeze().cpu().numpy()[:, :,:,[3,2,1]]*2.5
#             image_np = (image_np*255).astype(np.uint8)

            mask = mask > .5

            mask_np = mask.squeeze().cpu().numpy()
                
            models = models_tested[key]
            for model in models:
                model.eval()
                pred = model(image)

                pred = pred.squeeze().cpu().numpy()
                pred = logistic.cdf(pred)
#                 for thr in np.arange(.45,.75, .03):
                for thr in [.5]:
                    pred_ = (pred > thr)
                    
                    for i in range(mask_np.shape[0]):
                        TN, FP, FN, TP = confusion_matrix(mask_np[i].astype(bool).ravel(), pred_[i].ravel()).ravel() if mask_np[i].any() else [0, 0, 0, 0]
                        t.append([model.__class__.__name__, thr, key, idx*batch_size+i, iou(mask_np[i], pred_[i]), TN, FP, FN, TP])
    return t
 
    
    
args = get_args()
# weights = list(Path('../data/new_ds_logs/unet/').glob('fold*')) + list(Path('../data/new_ds_logs/canet/').glob('fold*'))
weights = list(args.weight_path.glob('fold*'))

def get_w(f):
    fold = list(f.resolve().iterdir())
    fold.sort(key=lambda x: os.path.getmtime(x))
    fold = [f for f in fold if 'test' not in str(f) and f.is_dir()][-1]
    try:
        w = next((fold / "checkpoints").glob('*best*'))
    except:
        w = next((fold / "checkpoints").glob('*.ckpt'))
    
    return w
weights = [get_w(f) for f in weights]

models_tested = {k:[] for k in groups}

n_channels = 12 if not args.mode_both else 24
if '_13l' in str(weights[0]): n_channels = 13
imagenet = False
for  w in weights:
    checkpoint = torch.load(w, map_location=lambda storage, loc: storage)
    
    if 'canet' in str(w):
#         print(f"-> CA-net\n")
        mod_str = 'canet'
        model = Comprehensive_Atten_Unet(in_ch=n_channels, n_classes=1, im_size=(480, 480)).to(device)
#         model.activate_attention_layers(checkpoint['hyper_parameters']['active_attention_layers'])
        
#         activations.append(checkpoint['hyper_parameters']['active_attention_layers'])
#         encoders.append(checkpoint['hyper_parameters']['encoder'])
    elif 'unet' in str(w):
#         print("-> Unet\n")
        model = UNet(n_channels, 1).to(device)
        mod_str = 'unet'
    elif 'attnet' in str(w):
        mod_str = 'attnet'
        active_attention_layers = [1, 2, 3, 4]
        model = AttentionUnet(encoder_name="resnet34", classes=1, in_channels=n_channels)
        model = model.to(device)
        model.activate_attention_layers(active_attention_layers)
        imagenet = True
        
    state_dict = checkpoint["state_dict"]
    ptr = rename_layers(state_dict, {"model.": ""})
    
    model.load_state_dict(ptr)
    models_tested[checkpoint['hyper_parameters']['key']].append(model)


    
channels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
mask_one_hot = False
filter_validity_mask = True
mask_filtering = False
only_burned = True
height, width = 480, 480
product_list = ['sentinel2']
mode = 'post' if not args.mode_both else 'both'

process_dict = {
    'sentinel2': ProductProcessor(channels, None, 'custom_s2_post'),
}


# Dataset augmentation and normalization
test_transform = transforms.Compose([
    ToTensor(round_mask=True),
])

mn = (0.5,) * n_channels
std = (0.5,) * n_channels

imgnet_mean = [1 for i in range(n_channels)]
imgnet_std = [1 for i in range(n_channels)]
imgnet_mean[1:4] = (0.406, 0.456, 0.485)  # rgb are 3,2,1
imgnet_std[1:4] = (0.225, 0.224, 0.229)
if args.mode_both:
    imgnet_mean[13:4] = (0.406, 0.456, 0.485)  # rgb are 3,2,1
    imgnet_std[13:4] = (0.225, 0.224, 0.229)

if imagenet:
    mn = imgnet_mean
    std = imgnet_std

norm = lambda im : Normalize(mn, std)(im)

datasets = {}
for k in tqdm(groups, desc='Creating datasets'):
    datasets[k] = SatelliteDataset(sentinel_hub_selected_dir, mask_intervals, mask_one_hot, height, width, product_list,
                 mode, filter_validity_mask, test_transform, process_dict, satellite_csv_path,
                 folder_list=groups[k], mask_filtering=mask_filtering, only_burnt=only_burned, mask_postfix='mask')
print()

datapath = f"../data/new_ds_logs/cv_{mod_str}13l_results.csv"
print(f'Results will be saved in "{datapath}"\n')
t = []
for k in groups.keys():
    t += eval_key(k)
    
pd.DataFrame(t, columns=['model', 'threshold', 'fold', 'idx', 'iou', 'TN', 'FP', 'FN', 'TP']).to_csv(datapath)