import os
import torch
import numpy as np
import argparse
import ast
from pathlib import Path
import pickle
import wandb
from neural_net.utils import str2bool

from torch import nn
from neural_net.cnn_configurations import TrainingConfig

from base_train import launch_training
from config import *
import segmentation_models_pytorch as smp

from neural_net.unet import UNet
from neural_net.utils import initialize_weight

from base_train import Satmodel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.loggers import WandbLogger

from neural_net.attention_unet import AttentionUnet
from neural_net.canet_parts.networks.network import Comprehensive_Atten_Unet


def get_args():
    parser = argparse.ArgumentParser(description='Training of the U-Net usign Pytorch Lightining framework.')
#     parser.add_argument('freeze', nargs='*', default=None)
    parser.add_argument('--discard_results', nargs='?', default=False, const=True, help = "Prevent Wandb to save validation result for each step.")
    parser.add_argument('--pretrained', nargs='?', default=False, const=True, help = "Encoder weights pretrained from landcover classification.")
    parser.add_argument('-k', '--key', type=str, default='blue', help = "Test set fold key. Default is 'blue'.")
    parser.add_argument('-m', '--model', type=str, default='unet', help = "Select the model (unet, canet or attnet available). Default is unet.")
    
    parser.add_argument('--lr', type=float, default=None, help = "Custom lr.")
    parser.add_argument('--seed', type=float, default=None, help = "Custom seed.")
    parser.add_argument('--encoder', type=str, default='resnet34', help = "Select the model encoder (only available for smp models). Default is resnet34.")
    
    parser.add_argument('-a', '--active_attention_layers', help="List of Active Attention layer (between 1~4). Write [1,3] to set AAL on blocks 1 and 3.", default=None)
    parser.add_argument('-a1', '--active_attention_layer1', type=str2bool, help="Activate Attention layer 1", nargs='?', const=True, default=False)
    parser.add_argument('-a2', '--active_attention_layer2', type=str2bool, help="Activate Attention layer 2", nargs='?', const=True, default=False)
    parser.add_argument('-a3', '--active_attention_layer3', type=str2bool, help="Activate Attention layer 3", nargs='?', const=True, default=False)
    parser.add_argument('-a4', '--active_attention_layer4', type=str2bool, help="Activate Attention layer 4", nargs='?', const=True, default=False)
    
    
    args = parser.parse_args()

    return args

def customize_configs(args, config):
    for attr, val in args.__dict__.items():
        if hasattr(config, attr) and val is not None:
            setattr(config, attr, val)

def main(args):        
    freeze = None#int(args.freeze[-1]) if args.freeze else None
    
    if args.active_attention_layers:
        active_attention_layers = ast.literal_eval(args.active_attention_layers)
    else:
        active_attention_layers = [i+1 for i, act in enumerate([args.active_attention_layer1, args.active_attention_layer2, args.active_attention_layer3, args.active_attention_layer4]) if act]
    
    
    config = TrainingConfig(scheduler_tuple=None, epochs=50)
#                             , n_channels=24, mode='both')
#                             , only_burnt=False)
    customize_configs(args, config)

    
    if freeze:
        print(f"Freezing up to layer {freeze}\n")
        
#     if args.model=='canet' and args.active_attention_layers:
#         name_run += "-".join([str(a) for a in active_attention_layers])
    outdir = Path("../data/new_ds_logs/Propaper") / f"{args.model}" / f"fold_{config.key}"
    
    outdir.mkdir(parents=True, exist_ok=True)
    print(f'Best checkpoints saved in "{outdir}"')

    
    criterion = nn.BCEWithLogitsLoss()
    result_filename = f'pretrained_{args.model}'

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model == 'unet':
        model = UNet(config.n_channels, len(config.mask_intervals) if config.classification else 1, act='relu')
        model.apply(initialize_weight)
        
    if args.model == 'canet':
        model = Comprehensive_Atten_Unet(in_ch=config.n_channels, n_classes=1, im_size=(480, 480))
    if args.model == 'attnet':
        if args.active_attention_layers is None: active_attention_layers = [1, 2, 3, 4]
        model = AttentionUnet(encoder_name=args.encoder, classes=1, in_channels=config.n_channels)
        model.activate_attention_layers(active_attention_layers)
        
        print(f'Attention layers: {active_attention_layers}')   
        config.update_transforms() #imagenet weights
                
    
#     if args.model == 'unet':
#         if args.pretrained:
#             model.load_state_dict(torch.load('../data/pretrained_weights/unet_landcover.pt'))
#             return
#         else:
    
    hparams = {
        'model' : str(model.__class__).split('.')[-1],
    }
    
    if active_attention_layers:
        hparams.update({'encoder' : args.encoder,
                        'active_attention_layers' : active_attention_layers,
                    })
    hparams.update(config.__dict__)
    
    name = f"{args.model}_{args.encoder}_lr{args.lr}_att{''.join([str(int(a in active_attention_layers)) for a in range(1,5)])}"
    
    run = wandb.init(project="rescue", entity="smonaco", name=name, settings=wandb.Settings(start_method='fork'))

    outdir = outdir / wandb.run.name
    pl_model = Satmodel(model, criterion, hparams, discard_res=args.discard_results)

    earlystopping_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=5,
        verbose=True,
        mode='min',
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=outdir / "checkpoints",
        monitor="val_iou",
        verbose=True,
        mode="max",
        save_top_k=2,
    )
    
    logger = WandbLogger(save_dir=outdir, name=name)
    logger.log_hyperparams(vars(config))
    logger.watch(model, log='all', log_freq=1)
    
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=config.epochs,
        # distributed_backend="ddp",  # DistributedDataParallel
        log_every_n_steps=1,
        progress_bar_refresh_rate=1,
        benchmark=True,
        callbacks=[checkpoint_callback,
                   earlystopping_callback
                   ],
        precision=16 if torch.cuda.is_available() else 32,
        gradient_clip_val=5.0,
        num_sanity_val_steps=5,
        sync_batchnorm=True,
        logger=logger,
        # resume_from_checkpoint="cyst_checkpoints/prova1/epoch=20-step=8546.ckpt"
    )

    trainer.fit(pl_model)
    
    best = Path(checkpoint_callback.best_model_path)
    best.rename(best.parent / f'{wandb.run.name}-best{best.suffix}')

if __name__ == '__main__':
    args = get_args()
    main(args)
    
