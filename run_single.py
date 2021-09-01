import os
import argparse
import ast
from pathlib import Path
import pickle
import wandb
from easydict import EasyDict as ed

import torch
from neural_net.cnn_configurations import TrainingConfig
from neural_net.utils import str2bool

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from base_train import Satmodel
import neural_net


def get_args():
    parser = argparse.ArgumentParser(description='Training of the U-Net usign Pytorch Lightining framework.')
    parser.add_argument('--discard_results', nargs='?', default=False, const=True, help = "Prevent Wandb to save validation result for each step.")
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


def main(args):        
    
    if args.active_attention_layers:
        active_attention_layers = ast.literal_eval(args.active_attention_layers)
    else:
        active_attention_layers = [i+1 for i, act in enumerate([args.active_attention_layer1, args.active_attention_layer2, args.active_attention_layer3, args.active_attention_layer4]) if act]
    
    
    hparams = TrainingConfig("configs/baseline.yaml", key=args.key)
#                             , n_channels=24, mode='both')
#                             , only_burnt=False)

    outdir = Path("../data/new_ds_logs/Propaper") / f"{args.model}" / f"fold_{hparams.key}"
    
    outdir.mkdir(parents=True, exist_ok=True)
    print(f'Best checkpoints saved in "{outdir}"')
    
    
    if active_attention_layers:
        hparams.update({'encoder' : args.encoder,
                        'active_attention_layers' : active_attention_layers,
                    })
    
    name = f"{args.model}_{args.encoder}_lr{args.lr}_att{''.join([str(int(a in active_attention_layers)) for a in range(1,5)])}"
    
#     run = wandb.init(project="rescue", entity="smonaco", name=name, settings=wandb.Settings(start_method='fork'))

#     outdir = outdir / wandb.run.name
    pl_model = Satmodel(hparams, discard_res=args.discard_results)

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
    
#     logger = WandbLogger(save_dir=outdir, name=name)
#     logger.log_hyperparams(hparams)
#     logger.watch(pl_model, log='all', log_freq=1)
    logger = None
    
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=hparams.epochs,
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
    
