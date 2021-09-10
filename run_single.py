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
    parser.add_argument('--discard_images', nargs='?', default=False, const=True, help = "Prevent Wandb to save validation result for each step.")
    parser.add_argument('-k', '--key', type=str, default='purple', help = "Test set fold key. Default is 'blue'.")
    parser.add_argument('-m', '--single_model', type=str, default='unet', help = "Select the model (unet, canet or attnet available). Default is unet.")
    
    parser.add_argument('--lr', type=float, default=None, help = "Custom lr.")
    parser.add_argument('--seed', type=float, default=7, help = "Custom seed.")
    parser.add_argument('--encoder', type=str, default='resnet34', help = "Select the model encoder (only available for smp models). Default is resnet34.")
    
    args = parser.parse_args()

    return args


def train(args):        
    hparams = TrainingConfig(**vars(args))
#                             , n_channels=24, mode='both')
#                             , only_burnt=False)
    
    if not torch.cuda.is_available():
        hparams.trainer.gpus = 0
        hparams.trainer.precision = 32
    
    name = f'test_{args.double_model}_{args.key}_{args.seed}'
    run = wandb.init(reinit=True, project="rescue_paper", entity="smonaco", name=name, settings=wandb.Settings(start_method='fork'))
    
    outdir = Path("../data/new_ds_logs/Propaper")
    outdir = outdir / wandb.run.name
    outdir.mkdir(parents=True, exist_ok=True)
    print(f'Best checkpoints saved in "{outdir}"')

    pl_model = Satmodel(hparams, {'log_imgs': not args.discard_images})
    
    earlystopping_callback = EarlyStopping(**hparams.earlystopping)
    hparams["checkpoint"]["dirpath"] = outdir
    checkpoint_callback = ModelCheckpoint(**hparams.checkpoint)
    
    logger = WandbLogger(save_dir=outdir, name=name)
    logger.log_hyperparams(hparams)
    logger.watch(pl_model, log='all', log_freq=1)
    
    trainer = pl.Trainer(
        **hparams.trainer,
        max_epochs=hparams.epochs,
        logger=logger,
        callbacks=[checkpoint_callback,
                   earlystopping_callback
                   ],
    )

    trainer.fit(pl_model)
    
    best = Path(checkpoint_callback.best_model_path)
    best.rename(best.parent / f'{wandb.run.name}-best{best.suffix}')
    wandb.finish()

if __name__ == '__main__':
    args = get_args()
    train(args)
    