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

from base_train import Satmodel, Double_Satmodel
import neural_net


def get_args():
    parser = argparse.ArgumentParser(description='Training of the U-Net usign Pytorch Lightining framework.')
    parser.add_argument('--discard_results', nargs='?', default=False, const=True, help = "Prevent Wandb to save validation result for each step.")
    parser.add_argument('-k', '--key', type=str, default='purple', help = "Test set fold key. Default is 'blue'.")
    parser.add_argument('-m', '--double_model', type=str, default='unet', help = "Select the model (unet, canet or attnet available). Default is unet.")
    
    parser.add_argument('--lr', type=float, default=None, help = "Custom lr.")
    parser.add_argument('--seed', type=float, default=None, help = "Custom seed.")
    parser.add_argument('--encoder', type=str, default='resnet34', help = "Select the model encoder (only available for smp models). Default is resnet34.")
    
    
    args = parser.parse_args()

    return args


def train(args):
    hparams = TrainingConfig(**vars(args))
#                             , n_channels=24, mode='both')
#                             , only_burnt=False)
    
#     run = wandb.init(reinit=True, project="rescue", entity="smonaco", name=name, settings=wandb.Settings(start_method='fork'))
    
    outdir = Path("../data/new_ds_logs/Propaper")
#     outdir = outdir / wandb.run.name
    outdir.mkdir(parents=True, exist_ok=True)
    print(f'Best checkpoints saved in "{outdir}"')
    
    earlystopping_1 = EarlyStopping(**hparams.earlystopping)
    earlystopping_2 = EarlyStopping(**hparams.earlystopping)
    
    hparams["checkpoint"]["dirpath"] = outdir
    checkpoint_1 = ModelCheckpoint(**hparams.checkpoint, filename='binary_model-{epoch}')
    checkpoint_2 = ModelCheckpoint(**hparams.checkpoint, filename='regression_model-epoch{epoch}')
    
    name=None
    
    #### 1st network ###################
    bin_model = Double_Satmodel(hparams, binary=True, discard_res=args.discard_results)
    
    logger = WandbLogger(save_dir=outdir, name=name)
    logger.log_hyperparams(hparams)
    logger.watch(bin_model, log='all', log_freq=1)
    
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=hparams.epochs,
        # distributed_backend="ddp",  # DistributedDataParallel
        log_every_n_steps=1,
        progress_bar_refresh_rate=1,
        benchmark=True,
        callbacks=[checkpoint_1,
                   earlystopping_1
                   ],
        precision=16 if torch.cuda.is_available() else 32,
        gradient_clip_val=5.0,
        num_sanity_val_steps=5,
        sync_batchnorm=True,
        logger=logger,
        # resume_from_checkpoint="cyst_checkpoints/prova1/epoch=20-step=8546.ckpt"
    )
    trainer.fit(bin_model)
    
    #### 2nd network ###################
    regr_model = Double_Satmodel.load_from_checkpoint(checkpoint_1.best_model_path,
                                               hparams=hparams, binary=False
                                              )
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=hparams.epochs,
        # distributed_backend="ddp",  # DistributedDataParallel
        log_every_n_steps=1,
        progress_bar_refresh_rate=1,
        benchmark=True,
        callbacks=[checkpoint_2,
                   earlystopping_2
                   ],
        precision=16 if torch.cuda.is_available() else 32,
        gradient_clip_val=5.0,
        num_sanity_val_steps=5,
        sync_batchnorm=True,
        logger=logger,
        # resume_from_checkpoint="cyst_checkpoints/prova1/epoch=20-step=8546.ckpt"
    )
    trainer.fit(regr_model)
    
    best = Path(checkpoint_2.best_model_path)
    best.rename(best.parent / f'{best.stem}_best{best.suffix}')
    wandb.finish()

if __name__ == '__main__':
    args = get_args()
    train(args)
    