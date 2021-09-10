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
from run_single import get_args


def train(args):
    args.double_model = args.single_model
    del args.single_model
    
    hparams = TrainingConfig(**vars(args))
#                             , n_channels=24, mode='both')
#                             , only_burnt=False)
    
    name = f'test_double-{args.double_model}_{args.key}_{args.seed}'
    
    outdir = Path(f"../data/new_ds_logs/Propaper/{name}")
    
    outdir.mkdir(parents=True, exist_ok=True)
    
    if any(outdir.iterdir()):
        print(f"Simulation already done ({name})")
        return
        
    run = wandb.init(reinit=True, project="rescue_paper", entity="smonaco", name=name, tags=["crossval_double"], settings=wandb.Settings(start_method='fork'))
    
    print(f'Best checkpoints saved in "{outdir}"\n')

    earlystopping_1 = EarlyStopping(**hparams.earlystopping)
    earlystopping_2 = EarlyStopping(**hparams.earlystopping)
    
    hparams.checkpoint.dirpath = outdir
    checkpoint_1 = ModelCheckpoint(**hparams.checkpoint, filename='binary_model-{epoch}')
    checkpoint_2 = ModelCheckpoint(**hparams.checkpoint, filename='regression_model-epoch{epoch}')
    
    if not torch.cuda.is_available():
        hparams.trainer.gpus = 0
        hparams.trainer.precision = 32
        
    
    #### 1st network ###################
    bin_model = Double_Satmodel(hparams, {'log_imgs': not args.discard_images, 'binary': True})
    
    logger = WandbLogger(save_dir=outdir, name=name)
    logger.log_hyperparams(hparams)
    logger.watch(bin_model, log='all', log_freq=1)
    
    trainer = pl.Trainer(
        **hparams.trainer,
        max_epochs=hparams.epochs,
        logger=logger,
        callbacks=[checkpoint_1,
                   earlystopping_1
                   ],
    )
    trainer.fit(bin_model)
    
    #### 2nd network ###################
    regr_model = Double_Satmodel.load_from_checkpoint(checkpoint_1.best_model_path,
                                                      opt={'log_imgs': not args.discard_images, 'binary': False}
                                              )
    trainer = pl.Trainer(
        **hparams.trainer,
        max_epochs=hparams.epochs,
        logger=logger,
        callbacks=[checkpoint_2,
                   earlystopping_2
                   ],
    )
    trainer.fit(regr_model)
    
    trainer.test()
    
    best = Path(checkpoint_2.best_model_path)
    best.rename(best.parent / f'{best.stem}_best{best.suffix}')
    wandb.finish()

if __name__ == '__main__':
    args = get_args()
    train(args)
    