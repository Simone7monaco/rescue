import torch
import pytorch_lightning as pl

from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from .contextcnn import ContextCNN
from .dataset import WindowDataset
from .sampler import ShuffleSampler

from torchvision import transforms

class WrappedContextCNN(pl.LightningModule):
    def __init__(self, window_size, in_features, cnn_list, fc_list, pool_type, n_classes, seed, dset_def: dict, dset_args: dict, batch_size: int, loss, lr):
        self.net = ContextCNN(window_size, in_features, cnn_list, fc_list, pool_type, n_classes)
        self.seed = seed
        self.batch_size = batch_size
        self.loss = loss
        self.lr = lr
        return

    def forward(self, x):
        x = self.net(x)
        return x

    def prepare_data(self):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.train_dataset = WindowDataset(folder_list=self.dset_def['train'], window_size=self.net.window_size, transform=transform, **self.dset_args)
        self.validation_dataset = WindowDataset(folder_list=self.dset_def['validation'], window_size=self.net.window_size, transform=transform, **self.dset_args)
        self.test_dataset = WindowDataset(folder_list=self.dset_def['test'], window_size=self.net.window_size, transform=transform, **self.dset_args)
        return

    def train_dataloader(self):
        sampler = ShuffleSampler(self.train_dataset, seed=self.seed)
        loader = DataLoader(self.train_dataset, sampler=sampler, batch_size=self.batch_size)
        return loader

    def val_dataloader(self):
        sampler = ShuffleSampler(self.validation_dataset, seed=self.seed)
        loader = DataLoader(self.validation_dataset, seed=self.seed, batch_size=self.batch_size)
        return loader

    def test_dataloader(self):
        sampler = ShuffleSampler(self.test_dataset, seed=self.seed)
        loader = DataLoader(self.test_dataset, sampler=sampler, batch_size=self.batch_size)
        return loader

    def training_step(self, batch, batch_idx):
        image, ground_truth = batch
        outputs = self.net(image)
        loss = self.loss(outputs, ground_truth)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        return optimizer

    def validation_step(self, batch, batch_idx):
        image, ground_truth = batch
        outputs = self.net(image)
        loss = self.loss(outputs, ground_truth)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        image, ground_truth = batch
        outputs = self.net(image)
        loss = self.loss(outputs, ground_truth)
        return {'test_loss': loss}
        
