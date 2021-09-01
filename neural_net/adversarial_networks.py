import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

import pytorch_lightning as pl

from neural_net import *
from neural_net.utils import *
from neural_net.unet import UNet

import wandb


class disc_block(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, act='relu', ns=0.2):
        super(disc_block, self).__init__()
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'lrelu': 
            self.act = nn.LeakyReLU(negative_slope=ns, inplace=True)
        else:
            raise ValueError('Invalid activation layer parameter')
        
        if bn:
            self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
                    nn.BatchNorm2d(out_ch),
                    self.act
                )
        else:
            self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
                    self.act
                )

    def forward(self, x):
        x = self.conv(x)
        return x
    
    
class last_block(nn.Module):
    def __init__(self, in_ch, use_sigm=True):
        super(last_block, self).__init__()
        
        if use_sigm:
            self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, 1, 4, padding=1),
                    nn.Sigmoid()
                )
        else:
            self.conv = nn.Conv2d(in_ch, 1, 4, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x
    
    
class Discriminator(nn.Module):
    def __init__(self, in_ch, bn=True, act='lrelu', ns=0.01, end_withsigm=False):
        super(Discriminator, self).__init__()
        layers_depths = [64, 128, 265, 521, 512]
        self.conv1 = disc_block(in_ch=in_ch, out_ch=layers_depths[0])
        self.conv2 = disc_block(in_ch=layers_depths[0], out_ch=layers_depths[1], act=act, bn=False)
        self.conv3 = disc_block(in_ch=layers_depths[1], out_ch=layers_depths[2], act=act)
        self.conv4 = disc_block(in_ch=layers_depths[2], out_ch=layers_depths[3], act=act)
        self.conv5 = disc_block(in_ch=layers_depths[3], out_ch=layers_depths[4], act=act)
        self.lastconv = last_block(in_ch=layers_depths[4], use_sigm=end_withsigm)
    
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), axis=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.lastconv(x)
        return x
        

class SatcGAN(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.save_hyperparameters()
        self.key = hparams["key"]
        self.drop_last = False
        
        # networks
    
        self.generator = UNet(hparams["n_channels"], 1, end_withsigm=False)
        self.discriminator = Discriminator(hparams["n_channels"] + 1, end_withsigm=False) # Image + fire (and water)
        self.initialize_models()

        self.adversarial_loss = hparams["criterion"]

        self.binary = hparams["binary"]
        
        self.max_val_iou = 0
        self.train_set = []
        self.validation_set = []
        self.test_set = []      
        
    def initialize_models(self):
        self.discriminator.apply(initialize_weight)
        self.generator.apply(initialize_weight)
        
    def forward(self, x):
        return self.generator(x)
    
    def configure_optimizers(self):
        lrd = self.hparams["opt_params"]["lrd"]
        lrg = self.hparams["opt_params"]["lrg"]
        wdd = self.hparams["opt_params"]["wdd"]
        wdg = self.hparams["opt_params"]["wdg"]
        minBd = self.hparams["opt_params"]["minBd"]
        minBg = self.hparams["opt_params"]["minBg"]
        
        # betas eventually to optimize
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lrg, betas=(minBg, .999), weight_decay=wdg)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lrd, betas=(minBd, .999), weight_decay=wdd)
        return [opt_g, opt_d], []    
    
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
        return result

    def generator_step(self, images, masks):
        """
        Training step for generator
        1. Sample random noise and labels
        2. Pass noise and labels to generator to
           generate images
        3. Classify generated images using
           the discriminator
        4. Backprop loss
        """
        
        fake_masks = self.forward(images)

        # Classify generated images
        # using the discriminator
        d_output = self.discriminator(images, fake_masks)

        # Backprop loss. We want to maximize the discriminator's
        # loss, which is equivalent to minimizing the loss with the true
        # labels flipped (i.e. y_true=1 for fake images). We do this
        # as PyTorch can only minimize a function instead of maximizing
        g_loss = self.adversarial_loss(d_output, torch.ones_like(d_output))
        train_iou = binary_mean_iou(fake_masks, masks)

        self.log('g_loss', g_loss)
        self.log('train_iou', train_iou)
        return g_loss

    def discriminator_step(self, images, masks):
        """
        Training step for discriminator
        1. Get actual images and labels
        2. Predict probabilities of actual images and get BCE loss
        3. Get fake images from generator
        4. Predict probabilities of fake images and get BCE loss
        5. Combine loss from both and backprop
        """
        
        # Real images
        d_output = self.discriminator(images, masks)
        loss_real = self.adversarial_loss(d_output, torch.ones_like(d_output))
        acc_real = (d_output == torch.ones_like(d_output)).sum() / d_output.shape.numel()
        
        # Fake images
        fake_masks = self.forward(images)
        d_output = self.discriminator(images, fake_masks)
        loss_fake = self.adversarial_loss(d_output, torch.zeros_like(d_output))
        acc_fake = (d_output == torch.zeros_like(d_output)).sum() / d_output.shape.numel()

        self.log('d_real_loss', loss_real)
        self.log('d_real_acc', acc_real)
        
        self.log('d_fake_loss', loss_fake)
        self.log('d_fake_acc', acc_fake)
        return loss_real + loss_fake

    def training_step(self, batch, batch_idx, optimizer_idx):
        images = batch["image"]
        masks = batch["mask"]
        
        if self.binary: masks = (masks > .5).type_as(masks)
        
        # train generator
        if optimizer_idx == 0:
            loss = self.generator_step(images, masks)

        # train discriminator
        if optimizer_idx == 1:
            loss = self.discriminator_step(images, masks)
        
#         self.log('lr', self._get_current_lr(optimizer_idx))
#         self.log('gan_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_id):
        images = batch["image"]
        masks = batch["mask"]

        logits = self.forward(images)

        loss = self.adversarial_loss(logits, masks)
        val_iou = binary_mean_iou(logits, masks)
        
        logits_ = (torch.sigmoid(logits) > 0.5).cpu().detach().numpy().astype("float")
        masks_ = (masks > 0.5).cpu().detach().numpy().astype("float")
            
        class_labels = {0: "background", 1: "fire"}
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
    
    def _get_current_lr(self, opt_id):
        lr = [x["lr"] for x in self.optimizers[opt_id].param_groups][0]  # type: ignore

        if torch.cuda.is_available(): return torch.Tensor([lr])[0].cuda()
        return torch.Tensor([lr])[0]

    def validation_epoch_end(self, outputs):
        self.log("epoch", self.trainer.current_epoch)
        avg_val_iou = find_average(outputs, "val_iou")
        self.max_val_iou = max(avg_val_iou, self.max_val_iou)
        
        self.log("val_iou", avg_val_iou)
        self.log("max_val_iou", self.max_val_iou)
        return

#     def on_epoch_end(self):
#         z = self.validation_z.type_as(self.generator.model[0].weight)

#         # log sampled images
#         sample_imgs = self(z)
#         grid = torchvision.utils.make_grid(sample_imgs)
#         self.logger.experiment.add_image('generated_images', grid, self.current_epoch)