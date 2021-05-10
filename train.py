import os
import argparse
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl 
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data.transform import transforms
from data.data_load import TranSeg

from models.Unet_model import UNet
from models.ResUnet import ResUNet

from utils.losses import dice_loss


parser = argparse.ArgumentParser(description='segmentation')

parser.add_argument('--path', metavar='DIR', 
                    help='path of dataset')
parser.add_argument('-w', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--start_iter', default=0, type=int, metavar='N',
                    help='manual iter number (useful on restarts')
parser.add_argument('-b', '--batch_size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('-f', '--flip_prob', default=0.5, type=float, 
                    help='flip probability for augmentation.')
parser.add_argument('--angle', default=15, type=int, 
                    help='rotation angle range in degrees for augmentation.')
parser.add_argument('-i', '--image_size', default=512, type=int, 
                    help='image size to fit into the network')


args = parser.parse_args()




class SegModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.lr = args.lr
        # self.net = UNet(n_channels=1, n_classes=2)
        self.net = ResUNet(in_ch=1, out_ch=2)


    def forward(self, x):
        return self.net(x)

    
    def training_step(self, train_batch, batch_idx):
        img, mask = train_batch
        img = img.float()
        mask = mask.long()
        out = self(img)

        loss_ce = nn.CrossEntropyLoss()(out, mask)
        self.log('train_ce_loss', loss_ce)
        out_soft = F.softmax(out, dim=1)
        loss_dice = dice_loss(out_soft[:, 1, :, :, :], mask==1)
        self.log('train_dice_loss', loss_dice)

        tr_loss = 0.5(loss_ce + loss_dice)
        self.log('train_loss', tr_loss)
        return tr_loss 

    
    def validation_step(self, val_batch, batch_idx):
        img, mask = val_batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        val_loss = self.loss(out, mask)
        self.log('val_loss', val_loss)
        return val_loss


    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x for x in outputs]).mean()
        self.log('val_loss_mean', val_loss_mean)

    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]



class SegDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_path = args.path
        self.batch_size = args.batch_size
        self.workers = args.workers
        self.image_size = args.image_size
        self.transform = transforms(angle=args.angle, flip_prob=args.flip_prob)


    def setup(self, stage=None):
        dataset = TranSeg(
            path=self.data_path, 
            image_size=self.image_size, 
            transforms=self.transform
        )
        train_num = int(len(dataset) * 0.7)
        val_num = len(dataset) - train_num
        self.trainset, self.valset = random_split(dataset, [train_num, val_num])


    def train_dataloader(self):
        return DataLoader(
            self.trainset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.workers, 
            pin_memory=True
        )
    

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.workers, 
            pin_memory=True
        )



def main():

    date = '0509'

    model = SegModel(args)
    segData = SegDataModule(args)

    pwd_path = os.path.dirname(__file__)
    ckpts_path = os.path.join(pwd_path, 'ckpts', date)
    logs_path = os.path.join(pwd_path, 'logs', date)

    if not os.path.exists(ckpts_path):
        os.makedirs(ckpts_path)    
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpts_path, 
        filename='ResUnet-{epoch:02d}-{step}-{val_loss:.3f}',
        verbose=True, 
        monitor='val_loss', 
        mode='min', 
        save_last=True, 
        save_top_k=5
    )

    trainer = pl.Trainer(
        logger=TensorBoardLogger(logs_path),
        checkpoint_callback=checkpoint_callback, 
        gpus=args.gpu, 
        benchmark=True, 
        amp_backend='apex', 
        amp_level='O1',
        auto_scale_batch_size=True, 
        max_epochs=args.epochs, 
    )

    trainer.fit(model, segData)


if __name__ == '__main__':
    main()