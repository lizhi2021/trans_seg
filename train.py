import os
import argparse
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl 
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from data.transform import transforms
from data.data_load import TranSeg

from models.Unet_model import UNet
from models.ResUnet import ResUNet

from utils.losses import dice_coef, dice_loss


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
parser.add_argument('--resume', default='best', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=1234, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('-f', '--flip_prob', default=0.5, type=float, 
                    help='flip probability for augmentation.')
parser.add_argument('--angle', default=15, type=int, 
                    help='rotation angle range in degrees for augmentation.')
parser.add_argument('-i', '--image_size', default=512, type=int, 
                    help='image size to fit into the network')
parser.add_argument('--val_inter', default=1, type=float, 
                    help='check val dataset 1 / val_inter times during one epoch')
parser.add_argument('--top_k', default=5, type=int, 
                    help='top k min val_loss model parameter saved during validation')


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
        self.log('train/ce_loss', loss_ce)
        out_soft = F.softmax(out, dim=1)
        loss_dice = dice_loss(out_soft[:, 1, :, :], mask==1)
        self.log('train/dice_loss', loss_dice)

        tr_loss = 0.5 * (loss_ce + loss_dice)
        self.log('train/joint_loss', tr_loss)
        return tr_loss 

    
    def validation_step(self, val_batch, batch_idx):
        img, mask = val_batch
        img = img.float()
        mask = mask.long()
        out = self(img)

        loss_ce = nn.CrossEntropyLoss()(out, mask)
        self.log('val/ce_loss', loss_ce)
        out_soft = F.softmax(out, dim=1)
        loss_dice = dice_loss(out_soft[:, 1, :, :], mask==1)
        self.log('val/dice_loss', loss_dice)

        val_loss = 0.5 * loss_ce + 0.5 * loss_dice
        self.log('val/joint_loss', val_loss)

        out_pre = torch.argmax(out, dim=1)
        dice = dice_coef(out_pre.data.cpu().numpy(), mask.data.cpu().numpy())
        self.log('val/dice', dice)
        
        sample_imgs = img[:1]
        grid = torchvision.utils.make_grid(sample_imgs, 1).cpu().numpy()
        grid = np.transpose(grid, (0, 2, 1))
        self.logger.experiment.add_image('val/example_image', grid, batch_idx)

        sample_mask = mask[:1]
        mask_grid = torchvision.utils.make_grid(sample_mask, 1).cpu().numpy()
        mask_grid = np.transpose(mask_grid, (0, 2, 1))
        self.logger.experiment.add_image('val/example_mask', mask_grid, batch_idx)

        sample_out = out_pre[:1]
        out_grid = torchvision.utils.make_grid(sample_out, 1).cpu().numpy()
        out_grid = np.transpose(out_grid, (0, 2, 1))
        self.logger.experiment.add_image('val/example_out', out_grid, batch_idx)

        return val_loss


    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x for x in outputs]).mean()
        self.log('val/loss_mean', val_loss_mean)


    def test_step(self, test_batch, batch_idx):
        img, mask = test_batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        # print(img.shape, mask.shape, out.shape)

        loss_ce = nn.CrossEntropyLoss()(out, mask)
        self.log('test/ce_loss', loss_ce)
        out_soft = F.softmax(out, dim=1)
        print(out_soft)
        out_pre = torch.argmax(out_soft, dim=1, keepdim=True)
        loss_dice = dice_loss(out_soft[:, 1, :, :], mask==1)
        self.log('test/dice_loss', loss_dice)

        test_loss = 0.2 * loss_ce + 0.8 * loss_dice
        self.log('test/joint_loss', test_loss)

        sample_imgs = img[:1]
        grid = torchvision.utils.make_grid(sample_imgs, 2).cpu().numpy()
        grid = np.transpose(grid, (0, 2, 1))
        self.logger.experiment.add_image('test/example_images', grid, batch_idx)

        sample_mask = mask[:1]
        mask_grid = torchvision.utils.make_grid(sample_mask, 2).cpu().numpy()
        mask_grid = np.transpose(mask_grid, (0, 2, 1))
        self.logger.experiment.add_image('test/example_masks', mask_grid, batch_idx)

        sample_out = out_pre[:1]
        out_grid = torchvision.utils.make_grid(sample_out, 1).cpu().numpy()
        out_grid = np.transpose(out_grid, (0, 2, 1))
        self.logger.experiment.add_image('test/example_out', out_grid, batch_idx)

        out_pre = out_pre.data.cpu().numpy()
        print(np.max(out_pre), np.min(out_pre))
        mask = mask.cpu().data.numpy()
        mask = np.expand_dims(mask, axis=1)
        # print(out_pre.shape, mask.shape)

        dice = dice_coef(out_pre, mask)
        self.log('test/dice', dice)
        
        return dice



    def test_epoch_end(self, outputs):        
        mean_dice = np.mean(outputs)
        self.log('test/mean_dice', mean_dice)
        print('dice:', mean_dice)

    
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
        train_num = int(len(dataset) * 0.6)
        val_num = int(len(dataset) * 0.2)
        test_num = len(dataset) - train_num - val_num
        self.trainset, self.valset, self.testset = random_split(dataset, [train_num, val_num, test_num])


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


    def test_dataloader(self):
        return DataLoader(
            dataset=self.testset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=args.workers, 
            pin_memory=True
        )



def main():
    
    seed_everything(seed=args.seed)

    date = '0520_liver_test'

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
        # filename='ResUnet-{epoch:02d}-{step}-{val_loss:.3f}',
        filename='ResUnet-{epoch:02d}-{step}-{val/joint_loss:.4f}',
        verbose=True, 
        # monitor='val_loss', 
        monitor='val/joint_loss',
        mode='min', 
        save_last=True, 
        save_top_k=args.top_k, 
    )

    trainer = pl.Trainer(
        logger=TensorBoardLogger(logs_path, name='ResUnet'),
        checkpoint_callback=checkpoint_callback, 
        gpus=args.gpu, 
        benchmark=True, 
        amp_backend='apex', 
        amp_level='O1',
        auto_scale_batch_size=True, 
        max_epochs=args.epochs, 
        val_check_interval=args.val_inter
    )

    if args.evaluate:
        trainer.test(model, datamodule=segData, ckpt_path=args.resume)
    else:
        trainer.fit(model, segData)


if __name__ == '__main__':
    main()