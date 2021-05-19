#!/bin/bash

python train.py \
--path '/media/drs/extra2/Datasets/mvi_seg' \
--gpu -1 \
--lr 0.0001 \
--batch_size 8 \
--image_size 512 \
--angle 0 \
--flip_prob 0 \
--seed 1234 \
--evaluate \
--workers 8 \
--resume 'ckpts/0514_liver/ResUnet-epoch=43-step=348145-val/joint_loss=0.002.ckpt'