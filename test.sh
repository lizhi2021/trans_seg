#!/bin/bash

python train.py \
--path '/media/drs/extra2/Datasets/mvi_seg' \
--gpu -1 \
--lr 0.0001 \
--batch_size 16 \
--image_size 512 \
--angle 0 \
--flip_prob 0 \
--seed 1234 \
--evaluate \
--resume 'ckpts/0511/last.ckpt'