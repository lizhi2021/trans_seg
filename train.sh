#!/bin/bash

python train.py \
--path '/media/drs/extra2/Datasets/mvi_seg' \
--gpu -1 \
--lr 0.0001 \
--batch_size 2 \
--epochs 50 \
--image_size 512 \
--val_inter 0.2 \
--angle 15 \
--flip_prob 0.5 \
--seed 1234 \
--top_k 8