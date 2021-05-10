#!/bin/bash

python train.py \
--path '/media/drs/extra2/Datasets/mvi_seg' \
--gpu -1 \
--lr 0.0001 \
--batch_size 4 \
--epochs 100 \
--image_size 512 \
--angle 10 \
--flip_prob 0.5