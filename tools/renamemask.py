import os 
import shutil

path = '/media/drs/extra2/Datasets/mvi_seg/liver_mask'

for i in os.listdir(path):
    print(i)
    # new_name = i.split('.')[0] + 'mask.npy'
    new_name = i.replace('_mask_', 'mask')
    os.rename(os.path.join(path, i), os.path.join(path, new_name))
    # shutil.move(os.path.join(path, i), os.path.join(path, 'mvi_seg', i))