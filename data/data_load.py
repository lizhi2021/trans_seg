import torch 
import os 
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .transform import pad_data, resize_data

class TranSeg(Dataset):
    def __init__(self, path, image_size, transforms=None):
        self.transforms = transforms
        self.image_size = image_size
        
        image_path = os.path.join(path, 'npy')
        mask_path = os.path.join(path, 'mask')

        self.info = []

        for image in sorted(os.listdir(image_path)):
            image_name = os.path.join(image_path, image)

            mask = image.split('.')[0] + 'mask.npy'
            mask_name = os.path.join(mask_path, mask)            

            self.info.append((image_name, mask_name))



    def __getitem__(self, index):
        image_name, mask_name = self.info[index]

        image_array = np.load(image_name, allow_pickle=True)
        image_array = pad_data(image_array)
        image_array = resize_data(image_array, self.image_size)

        mask_array = np.load(mask_name, allow_pickle=True)
        mask_array = pad_data(mask_array)
        mask_array = resize_data(image_array, self.image_size)
        
        if self.transforms is not None:
            image_array, mask_array = self.transforms(image_array, mask_array)

        image_array = np.expand_dims(image_array, axis=0)
        image_tensor = torch.from_numpy(image_array.astype(np.float32))
        mask_tensor = torch.from_numpy(mask_array.astype(np.uint8))

        return image_tensor, mask_tensor



    def __len__(self):
        return len(self.info)