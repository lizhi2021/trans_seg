import torch 
import os 
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .transform import pad_data, resize_data

class tranSeg(Dataset):
    def __init__(self, path, image_size, transforms=None):
        self.transforms = transforms
        
        image_path = os.path.join(path, 'image')
        mask_path = os.path.join(path, 'mask')

        self.info = []

        for image in sorted(os.listdir(image_path)):
            image_array = np.load(os.path.join(image_path, image))
            image_array = pad_data(image_array)
            image_array = resize_data(image_array)

            mask = image.split('.')[0] + 'mask.npy'
            mask_array = np.load(os.path.join(mask_path, mask))
            mask_array = pad_data(mask_array)
            mask_array = resize_data(image_array)

            self.info.append((image_array, mask_array))



    def __getitem__(self, index):
        image_array, mask_array = self.info(index)
        
        if self.transforms is not None:
            image_array, mask_array = self.transforms(image_array, mask_array)

        image_array = np.expand_dims(image_array, axis=0)
        image_tensor = torch.from_numpy(image_array.astype(np.float32))
        mask_tensor = torch.from_numpy(mask_array.astype(np.uint8))

        return image_tensor, mask_tensor



    def __len__(self):
        return len(self.info)