import numpy as np
import skimage.transform
from scipy import ndimage

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask 

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


def pad_data(data):
    a = data.shape[0]
    b = data.shape[1]
    if a == b:
        return data
    diff = (max(a, b) - min(a, b)) / 2.0
    if a > b:
        padding = ((0, 0), (int(np.floor(diff)), int(np.ceil(diff))))
    else:
        padding = ((int(np.floor(diff)), int(np.ceil(diff))), (0, 0))
    data = np.pad(data, padding, mode='constant', constant_values=0)
    return data


def resize_data(data, size):
    [width, height] = data.shape
    scale = [size * 1.0 / width, size * 1.0 / height]
    data = ndimage.interpolation.zoom(data, scale, order=0)
    
    return data


def transforms(angle=None, flip_prob=None):
    transform_list = []

    if angle is not None:
        transform_list.append(Rotate(angle))
    if flip_prob is not None:
        transform_list.append(HorizontalFlip(flip_prob))

    return Compose(transform_list)


class Rotate(object):
    def __init__(self, angle):
        self.angle = angle 

    def __call__(self, image, mask):
        angle = np.random.uniform(low=-self.angle, high=self.angle)
        image = skimage.transform.rotate(image, angle, resize=False, preserve_range=True, mode='constant')
        mask = skimage.transform.rotate(mask, angle, resize=False, preserve_range=True, mode='constant')
        mask[mask > 0] = 1
        return image, mask 


class HorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, mask):
        if np.random.rand() > self.flip_prob:
            return image, mask
        
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()
        return image, mask