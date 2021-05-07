import os 
import numpy as np
import nrrd
import logging 
import nibabel as nib
from skimage import measure
import imageio
import glob 
import SimpleITK as sitk



def buildDir(path):
    image_dir = os.path.join(path, 'image')
    npy_dir = os.path.join(path, 'npy')
    mask_dir = os.path.join(path, 'mask')
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    if not os.path.exists(npy_dir):
        os.mkdir(npy_dir)
    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)

    return image_dir, npy_dir, mask_dir



def windowProcess(img):
    upper = np.percentile(img, 99.86)
    lower = np.percentile(img, 0.14)
    img[img > upper] = upper
    img[img < lower] = lower
    return img


def normalize(img):
    ymax = 1
    ymin = 0
    xmax = np.max(img)
    xmin = np.min(img)
    norm_img = (ymax - ymin) / (xmax - xmin) * (img - xmin)  + ymin  # -->[0,1]
    return norm_img



def getImgMask(case_path, item_list, slice):
    image_path = case_path + '/' + item_list[slice - 1]
    mask_path = case_path + '/' + item_list[slice]
    image_array = nib.load(image_path).get_fdata()
    mask_array, mask_head = nrrd.read(mask_path)
    image_array = windowProcess(image_array)
    image_array = normalize(image_array)
    mask_array[mask_array > 0] = 1
    return image_array, mask_array



def saveImg(image_array, mask_array, phase):
    mask_list = sorted(list(set(np.nonzero(mask_array)[-1])))
    for i in range(mask_list[0] - 1, mask_list[-1] + 2):
        slice_array = image_array[:, :, i]
        slice_mask = mask_array[:, :, i]
        image_name = test_num + '_' + str(i) + phase + '.jpg'
        npy_name = test_num + '_' + str(i) + phase + '.npy'
        mask_name = test_num + '_' + str(i) + phase + '_mask_.npy'

        imageio.imwrite(os.path.join(image_dir, image_name), slice_array)
        np.save(os.path.join(npy_dir, npy_name), slice_array)
        np.save(os.path.join(mask_dir, mask_name), slice_mask)
        

    


if __name__ == "__main__":
    mvi_path = '/media/drs/extra/Datasets/MVI/MVI_1202'

    dir_root = '/media/drs/extra2/Datasets/mvi_seg'
    image_dir, npy_dir, mask_dir = buildDir(dir_root)

    case_list = glob.glob(mvi_path + '/*/*')

    for case_path in case_list:
        print(case_path)

        idx = -1

        item_list = sorted(os.listdir(case_path))
        for item in item_list:
            idx += 1
            if item.split('_')[-1] == 'ART.nrrd':
                art_mask = idx 
            elif item.split('_')[-1] == 'NC.nrrd':
                nc_mask = idx
            elif item.split('_')[-1] == 'PV.nrrd':
                pv_mask = idx 
            elif item.split('_')[-1] == 'DL.nrrd':
                dl_mask = idx 
        
        art_image_array, art_mask_array = getImgMask(case_path, item_list, art_mask)
        nc_image_array, nc_mask_array = getImgMask(case_path, item_list, nc_mask)
        pv_image_array, pv_mask_array = getImgMask(case_path, item_list, pv_mask)
        dl_image_array, dl_mask_array = getImgMask(case_path, item_list, dl_mask)

        test_num = case_path.split('/')[-2]

        saveImg(art_image_array, art_mask_array, '_art_')
        saveImg(nc_image_array, nc_mask_array, '_nc_')
        saveImg(pv_image_array, pv_mask_array, '_pv_')
        saveImg(dl_image_array, dl_mask_array, '_dl_')