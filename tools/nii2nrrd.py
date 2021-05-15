import os 
import nrrd 
import nibabel as nib

path = '/media/drs/extra/Datasets/MVI/MVI_1202/5324694/5586766'
nii = '5324694_5586766_BH_Ax_LAVA-xv+C__7_0_0_axial_DL_liver.nii'
nii_path = os.path.join(path, nii)
nrrd_path = nii_path.split('.')[0] + '.nrrd'

itk_image = nib.load(nii_path).get_fdata()
print(itk_image.shape)

nrrd.write(nrrd_path, itk_image)