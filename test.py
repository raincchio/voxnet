import nibabel as nib
import skimage.io as io
from skimage.color import gray2rgb
import numpy as np
import SimpleITK as sitk
# img=nib.load('data/training_axial_crop_pat0.nii.gz')
# img_arr=img.get_fdata()
id=17
vol_path = 'test/testing_axial_crop_pat'+ str(id) + ".nii.gz"
vol_src = sitk.GetArrayFromImage(sitk.ReadImage(vol_path))

n = 0
for img_ in vol_src:
    img_ = img_ / img_.max() * 255
    img_ = img_.astype(np.uint8)
    io.imsave('ori_label/ori_label' + str(n) + '.png', img_)
    n += 1

# vote_label
# pri_1 = np.where(vote_label==1, 1 ,0)
# pri_2 = np.where(vote_label==2, 1 ,0)
#
# ori_1 = np.where(vol_label==1, 1 ,0)
# ori_2 = np.where(vol_label==2, 1 ,0)
# np.sum(pri_1[ori_1 == 1]) * 2.0 / (np.sum(pri_1) + np.sum(ori_1))
#vote_label[1]
# for img_ in img_arr:
#
#     img_ = img_/2*255
#
#     # img_ = np.squeeze(img_arr[50])
#     io.imsave('ori/ori'+str(n)+'.png', img_)
#     n += 1
#     # io.imshow(vol_src[60])
#     # io.show()
#
#     io.imshow(patch_list_avg[100][5])
