import math
import random
from surface_distance.metrics import *
import numpy as np
from scipy.ndimage import label, binary_fill_holes
from skimage.measure import regionprops
from skimage.transform import resize
import skimage.io as io
import os
import tensorflow as tf


def optimistic_restore(session, save_file, graph=tf.get_default_graph()):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    for var_name, saved_var_name in var_names:
        curr_var = graph.get_tensor_by_name(var_name)
        var_shape = curr_var.get_shape().as_list()
        if var_shape == saved_shapes[saved_var_name]:
            restore_vars.append(curr_var)
    opt_saver = tf.train.Saver(restore_vars)
    opt_saver.restore(session, save_file)


def frame_similarity(x,y,name="frame_similarity"):
    # loss = 0
    # for i in range(1,64):
    #     for j in range(1,64):
    #         for k in range(1,64):
    #             x_ = tf.reshape(x[:,i-1:i+2,j-1:j+2,k-1:k+2],[-1,9])
    #             y_ = tf.reshape(y[:,i-1:i+2,j-1:j+2,k-1:k+2],[-1,9])
    #             loss += tf.losses.cosine_distance(x_, y_, axis=1)
    #             print(i,j,k)
    # return loss

    x_= tf.reshape(x,[-1,64*64*64])
    y_ = tf.reshape(y,[-1,64*64*64])

    return tf.losses.cosine_distance(x_, y_, axis=1)*0.001

    # np.sum(a * np.log(a / b), axis=1)

def kl_divergence(x,y,name="kl_divergence"):
    # np.sum(a * np.log(a / b), axis=1)
    x = tf.reshape(x,[-1,3])
    y = tf.reshape(y,[-1,3])
    return tf.reduce_mean(x*tf.log(x/(y+1e-8)))

def pre_process_isotropic(img_nii, seg_nii, use_isotropic, id, **kwargs):
    # Pre-process the data
    # Normalize the intensity of the images and also do some re-sample.
    #
    # If use_isotropic is True, we will resize all samples to same resolution,
    # otherwise, we will only resize the sample 4 and 5 (which have much low
    # resolution than other samples)

    if len(kwargs) < 4:
        id = 1

    if use_isotropic:
        scale = np.array(img_nii.shape) / np.array([0.80, 0.7588, 0.7588])
        new_resize = round(img_nii.shape * scale)
        img = imresize3d(img_nii, [], new_resize, 'reflect')
        seg = imresize3d(seg_nii, [], new_resize, 'reflect')
    else:
        if id == 4 or id == 5:
            # resize img in up an down plane
            img = np.transpose(img_nii, [2, 1, 0])
            img = resize(img, 1.5, mode='reflect')
            img = np.transpose(img, [2, 1, 0])
            seg = np.transpose(seg_nii, [2, 1, 0])
            seg = resize(seg, 1.5, 'reflect')
            seg = np.transpose(seg, [2, 1, 0])
        else:
            img = img_nii
            seg = seg_nii

    # Normalize the intensity of images
    mask = img > 0
    mean_value = np.mean(img[mask])
    std_value = np.std(img[mask])
    img = (img - mean_value) / std_value
    img = np.float32(img)
    seg = np.uint(seg)

    return img, seg


def imresize3d(image, scale, tsize, npad='reflect'):
    """
    This function resizes a 3D image volume to new dimensions
    :param image: The input image volume
    :param scale: scaling factor, when used set tsize to []
    :param tsize: new dimensions, when used set scale to []
    :param ntype: Type of interpolation ('nearest', 'linear', 'cubic')
    :param npad: Boundary condition ('replicate', 'symmetric', 'circular', 'fill', or 'bound')
    :return: The resized image volume
    """

    # Check the inputs
    if tsize == []:
        tsize = round(image.shape * scale)
    if scale == []:
        scale = tsize / image.shape

    new_image = resize(image, tsize, mode=npad)  # need to update

    return new_image


def create_ROIs(img, seg, patchSize):
    """
    Create the ROIs needed, x*y*z
    Also randomly crop patches to increase background information
    :param img:
    :param seg:
    :param patchSize:
    :return:
    """
    # the agumented size of surrounding context
    aug_r = 20

    # get bounding box
    s1, s2, s3 = seg.shape
    positions = regionprops(np.int8(seg > 0))
    position = positions[0].bbox
    cx = int(max(math.ceil(position[0]) - aug_r / 2, 0))
    cy = int(max(math.ceil(position[1]) - aug_r / 2, 0))
    cz = int(max(math.ceil(position[2]) - aug_r / 2, 0))
    x = int(min(math.ceil(position[3] - position[0]) + aug_r, s1 - cx))
    y = int(min(math.ceil(position[4] - position[1]) + aug_r, s2 - cy))
    z = int(min(math.ceil(position[5] - position[2]) + aug_r, s3 - cz))

    # crop ROI
    new_img = img[cx: cx + x, cy: cy + y, cz: cz + z]  # need to be validated the dimensions
    new_seg = seg[cx: cx + x, cy: cy + y, cz: cz + z]

    # If the size of ROI is smaller than patchSize, padding the ROI to patchSize*patchSize*patchSize
    padding_size_x = max(0, math.ceil((patchSize - img.shape[0]) / 2))
    padding_size_y = max(0, math.ceil((patchSize - img.shape[1]) / 2))
    padding_size_z = max(0, math.ceil((patchSize - img.shape[2]) / 2))
    new_img = np.pad(new_img, (
        (padding_size_x, padding_size_x), (padding_size_y, padding_size_y), (padding_size_z, padding_size_z)),
                     'constant')
    new_seg = np.pad(new_seg, (
        (padding_size_x, padding_size_x), (padding_size_y, padding_size_y), (padding_size_z, padding_size_z)),
                     'constant',
                     constant_values=100)

    # randomly crop patches from the whole volumes
    rimgs = []
    rsegs = []
    for k in range(1):
        sx, sy, sz = img.shape
        cx = int(math.floor(random.random() * (sx - x)))
        cy = int(math.floor(random.random() * (sy - y)))
        cz = int(max(math.floor(random.random() * (sz - z)), 0))
        random_per_imgs = img[cx: cx + x, cy: cy + y, cz: cz + z]
        random_per_seg1 = seg[cx: cx + x, cy: cy + y, cz: cz + z]
        rimgs.append(random_per_imgs)
        rsegs.append(random_per_seg1)

    return new_img, new_seg, rimgs, rsegs


def generate_score_map(prob, img, patchSize, ita):
    # set parameter
    tr_mean = 0

    # pad the img if the size is smaller than the crop size
    padding_size_x = max(0, math.ceil((patchSize - img.shape[0]) / 2))
    padding_size_y = max(0, math.ceil((patchSize - img.shape[1]) / 2))
    padding_size_z = max(0, math.ceil((patchSize - img.shape[2]) / 2))
    img = np.pad(img,
                 ((padding_size_x, padding_size_x), (padding_size_y, padding_size_y), (padding_size_z, padding_size_z)),
                 'constant')

    data = img - tr_mean
    score = np.zeros((img.shape[0], img.shape[1], img.shape[2], 3))
    cnt = np.zeros((img.shape[0], img.shape[1], img.shape[2]))

    ss_h, ss_w, ss_l = img.shape
    fold_h = math.floor(ss_h / patchSize) + ita
    fold_w = math.floor(ss_w / patchSize) + ita
    fold_l = math.floor(ss_l / patchSize) + ita
    overlap_h = int(math.ceil((ss_h - patchSize) / (fold_h - 1)))
    overlap_w = int(math.ceil((ss_w - patchSize) / (fold_w - 1)))
    overlap_l = int(math.ceil((ss_l - patchSize) / (fold_l - 1)))
    idx_h = [i for i in range(0, ss_h - patchSize + 1, overlap_h)]
    idx_h.append(ss_h - patchSize)
    idx_w = [i for i in range(0, ss_w - patchSize + 1, overlap_w)]
    idx_w.append(ss_w - patchSize)
    idx_l = [i for i in range(0, ss_l - patchSize + 1, overlap_l)]
    idx_l.append(ss_l - patchSize)

    for itr_h in idx_h:
        for itr_w in idx_w:
            for itr_l in idx_l:
                crop_data = data[itr_h: itr_h + patchSize, itr_w: itr_w + patchSize, itr_l: itr_l + patchSize]
                """add the forward"""
                res_score = prob
                cnt[itr_h: itr_h + patchSize, itr_w: itr_w + patchSize, itr_l: itr_l + patchSize] += 1
                score[itr_h: itr_h + patchSize, itr_w: itr_w + patchSize, itr_l: itr_l + patchSize, :] += res_score
    score_map = score / np.tile(cnt, (1, 1, 3)).reshape([cnt.shape[0], cnt.shape[1], cnt.shape[2], 3])

    # remove the margin of padding
    score_map = score_map[padding_size_x: score_map.shape[0] - padding_size_x,
                padding_size_y: score_map.shape[1] - padding_size_y,
                padding_size_z: score_map.shape[2] - padding_size_z, :]

    return score_map


def generate_score_map_partition(img, patchSize, ita):
    # set parameter
    tr_mean = 0

    # pad the img if the size is smaller than the crop size
    padding_size_x = max(0, math.ceil((patchSize - img.shape[0]) / 2))
    padding_size_y = max(0, math.ceil((patchSize - img.shape[1]) / 2))
    padding_size_z = max(0, math.ceil((patchSize - img.shape[2]) / 2))
    img = np.pad(img,
                 ((padding_size_x, padding_size_x), (padding_size_y, padding_size_y), (padding_size_z, padding_size_z)),
                 'constant')

    data = img - tr_mean

    ss_h, ss_w, ss_l = img.shape
    fold_h = math.floor(ss_h / patchSize) + ita
    fold_w = math.floor(ss_w / patchSize) + ita
    fold_l = math.floor(ss_l / patchSize) + ita
    overlap_h = int(math.ceil((ss_h - patchSize) / (fold_h - 1)))
    overlap_w = int(math.ceil((ss_w - patchSize) / (fold_w - 1)))
    overlap_l = int(math.ceil((ss_l - patchSize) / (fold_l - 1)))
    idx_h = [i for i in range(0, ss_h - patchSize + 1, overlap_h)]
    idx_h.append(ss_h - patchSize)
    idx_w = [i for i in range(0, ss_w - patchSize + 1, overlap_w)]
    idx_w.append(ss_w - patchSize)
    idx_l = [i for i in range(0, ss_l - patchSize + 1, overlap_l)]
    idx_l.append(ss_l - patchSize)

    crop_data_list = []
    for itr_h in idx_h:
        for itr_w in idx_w:
            for itr_l in idx_l:
                crop_data = data[itr_h: itr_h + patchSize, itr_w: itr_w + patchSize, itr_l: itr_l + patchSize]
                crop_data_list.append(crop_data)

    return crop_data_list, ss_h, ss_w, ss_l, padding_size_x, padding_size_y, padding_size_z


def generate_score_map_patch2Img(patch_list, ss_h, ss_w, ss_l, padding_size_x, padding_size_y, padding_size_z,
                                 patchSize, ita):
    score = np.zeros((ss_h, ss_w, ss_l, 3))
    cnt = np.zeros((ss_h, ss_w, ss_l))

    fold_h = math.floor(ss_h / patchSize) + ita
    fold_w = math.floor(ss_w / patchSize) + ita
    fold_l = math.floor(ss_l / patchSize) + ita
    overlap_h = int(math.ceil((ss_h - patchSize) / (fold_h - 1)))
    overlap_w = int(math.ceil((ss_w - patchSize) / (fold_w - 1)))
    overlap_l = int(math.ceil((ss_l - patchSize) / (fold_l - 1)))
    idx_h = [i for i in range(0, ss_h - patchSize + 1, overlap_h)]
    idx_h.append(ss_h - patchSize)
    idx_w = [i for i in range(0, ss_w - patchSize + 1, overlap_w)]
    idx_w.append(ss_w - patchSize)
    idx_l = [i for i in range(0, ss_l - patchSize + 1, overlap_l)]
    idx_l.append(ss_l - patchSize)

    p_count = 0
    for itr_h in idx_h:
        for itr_w in idx_w:
            for itr_l in idx_l:
                res_score = patch_list[p_count]
                cnt[itr_h: itr_h + patchSize, itr_w: itr_w + patchSize, itr_l: itr_l + patchSize] += 1
                score[itr_h: itr_h + patchSize, itr_w: itr_w + patchSize, itr_l: itr_l + patchSize, :] += res_score
    score_map = score / np.tile(cnt, (1, 1, 3)).reshape([cnt.shape[0], cnt.shape[1], cnt.shape[2], 3])

    # remove the margin of padding
    score_map = score_map[padding_size_x: score_map.shape[0] - padding_size_x,
                padding_size_y: score_map.shape[1] - padding_size_y,
                padding_size_z: score_map.shape[2] - padding_size_z, :]

    # return score_map
    return np.argmax(score_map, axis=3)


def partition_Img(srcImg, patchSize, ita):
    r, c, h = srcImg.shape

    # patch number and overlap in first dimension
    r_fold = r // patchSize
    # r_mod = r % patchSize
    r_fold += ita
    r_overlap = int(math.ceil(float(r_fold * patchSize - r) / (r_fold - 1)))

    # patch number and overlap in second dimension
    c_fold = c // patchSize
    # c_mod = c % patchSize
    c_fold += ita
    c_overlap = int(math.ceil(float(c_fold * patchSize - c) / (c_fold - 1)))

    # patch number and overlap in three dimension
    h_fold = h // patchSize
    # h_mod = h % patchSize
    h_fold += ita
    h_overlap = int(math.ceil(float(h_fold * patchSize - h) / (h_fold - 1)))

    # print("%d, %d, %d\n" % (r_overlap, c_overlap, h_overlap))

    # partition into patches
    p_count = 0
    patch_list = []
    for R in range(0, r_fold):
        r_s = int(R * patchSize - R * r_overlap)
        r_e = r_s + patchSize
        for C in range(0, c_fold):
            c_s = int(C * patchSize - C * c_overlap)
            c_e = c_s + patchSize
            for H in range(0, h_fold):
                h_s = int(H * patchSize - H * h_overlap)
                h_e = h_s + patchSize
                patch_list.append(srcImg[r_s: r_e, c_s: c_e, h_s: h_e])
                p_count += 1

    return patch_list, r, c, h


def patches2Img_vote(patch_list, r, c, h, patchSize, ita):
    fusion_Img = np.zeros((r, c, h))
    label_0_array = np.zeros((r, c, h))
    label_1_array = np.zeros((r, c, h))
    label_2_array = np.zeros((r, c, h))
    label_array = np.zeros((r, c, h, 3))

    # patch number and overlap in first dimension
    r_fold = r // patchSize
    # r_mod = r % patchSize
    r_fold += ita
    r_overlap = int(math.ceil(float(r_fold * patchSize - r) / (r_fold - 1)))

    # patch number and overlap in second dimension
    c_fold = c // patchSize
    # c_mod = c % patchSize
    c_fold += ita
    c_overlap = int(math.ceil(float(c_fold * patchSize - c) / (c_fold - 1)))

    # patch number and overlap in thrid dimension
    h_fold = h // patchSize
    # h_mod = h % patchSize
    h_fold += ita
    h_overlap = int(math.ceil(float(h_fold * patchSize - h) / (h_fold - 1)))

    # partition into patches
    p_count = 0
    for R in range(0, r_fold):
        r_s = R * patchSize - R * r_overlap
        r_e = r_s + patchSize
        for C in range(0, c_fold):
            c_s = C * patchSize - C * c_overlap
            c_e = c_s + patchSize
            for H in range(0, h_fold):
                h_s = H * patchSize - H * h_overlap
                h_e = h_s + patchSize

                fusion_Img[r_s: r_e, c_s: c_e, h_s: h_e] = patch_list[p_count]

                # histogram for voting
                idx_0 = np.float16(patch_list[p_count] == 0)
                idx_1 = np.float16(patch_list[p_count] == 1)
                idx_2 = np.float16(patch_list[p_count] == 2)
                label_0_array[r_s: r_e, c_s: c_e, h_s: h_e] += idx_0
                label_1_array[r_s: r_e, c_s: c_e, h_s: h_e] += idx_1
                label_2_array[r_s: r_e, c_s: c_e, h_s: h_e] += idx_2

                p_count += 1

    label_array[:, :, :, 0] = label_0_array
    label_array[:, :, :, 1] = label_1_array
    label_array[:, :, :, 2] = label_2_array

    vote_label = np.argmax(label_array, axis=3)
    # vote_Img = vote_label
    #
    # return fusion_Img, vote_Img

    return vote_label


def Connection_Judge_3D(Binary_Img, reject_T):
    Connect_Elimi = Binary_Img
    # find components
    labeled, C_number = label(Binary_Img, np.ones((3, 3, 3)))
    totalPixels_N = np.sum(np.int8(labeled > 0))
    # remove minor components
    for i in range(C_number):
        idx = (labeled == (i + 1))
        cPixel_N = np.sum(np.int8(idx))
        ratio = float(cPixel_N) / totalPixels_N
        if ratio < reject_T:
            Connect_Elimi[idx] = 0

    return Connect_Elimi


def RemoveMinorCC(SegImg, reject_T=0.2):
    Retain_Img = np.zeros(np.shape(SegImg))

    # class 2
    Img_C2 = np.int8(SegImg == 2)
    Retain_C2 = Connection_Judge_3D(Img_C2, reject_T)
    Retain_C2 = binary_fill_holes(Retain_C2)
    Retain_Img[Retain_C2] = 2

    # class 1
    Img_C1 = np.int8(SegImg == 1)
    Retain_C1 = Connection_Judge_3D(Img_C1, reject_T)
    Retain_C1 = binary_fill_holes(Retain_C1)
    Retain_Img[Retain_C1] = 1

    return Retain_Img


def saveLableImage(vote_label, dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    n = 0
    for img_ in vote_label:
        img_ = img_ / 2*255
        img_ = img_.astype(np.uint8)
        io.imshow(img_)
        io.imsave(dir+'/label' + str(n) + '.png', img_)
        n += 1


# def evaluation(pred, label):
#     dice = []*2
#     asd = []*2
#     has = []*2
#     pri_1 = np.where(pred == 1, 1, 0)
#     pri_2 = np.where(pred == 2, 1, 0)
#
#     ori_1 = np.where(label == 1, 1, 0)
#     ori_2 = np.where(label == 2, 1, 0)
#
#     dice[0] = dice_ratio(pri_1, ori_1)
#     dice[1] = dice_ratio(pri_2, ori_2)
#
#     vol_1 = np.argwhere(pred == 1)
#     vol_2 = np.argwhere(pred == 2)
#
#     vol_ori_1 = np.argwhere(label == 1)
#     vol_ori_2 = np.argwhere(label == 2)
#     has[0] = hausdorff_dist(vol_1, vol_ori_1)
#     has[1] = hausdorff_dist(vol_2, vol_ori_2)
#
#     return dice, asd, has
#
#
def dice_ratio(pred, label):
    '''Note: pred & label should only contain 0 or 1.
    '''

    return np.sum(pred[label == 1]) * 2.0 / (np.sum(pred) + np.sum(label))
#
#
# def hausdorff_dist(vol_a, vol_b):
#     dist_lst = []
#     for idx in range(len(vol_a)):
#         dist_min = 1000.0
#         for idx2 in range(len(vol_b)):
#             dist = np.linalg.norm(vol_a[idx]-vol_b[idx2])
#             if dist_min > dist:
#                 dist_min = dist
#         dist_lst.append(dist_min)
#         return np.max(dist_lst)


def evaluation(pred, label):
    dice = [0]*2
    asd = [0]*2
    has = [0]*2
    pri_1 = np.where(pred == 1, 1, 0)
    pri_2 = np.where(pred == 2, 1, 0)

    ori_1 = np.where(label == 1, 1, 0)
    ori_2 = np.where(label == 2, 1, 0)

    dice[0] = dice_ratio(pri_1, ori_1)
    dice[1] = dice_ratio(pri_2, ori_2)
    #
    # vol_1 = np.argwhere(pred == 1)
    # vol_2 = np.argwhere(pred == 2)
    #
    # vol_ori_1 = np.argwhere(label == 1)
    # vol_ori_2 = np.argwhere(label == 2)
    # has[0] = hausdorff_dist(vol_1, vol_ori_1)
    # has[1] = hausdorff_dist(vol_2, vol_ori_2)

    surface_distances_1 = compute_surface_distances(
        ori_1, pri_1, spacing_mm=(1, 1, 1))
    surface_distances_2 = compute_surface_distances(
        ori_2, pri_2, spacing_mm=(1, 1, 1))

    dice[0] = compute_dice_coefficient(pri_1, ori_1)
    dice[1] = compute_dice_coefficient(pri_2, ori_2)
    asd[0] = compute_average_surface_distance(surface_distances_1)
    asd[1] = compute_average_surface_distance(surface_distances_2)
    has[0] = compute_robust_hausdorff(surface_distances_1,99)
    has[1] = compute_robust_hausdorff(surface_distances_2,99)

    return dice, asd, has


