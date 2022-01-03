import sys

import cv2
import numpy as np
import skimage.transform as skimage
import skimage.io as sk
from matplotlib import pyplot as plt


def process(image, file_name):
    b, g, r = slice_images(image)
    r2g_1, r2g_2, images_r2g, offsets_r2g = pyramid(g, r)
    b2g_1, b2g_2, images_b2g, offsets_b2g = pyramid(g, b)
    # save_all_images(images_b2g, images_r2g, offsets_b2g, offsets_r2g, file_name)
    result = auto_crop(image, r2g_1, r2g_2, b2g_1, b2g_2)
    result = (result / 256).astype('uint8')
    cv2.imwrite(file_name + ".jpeg", result)


def save_all_images(images_b2g, images_r2g, offsets_b2g, offsets_r2g, file_name):
    for i in range(4):
        img = get_matched_image(images_b2g[i], images_r2g[i], offsets_b2g[i], offsets_r2g[i])
        img = (img * 255).astype('uint8')
        cv2.imwrite("{file_name}_layer{i}.jpeg".format(file_name=file_name, i=i), img)


def get_matched_image(b_tuple, r_tuple, b_offset, r_offset):
    g, b = b_tuple
    g, r = r_tuple
    b_1, b_2 = b_offset
    r_1, r_2 = r_offset
    b, r = roll_images(b, r, b_1, b_2, r_1, r_2)
    b = skimage.resize(b, r.shape)
    return cv2.merge((b, g, r))


def slice_images(img):
    h, w = img.shape
    h = int(h / 3)
    h_border, w_border = 160, 180
    img_1 = img[h_border: h - h_border, w_border: -w_border]
    img_2 = img[h + h_border: 2 * h - h_border, w_border: -w_border]
    img_3 = img[2 * h + h_border: 3 * h - h_border, w_border: -w_border]
    return img_1, img_2, img_3


def roll_images(b, g, b_1, b_2, g_1, g_2):
    b = np.roll(b, (b_1, b_2), axis=(1, 0))
    g = np.roll(g, (g_1, g_2), axis=(1, 0))
    return b, g


def get_height_border(img, white_thresh, black_thresh):
    l = int(img.shape[0] * 0.1 / 3)
    values_1 = []
    values_2 = []
    for i in range(l):
        value_1 = np.average(img[i, :, ])
        value_2 = np.average(img[-i, :, ])
        if value_1 > white_thresh or value_1 < black_thresh:
            values_1.append(i)
        if value_2 > white_thresh or value_2 < black_thresh:
            values_2.append(i)
    return np.max(values_1), np.max(values_2)


def get_width_border(img, white_thresh, black_thresh):
    l = int(img.shape[1] * 0.1)
    values_1 = []
    values_2 = []
    for i in range(l):
        value_1 = np.average(img[:, i, ])
        value_2 = np.average(img[:, -i, ])
        if value_1 > white_thresh or value_1 < black_thresh:
            values_1.append(i)
        if value_2 > white_thresh or value_2 < black_thresh:
            values_2.append(i)
    return np.max(values_1), np.max(values_2)


def crop_borders(img, r, g, b):
    white_thresh = 225 * 256
    black_thresh = 40 * 256
    safety_limit = 7
    h_border_1, h_border_2 = get_height_border(img, white_thresh, black_thresh)
    w_border_1, w_border_2 = get_width_border(img, white_thresh, black_thresh)
    r = r[h_border_1 + safety_limit: -h_border_2 - safety_limit, w_border_1 + safety_limit: -w_border_2 - safety_limit]
    g = g[h_border_1 + safety_limit: -h_border_2 - safety_limit, w_border_1 + safety_limit: -w_border_2 - safety_limit]
    b = b[h_border_1 + safety_limit: -h_border_2 - safety_limit, w_border_1 + safety_limit: -w_border_2 - safety_limit]
    return r, g, b


def auto_crop(img, r2g_1, r2g_2, b2g_1, b2g_2):
    r_1, r_2 = r2g_1, r2g_2
    b_1, b_2 = b2g_1, b2g_2

    h, w = img.shape
    h = int(h / 3)
    b = img[: h, :]
    g = img[h: 2 * h, :]
    r = img[2 * h: 3 * h, :]

    b, r = roll_images(b, r, b_1, b_2, r_1, r_2)
    print("b_offset: ", b_1, b_2)
    print("r_offset: ", r_1, r_2)

    h, w = r.shape
    h_1 = max(r_2, b_2, 0)
    h_2 = h + min(r_2, b_2, 0)
    w_1 = max(r_1, b_1, 0)
    w_2 = w + min(r_1, b_1, 0)

    r = r[h_1: h_2, w_1:w_2]
    g = g[h_1: h_2, w_1:w_2]
    b = b[h_1: h_2, w_1:w_2]

    r, g, b = crop_borders(img, r, g, b)
    return cv2.merge((b, g, r))


def align_images(img_1, img_2):
    offset_1, offset_2 = 0, 0
    min_diff = sys.maxsize
    for i in range(-5, 5):
        for j in range(-5, 5):
            updated_img_1, updated_img_2 = get_overlapped(img_1, img_2, i, j)
            diff = norm_2(updated_img_1, updated_img_2)
            if diff < min_diff:
                min_diff = diff
                offset_1 = i
                offset_2 = j
    return offset_1, offset_2


def norm_2(img_1, img_2):
    return np.sum((img_1 - img_2) ** 2)


def norm_1(img_1, img_2):
    return np.sum(img_1 - img_2)


def pyramid(img_1, img_2):
    images, offsets = [], []
    offset_1, offset_2 = 0, 0
    layers = 5
    scale_num = 2 ** layers
    shifted_1, shifted_2 = img_1, img_2
    for i in range(layers):
        shifted_1, shifted_2 = rescale_images(shifted_1, shifted_2, scale_num)
        update_offset_1, update_offset_2 = align_images(shifted_1, shifted_2)
        images.append((shifted_1, shifted_2))
        offsets.append((update_offset_1, update_offset_2))
        offset_1 += scale_num * update_offset_1
        offset_2 += scale_num * update_offset_2
        scale_num = int(scale_num / 2)
        shifted_1, shifted_2 = get_overlapped(img_1, img_2, offset_1, offset_2)
    return offset_1, offset_2, images, offsets


def get_overlapped(img_1, img_2, offset_1, offset_2):
    h, w = img_1.shape
    i, j = offset_1, offset_2
    if i >= 0 and j >= 0:
        return img_1[j:, i:], img_2[:h - j, :w - i]
    elif i >= 0 and j < 0:
        return img_1[:h + j, i:], img_2[-j:, :w - i]
    elif i < 0 and j >= 0:
        return img_1[j:, :w + i], img_2[:h - j, -i:]
    elif i < 0 and j < 0:
        return img_1[:h + j, :w + i], img_2[-j:, -i:]


def rescale_images(img_1, img_2, scale_num):
    img_1 = skimage.rescale(img_1, 1 / scale_num)
    img_2 = skimage.rescale(img_2, 1 / scale_num)

    if img_2.shape[0] > img_1.shape[0]:
        img_2 = img_2[1:, :]
    elif img_2.shape[0] < img_1.shape[0]:
        img_1 = img_1[1:, :]

    if img_2.shape[1] > img_1.shape[1]:
        img_2 = img_2[:, 1:]
    elif img_2.shape[1] < img_1.shape[1]:
        img_1 = img_1[:, 1:]

    return img_1, img_2


image = sk.imread("Amir.tif")
process(image, "res03_Amir")

image = sk.imread("Mosque.tif")
process(image, "res04_Mosque")

image = sk.imread("Train.tif")
process(image, "res05_train")
