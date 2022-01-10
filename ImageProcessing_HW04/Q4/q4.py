import numpy as np
import cv2
import skimage.feature
from skimage import color as skcolor
from matplotlib import pyplot as plt
import scipy.ndimage as ndi


def get_mask(img, indexes, image_mask, h, w):
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    mask = np.zeros((img.shape[0], img.shape[1])).astype('bool')
    for i, ind in enumerate(indexes):
        img_mask = image_mask[ind[0]: ind[0] + h, ind[1]:ind[1] + w]
        new_mask = np.ones(img_mask.shape, np.uint8) * 2
        new_mask[img_mask >= 240] = 1
        new_mask[:, 0:5] = 0
        new_mask[:, -5:] = 0
        new_mask[0:5, :] = 0
        new_mask[-5:, :] = 0
        bird_image = img[ind[0]: ind[0] + h, ind[1]:ind[1] + w, :]
        bird_mask_grabcut, _, _ = cv2.grabCut(bird_image, new_mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
        bird_mask_grabcut = np.where((bird_mask_grabcut == 2) | (bird_mask_grabcut == 0), 0, 1)
        mask[ind[0]: ind[0] + h, ind[1]:ind[1] + w] = mask[ind[0]: ind[0] + h, ind[1]:ind[1] + w] | bird_mask_grabcut
        mask[ind[0] + h:ind[0] + h, ind[1]:ind[1] + w] = True
    return mask


def match_template(img, temp, threshold):
    matched_interval = cv2.matchTemplate(img, temp, cv2.TM_CCOEFF_NORMED)
    matched_interval[matched_interval < threshold] = 0
    max = skimage.feature.peak_local_max(matched_interval, min_distance=5)
    return max


def hole_filling(edges, template_name, threshold, image):
    template = cv2.imread(template_name)
    h, w, _ = template.shape
    indexes = match_template(image, template, threshold=threshold)

    mask = np.zeros((image.shape[:2])).astype('bool')
    for ind in indexes:
        bird_edge = edges[ind[0]: ind[0] + h, ind[1]:ind[1] + w]
        fill_holes = ndi.binary_fill_holes(bird_edge)
        mask[ind[0]: ind[0] + h, ind[1]:ind[1] + w] = mask[ind[0]: ind[0] + h, ind[1]:ind[1] + w] | fill_holes
        mask[ind[0] + h:ind[0] + h, ind[1]:ind[1] + w] = True
    return mask


def find_birds_grab_cut(file_name):
    image = cv2.imread('birds.jpg')
    image_mask = cv2.imread('birds_mask.jpg', 0)
    template = cv2.imread('sample1.jpg')
    h, w, _ = template.shape
    indexes = match_template(image, template, threshold=0.61)
    mask1 = get_mask(image, indexes, image_mask, h, w)

    template = cv2.imread('sample2.jpg')
    h, w, _ = template.shape
    indexes = match_template(image, template, threshold=0.9)
    mask2 = get_mask(image, indexes, image_mask, h, w)
    mask = mask1 | mask2

    color_mask = np.zeros_like(mask)
    color_mask[mask] = 20
    image = cv2.imread('birds.jpg', 0)
    # imaged = image * mask[:, :, np.newaxis]
    imaged = skcolor.label2rgb(label=color_mask, image=image, bg_label=0)
    imaged = imaged / imaged.max() * 255
    cv2.imwrite(file_name, imaged)


def find_birds_hole_filling(file_name):
    image = cv2.imread('birds.jpg')
    image = cv2.blur(image, (5, 5))
    edges = cv2.Canny(image, 20, 20)
    edges = cv2.blur(edges, (5, 5))

    mask = hole_filling(edges, 'sample1.jpg', 0.62, image)
    # plt.imshow(mask)
    # plt.show()

    mask1 = hole_filling(edges, 'sample5.jpg', 0.67, image)
    # plt.imshow(mask1)
    # plt.show()
    mask = mask | mask1

    mask2 = hole_filling(edges, 'sample2.jpg', 0.9, image)
    # plt.imshow(mask2)
    # plt.show()
    mask = mask | mask2

    mask3 = hole_filling(edges, 'sample3.jpg', 0.71, image)
    # plt.imshow(mask3)
    # plt.show()
    mask = mask | mask3

    # plt.imshow(mask)
    # plt.show()
    color_mask = np.zeros_like(mask)
    color_mask[mask] = 20
    image = cv2.imread('birds.jpg', 0)
    imaged = skcolor.label2rgb(label=color_mask, image=image, bg_label=0)
    imaged = imaged / imaged.max() * 255
    cv2.imwrite(file_name, imaged)


# find_birds_grab_cut('res10.jpg')
find_birds_hole_filling('res10-hole-filling.jpg')
