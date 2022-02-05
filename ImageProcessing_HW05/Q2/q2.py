import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve


def get_process_regions(src, tar, msk, p_tar, p_src, h, w):
    src_region = src[p_src[0]:p_src[0] + h, p_src[1]:p_src[1] + w].copy()
    tar_region = tar[p_tar[0]:p_tar[0] + h, p_tar[1]:p_tar[1] + w].copy()
    mask_region = msk[p_tar[0]:p_tar[0] + h, p_tar[1]:p_tar[1] + w].copy()
    return src_region, tar_region, mask_region


def get_matrices(laplacian, number_of_pixels, mask_region, width, src_region):
    diagonal_value = np.ones((1, number_of_pixels))
    domain_indexes = np.argwhere(mask_region > 0.01)
    domain_indexes = domain_indexes[:, 0] * width + domain_indexes[:, 1]
    diagonal_value[0, domain_indexes] = 4
    a = sparse.lil_matrix((number_of_pixels, number_of_pixels))
    a.setdiag(diagonal_value[0, :])
    a[domain_indexes, domain_indexes - width] = -1
    a[domain_indexes, domain_indexes - 1] = -1
    a[domain_indexes, domain_indexes + 1] = -1
    a[domain_indexes, domain_indexes + width] = -1
    b = src_region.copy().reshape((number_of_pixels, 3))
    b[domain_indexes] = -laplacian.reshape((number_of_pixels, 3))[domain_indexes]
    return a.tocsr(), b


def solve_system(a, b, height, width, src, p, h, w):
    result = src.copy()
    res_region = spsolve(a, b)
    res_region = res_region.reshape((height, width, 3))
    result[p[0]:p[0] + h, p[1]:p[1] + w] = res_region
    return result


def process(src, tar, msk, p_src, p_tar, h, w):
    src_region, tar_region, mask_region = get_process_regions(src, tar, msk, p_src, p_tar, h, w)
    height, width, _ = src_region.shape
    number_of_pixels = height * width
    laplacian = cv2.Laplacian(tar_region, cv2.CV_64F)
    a, b = get_matrices(laplacian, number_of_pixels, mask_region, width, src_region)
    result = solve_system(a, b, height, width, src, p_tar, h, w)
    result[result < 0] = 0
    result[result > 255] = 255
    return result


source = cv2.imread('res05.jpg').astype(np.float64)
target = cv2.imread('res06.jpg').astype(np.float64)
mask = cv2.imread('mask.jpg', 0)
result = process(target, source, mask, [148, 195], [148, 195], 150, 150)
cv2.imwrite('res07.jpg', result)

# source = cv2.imread('img1.jpg').astype(np.float64)
# target = cv2.imread('img2.jpg').astype(np.float64)
# mask = cv2.imread('mask2.jpg', 0)
# result = process(target, source, mask, [55, 55], [420, 720], 230, 450)
# cv2.imwrite('res3.jpg', result)

