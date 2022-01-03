import cv2
import numpy as np
import matplotlib.pylab as plt


def read(file_name, image_type):
    img = cv2.imread(file_name)
    if image_type == "":
        return img
    elif image_type == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif image_type == "HSV":
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def warp_perspective(img, t_inv, size):
    w, h = size
    result = np.zeros((h, w, 3))
    h_i, w_i, d_i = img.shape
    for j in range(h):
        for i in range(w):
            res = np.dot(t_inv, [i, j, 1])
            i_2, j_2, k = res
            i_2, j_2 = i_2 / k, j_2 / k
            if 1 < j_2 < h_i - 1 and 1 < i_2 < w_i - 1:
                x = np.floor(i_2).astype(int)
                y = np.floor(j_2).astype(int)
                a, b = i_2 - x, j_2 - y
                a_m, b_m = [1 - a, a], [1 - b, b]
                result[j, i, 0] = np.dot(np.dot(a_m, get_mtr(img, y, x, 0)), b_m)
                result[j, i, 1] = np.dot(np.dot(a_m, get_mtr(img, y, x, 1)), b_m)
                result[j, i, 2] = np.dot(np.dot(a_m, get_mtr(img, y, x, 2)), b_m)
    return result


def get_mtr(img, x, y, d):
    return [[img[x, y, d], img[x + 1, y, d]], [img[x, y + 1, d], img[x + 1, y + 1, d]]]


def get_length(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def process(src, p1, p2, p3, p4, file_name):
    src_points = np.array([p1, p2, p3, p4], dtype=float)
    h = ((get_length(p1, p2) + get_length(p3, p4)) / 2).astype(int)
    w = ((get_length(p1, p3) + get_length(p2, p4)) / 2).astype(int)
    dst_points = np.array([[0, 0], [h - 1, 0], [0, w - 1], [h - 1, w - 1]], dtype=float)
    transform, f = cv2.findHomography(src_points, dst_points)
    transform_inv = np.linalg.inv(transform)
    dst = warp_perspective(src, transform_inv, (h, w))
    cv2.imwrite(file_name, dst.astype('uint8'))


img_src = read('books.jpg', "")
process(img_src, [666., 215.], [600., 394.], [385., 112.], [318., 289.], "res16.jpg")
process(img_src, [347., 740.], [153., 709.], [395., 465.], [205., 428.], "res17.jpg")
process(img_src, [805., 973.], [609., 1098.], [613., 674.], [420., 796.], "res18.jpg")
