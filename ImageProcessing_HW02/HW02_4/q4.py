import cv2
import numpy as np
from matplotlib import pyplot as plt


def read(file_name, image_type):
    img = cv2.imread(file_name)
    if image_type == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif image_type == "HSV":
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def align_images(img_one, img_two):
    h1, w1, _ = img_one.shape
    h, w, _ = img_two.shape
    src_points = np.array(
        [[324, 488], [597, 488], [466, 694], [463, 990], [463, 56], [153, 424], [802, 432], [0, 0], [0, h1 - 1],
         [w1 - 1, 0], [w1 - 1, h1 - 1]], dtype=float)
    dst_points = np.array(
        [[341, 511], [606, 514], [484, 718], [481, 989], [487, 56], [178, 426], [825, 430], [0, 0], [0, h - 1],
         [w - 1, 0], [w - 1, h - 1]], dtype=float)
    transform, f = cv2.findHomography(src_points, dst_points)
    img_one = cv2.warpPerspective(img_one, transform, (w, h))
    return img_one


def process(img_one, img_two, s_h, s_l):
    h, w = img_one.shape
    high_kernel = get_kernel(h, w, s_h, "High_pass")
    low_kernel = get_kernel(h, w, s_l, "Low_pass")
    img_f1, res_img1, img_back1 = apply_kernel(img_one, high_kernel)
    img_f2, res_img2, img_back2 = apply_kernel(img_two, low_kernel)
    return img_f1, res_img1, img_back1, img_f2, res_img2, img_back2, high_kernel, low_kernel


def apply_kernel(img, kernel):
    img_f = np.fft.fft2(img)
    img_shifted = np.fft.fftshift(img_f)
    res_img = img_shifted * kernel
    res_shifted = np.fft.ifftshift(res_img)
    res_back = np.fft.ifft2(res_shifted)
    return img_shifted, res_img, res_back


def get_kernel(rows, cols, s, kernel_type):
    m, n = np.ceil(rows / 2).astype(int), np.ceil(cols / 2).astype(int)
    if kernel_type == "High_pass":
        return np.array([[1 - gaussian(m, n, i, j, s) for j in range(cols)] for i in range(rows)])
    else:
        return np.array([[gaussian(m, n, i, j, s) for j in range(cols)] for i in range(rows)])


def gaussian(m, n, i, j, sigma):
    return np.exp(-((i - m) ** 2 + (j - n) ** 2) / (2 * sigma ** 2))


def save_images(r_f1, g_f1, b_f1, r_res1, g_res1, b_res1, r_back1, g_back1, b_back1, r_f2, g_f2, b_f2, r_res2, g_res2,
                b_res2, r_back2, g_back2, b_back2, high_pass, low_pass, h, w):
    image_one_f = np.dstack((r_f1, g_f1, b_f1))
    image_one_f = np.log(np.abs(image_one_f))
    image_one_f = cv2.normalize(image_one_f, 0, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite("res23-dft-near.jpg", image_one_f)
    image_two_f = np.dstack((r_f2, g_f2, b_f2))
    image_two_f = np.log(np.abs(image_two_f))
    image_two_f = cv2.normalize(image_two_f, 0, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite("res24-dft-far.jpg", image_two_f)
    high_pass = cv2.normalize(high_pass, 0, 0, 255, cv2.NORM_MINMAX)
    low_pass = cv2.normalize(low_pass, 0, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite("res25-highpass.jpg", high_pass)
    cv2.imwrite("res26-lowpass.jpg", low_pass)
    r_res_one = np.dstack((r_res1, g_res1, b_res1))
    r_res_one = np.abs(r_res_one)
    r_res_one = cv2.normalize(r_res_one, 0, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite("res27-highpassed.jpg", r_res_one)
    r_res_two = np.dstack((r_res2, g_res2, b_res2))
    r_res_two = np.abs(r_res_two)
    r_res_two = cv2.normalize(r_res_two, 0, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite("res28-lowpassed.jpg", r_res_two)
    res = high_mul * r_res_one + low_mul * r_res_two
    res = cv2.normalize(res, 0, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite("res29-hybrid.jpg", res)
    r_back1, g_back1, b_back1 = np.real(r_back1), np.real(g_back1), np.real(b_back1)
    r_back2, g_back2, b_back2 = np.real(r_back2), np.real(g_back2), np.real(b_back2)
    res1 = np.dstack((r_back1, g_back1, b_back1))
    res2 = np.dstack((r_back2, g_back2, b_back2))
    res = high_mul * res1 + low_mul * res2
    res = cv2.normalize(res, 0, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite("res30-hybrid-near.jpg", res)
    mul = 16
    res = cv2.resize(res, (h // mul, w // mul))
    cv2.imwrite("res31-hybrid-far.jpg", res)


image_one = cv2.imread("res19-near.jpg")
image_two = cv2.imread("res20-far.jpg")
image_one = align_images(image_one, image_two)
cv2.imwrite("res21-near.jpg", image_one)
cv2.imwrite("res22-far.jpg", image_two)
r1, g1, b1 = cv2.split(image_one)
r2, g2, b2 = cv2.split(image_two)
r, s = 20, 10
high_mul, low_mul = 0.35, 0.65
r_f1, r_res1, r_back1, r_f2, r_res2, r_back2, high_pass, low_pass = process(r1, r2, r, s)
g_f1, g_res1, g_back1, g_f2, g_res2, g_back2, high_pass, low_pass = process(g1, g2, r, s)
b_f1, b_res1, b_back1, b_f2, b_res2, b_back2, high_pass, low_pass = process(b1, b2, r, s)
h, w, _ = image_one.shape
save_images(r_f1, g_f1, b_f1, r_res1, g_res1, b_res1, r_back1, g_back1, b_back1, r_f2, g_f2, b_f2, r_res2, g_res2,
            b_res2, r_back2, g_back2, b_back2, high_pass, low_pass, h, w)
