import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi


def read(file_name, image_type):
    img = cv2.imread(file_name)
    if image_type == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif image_type == "HSV":
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def task_one(img, s, k, alpha):
    kernel, blurred, mask, img = sharpen_image_one(img, s, k, alpha)
    mask = mask + 128

    mat = np.zeros((101, 101))
    mat[50, 50] = 255
    kernel = cv2.filter2D(mat, ddepth=-1, kernel=kernel)
    kernel = cv2.normalize(kernel, 0, 0, 255, cv2.NORM_MINMAX)

    cv2.imwrite("res01.jpg", kernel)
    cv2.imwrite("res02.jpg", blurred)
    cv2.imwrite("res03.jpg", mask)
    cv2.imwrite("res04.jpg", img)


def sharpen_image_one(img, s, k, alpha):
    img = img.astype('float64')
    kernel = get_gaussian_kernel(s, k)
    blurred = cv2.filter2D(img, ddepth=-1, kernel=kernel)
    mask_1 = img - blurred
    mask = alpha * (img - blurred)
    img = img + alpha * mask
    img[img < 0] = 0
    img[img > 255] = 255
    return kernel, blurred, mask_1, img


def get_gaussian_kernel(s, k):
    n = 2 * k + 1
    return np.array([[gaussian(k, k, i, j, s) / (2 * pi * s ** 2) for j in range(n)] for i in range(n)])


def gaussian(m, n, i, j, sigma):
    return np.exp(-((i - m) ** 2 + (j - n) ** 2) / (2 * sigma ** 2))


def task_two(img, s, k):
    kernel, mask, img = sharpen_image_two(img, s, k)
    mask = mask + 128

    mat = np.zeros((101, 101))
    mat[50, 50] = 255
    kernel = cv2.filter2D(mat, ddepth=-1, kernel=kernel)
    kernel = cv2.normalize(kernel, 0, 0, 255, cv2.NORM_MINMAX)

    cv2.imwrite("res05.jpg", kernel)
    cv2.imwrite("res06.jpg", mask)
    cv2.imwrite("res07.jpg", img)


def sharpen_image_two(img, s, k):
    img = img.astype('float64')
    kernel = get_laplacian(s, 9)
    mask_1 = cv2.filter2D(img, ddepth=-1, kernel=kernel)
    img = img - k * mask_1
    img[img < 0] = 0
    img[img > 255] = 255
    return kernel, mask_1, img


def get_laplacian(sigma, n):
    kernel = np.array([[laplacian(sigma, (i - (n - 1) / 2), (j - (n - 1) / 2)) for j in range(n)] for i in range(n)])
    return kernel


def laplacian(sigma, x, y):
    laplace = (((x ** 2 + y ** 2) - (2 * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))) / (
            np.pi * sigma ** 4)
    return laplace


def get_kernel(rows, cols, s, kernel_type):
    m, n = np.floor(rows / 2).astype(int), np.floor(cols / 2).astype(int)
    kernel = np.array([[gaussian(m, n, i, j, s) for j in range(cols)] for i in range(rows)])
    if kernel_type == "High_pass":
        return np.ones(kernel.shape) - kernel
    else:
        return kernel


def task_three(img, sigma, alpha):
    img_f, high_pass, mask, res_image = sharpen_image_three(img, sigma, alpha)
    img_f = np.log(np.abs(img_f))
    high_pass = np.abs(high_pass)
    mask = np.abs(mask)
    img_f = cv2.normalize(img_f, 0, 0, 255, cv2.NORM_MINMAX)
    high_pass = cv2.normalize(high_pass, 0, 0, 255, cv2.NORM_MINMAX)
    mask = cv2.normalize(mask, 0, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite("res08.jpg", img_f)
    cv2.imwrite("res09.jpg", high_pass)
    cv2.imwrite("res10.jpg", mask)
    cv2.imwrite("res11.jpg", res_image)


def sharpen_image_three(img, sigma, alpha):
    img = img.astype('float64')
    b, g, r = cv2.split(img)
    h, w, _ = img.shape
    kernel = get_kernel(h, w, sigma, "High_pass")
    f_r, mask_r, res_r = apply_fourier(r, kernel, alpha)
    f_g, mask_g, res_g = apply_fourier(g, kernel, alpha)
    f_b, mask_b, res_b = apply_fourier(b, kernel, alpha)
    res_f = np.dstack((f_b, f_g, f_r))
    res_mask = np.dstack((mask_b, mask_g, mask_r))
    res_image = np.dstack((res_b, res_g, res_r))
    img[img < 0] = 0
    img[img > 255] = 255
    return res_f, kernel, res_mask, res_image


def apply_fourier(channel, kernel, alpha):
    img_f = np.fft.fft2(channel)
    img_shifted = np.fft.fftshift(img_f)
    mask = 1 + alpha * kernel
    res_image = mask * img_shifted
    res_image = np.fft.ifftshift(res_image)
    res_image = np.fft.ifft2(res_image)
    res_image = np.real(res_image)
    return img_shifted, mask, res_image


def task_four(img, alpha):
    img = img.astype('float64')
    b, g, r = cv2.split(img)
    h, w = r.shape
    h2, w2 = h // 2, w // 2
    indexes = np.array([[np.abs(i - h2) ** 2 + np.abs(j - w2) ** 2 for j in range(w)] for i in range(h)])
    r_mask, r_f, r = apply(r, indexes, alpha)
    g_mask, g_f, g = apply(g, indexes, alpha)
    b_mask, b_f, b = apply(b, indexes, alpha)
    res_mask = np.dstack((b_mask, g_mask, r_mask))
    res_mask = np.abs(res_mask)
    res_f = np.dstack((b_f, g_f, r_f))
    res_f = np.abs(res_f)
    res_f = res_f / np.max(res_f) * 255
    res_f = 128 + res_f
    res_image = np.dstack((b, g, r))
    cv2.imwrite("res12.jpg", res_mask)
    cv2.imwrite("res13.jpg", res_f)
    cv2.imwrite("res14.jpg", res_image)


def apply(channel, indexes, alpha):
    img_f = np.fft.fft2(channel)
    img_shifted = np.fft.fftshift(img_f)
    mask = (4 * pi ** 2) * indexes * img_shifted
    img_back = np.fft.ifftshift(mask)
    img_back = np.fft.ifft2(img_back)
    res_image = channel + alpha * img_back
    res_image = np.abs(res_image)
    res_image[res_image < 0] = 0
    res_image[res_image > 255] = 255
    return mask, img_back, res_image


image = cv2.imread("flowers.blur.png")
task_one(image, 1, 2, 2)
task_two(image, 1, 2)
task_three(image, 300, 10)
task_four(image, 0.000003)
