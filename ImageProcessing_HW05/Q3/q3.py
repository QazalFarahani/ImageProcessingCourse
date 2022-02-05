import cv2
from matplotlib import pyplot as plt
from skimage import color


def laplacian_pyramid(img, iterations):
    pyramid = [img.copy()]
    for i in range(iterations):
        blurred = cv2.GaussianBlur(pyramid[i], (17, 17), 0)
        pyramid[i] -= blurred
        pyramid.append(blurred)
    return pyramid


def blend(src, tar, msk, bandwidth):
    msk = cv2.GaussianBlur(msk, (bandwidth, bandwidth), 0)
    msk = cv2.merge((msk, msk, msk))
    return src * msk + tar * (1 - msk)


def process(src, tar, msk, iterations, ws1, ws2):
    src_laplacian = laplacian_pyramid(src, iterations)
    tar_laplacian = laplacian_pyramid(tar, iterations)

    src_laplacian[iterations] = blend(src_laplacian[iterations], tar_laplacian[iterations], msk, ws1)
    for i in range(iterations - 1, -1, -1):
        src_laplacian[i] = blend(src_laplacian[i], tar_laplacian[i], msk, ws2)
        src_laplacian[i] += src_laplacian[i + 1]
    return src_laplacian[0]


source = cv2.imread('res08.jpg').astype(float)
target = cv2.imread('res09.jpg').astype(float)
mask = color.rgb2gray(cv2.imread('mask.jpg'))
result = process(source, target, mask, 8, 19, 9)
cv2.imwrite('res10.jpg', result)

# source = cv2.imread('kiwi.jpg').astype(float)
# target = cv2.imread('orange.jpg').astype(float)
# mask = color.rgb2gray(cv2.imread('mask2.jpg'))
# result = process(source, target, mask, 8, 419, 53)
# cv2.imwrite('res4.jpg', result)
