import numpy as np
import cv2
from matplotlib import pyplot as plt

from timeit import default_timer as timer


def read(file_name, image_type):
    img = cv2.imread(file_name)
    if image_type == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif image_type == "HSV":
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def library_blur(img):
    filter = np.ones((3, 3)) / 9
    return cv2.filter2D(src=img, ddepth=-1, kernel=filter)[1:-1, 1:-1, :]
    # img = cv2.blur(img, (3, 3))
    # return img[1:-1, 1:-1, :]


def each_pixel_blur(img):
    r, g, b = cv2.split(img)
    m = b.shape[0]
    n = b.shape[1]
    vp = np.zeros(b.shape)
    sp = np.zeros(b.shape)
    hp = np.zeros(b.shape)
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            vp[i, j] = apply_box_filter(i, j, b)
            sp[i, j] = apply_box_filter(i, j, g)
            hp[i, j] = apply_box_filter(i, j, r)
    img = cv2.merge((hp.astype('uint8'), sp.astype('uint8'), vp.astype('uint8')))
    return img[1:-1, 1:-1, :]


def apply_box_filter(i, j, img):
    value = 0
    for p in range(-1, 2):
        for q in range(-1, 2):
            value += (img[i + p, j + q] / 9)
    return value


def overlap_matrices_blur(img):
    r, g, b = cv2.split(img)
    r = blur(r)
    g = blur(g)
    b = blur(b)
    return cv2.merge((r, g, b))


def blur(img):
    img_1 = img[:-2, :-2]
    img_2 = img[:-2, 1:-1]
    img_3 = img[:-2, 2:]
    img_4 = img[1:-1, :-2]
    img_5 = img[1:-1, 1:-1]
    img_6 = img[1:-1, 2:]
    img_7 = img[2:, :-2]
    img_8 = img[2:, 1:-1]
    img_9 = img[2:, 2:]

    return (img_1 / 9 + img_2 / 9 + img_3 / 9 + img_4 / 9 + img_5 / 9
            + img_6 / 9 + img_7 / 9 + img_8 / 9 + img_9 / 9).astype('uint8')


Pink = read("Pink.jpg", "RGB")

start = timer()
Pink_1 = library_blur(Pink)
end = timer()
print("function1: ", end - start)
Pink_1 = cv2.cvtColor(Pink_1, cv2.COLOR_RGB2BGR)
cv2.imwrite("res07.jpg", Pink_1)

start = timer()
Pink_2 = each_pixel_blur(Pink)
end = timer()
print("function2: ", end - start)
Pink_2 = cv2.cvtColor(Pink_2, cv2.COLOR_RGB2BGR)
cv2.imwrite("res08.jpg", Pink_2)

start = timer()
Pink_3 = overlap_matrices_blur(Pink)
end = timer()
print("function3: ", end - start)
Pink_3 = cv2.cvtColor(Pink_3, cv2.COLOR_RGB2BGR)
cv2.imwrite("res09.jpg", Pink_3)


# function1:  0.025650200000000067
# function2:  336.6476464
# function3:  0.6048652000000061
