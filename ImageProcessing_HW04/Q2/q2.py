import cv2
import numpy as np
import tqdm
from matplotlib import pyplot as plt


def get_feature_space(img):
    feature_space_num = np.zeros((256, 256, 256))
    feature_space_sum = np.zeros((256, 256, 256, 3))
    for i, point in enumerate(np.ndindex(img.shape[:2])):
        pixel = img[point]
        r, g, b = pixel[0], pixel[1], pixel[2]
        feature_space_num[r, g, b] += 1
        feature_space_sum[r, g, b, :] += pixel
    return feature_space_num, feature_space_sum


def get_cumsum(feature_space_num, feature_space_sum):
    feature_space_num = np.cumsum(feature_space_num, axis=0)
    feature_space_num = np.cumsum(feature_space_num, axis=1)
    feature_space_num = np.cumsum(feature_space_num, axis=2)
    feature_space_sum = np.cumsum(feature_space_sum, axis=0)
    feature_space_sum = np.cumsum(feature_space_sum, axis=1)
    feature_space_sum = np.cumsum(feature_space_sum, axis=2)
    return feature_space_num, feature_space_sum


def mean_shift(point, feature_space_num, feature_space_sum):
    window_size = 17
    half_len = int(window_size / 2)
    r, g, b = point[0], point[1], point[2]
    r_minus, r_plus = np.maximum(0, r - half_len), np.minimum(255, r + half_len)
    g_minus, g_plus = np.maximum(0, g - half_len), np.minimum(255, g + half_len)
    b_minus, b_plus = np.maximum(0, b - half_len), np.minimum(255, b + half_len)
    neighbours_sum = feature_space_sum[r_plus, g_plus, b_plus, :] \
                     - feature_space_sum[r_minus, g_plus, b_plus, :] \
                     - feature_space_sum[r_plus, g_minus, b_plus, :] \
                     - feature_space_sum[r_plus, g_plus, b_minus, :] \
                     + feature_space_sum[r_minus, g_minus, b_plus, :] \
                     + feature_space_sum[r_minus, g_plus, b_minus, :] \
                     + feature_space_sum[r_plus, g_minus, b_minus, :] \
                     - feature_space_sum[r_minus, g_minus, b_minus, :]
    neighbours_num = feature_space_num[r_plus, g_plus, b_plus] \
                     - feature_space_num[r_minus, g_plus, b_plus] \
                     - feature_space_num[r_plus, g_minus, b_plus] \
                     - feature_space_num[r_plus, g_plus, b_minus] \
                     + feature_space_num[r_minus, g_minus, b_plus] \
                     + feature_space_num[r_minus, g_plus, b_minus] \
                     + feature_space_num[r_plus, g_minus, b_minus] \
                     - feature_space_num[r_minus, g_minus, b_minus]
    return neighbours_sum / neighbours_num


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def get_centroid(point, feature_space_num, feature_space_sum, threshold):
    iteration = 0
    current_point = point
    while iteration < 50:
        iteration += 1
        centroid = mean_shift(current_point, feature_space_num, feature_space_sum)
        distance = euclidean_distance(current_point, centroid)
        if distance < threshold:
            break
        current_point = centroid.astype(int)
    return centroid.astype(int)


def process(img):
    img_luv = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)
    flat_image = img_luv.reshape((-1, img.shape[-1]))
    feature_space_num, feature_space_sum = get_feature_space(img_luv)
    points = np.argwhere(feature_space_num != 0)
    num_cumsum, sum_cumsum = get_cumsum(feature_space_num, feature_space_sum)
    for point in tqdm.tqdm(points):
        centroid = get_centroid(point, num_cumsum, sum_cumsum, 0.5)
        ind = np.all(flat_image == point, axis=1)
        flat_image[ind, :] = centroid
    return flat_image.reshape(img_luv.shape)


image = cv2.imread("park.jpg")
image = cv2.blur(image, (20, 20))
image = cv2.resize(image, (image.shape[1] // 10, image.shape[0] // 10))
print(image.shape)
result = process(image)
result = cv2.cvtColor(result, cv2.COLOR_Luv2BGR)
cv2.imwrite("res05.jpg", result)
