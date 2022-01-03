import numpy as np
import cv2
from matplotlib import pyplot as plt


def read(file_name, image_type):
    img = cv2.imread(file_name)
    if image_type == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif image_type == "HSV":
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def process(input_image, target_image):
    input_image_r, input_image_g, input_image_b = cv2.split(input_image)
    target_image_r, target_image_g, target_image_b = cv2.split(target_image)
    input_image_r = match_histograms(input_image_r, target_image_r)
    input_image_g = match_histograms(input_image_g, target_image_g)
    input_image_b = match_histograms(input_image_b, target_image_b)
    return np.dstack((input_image_r, input_image_g, input_image_b))


def match_histograms(input_image, target_image):
    input_hist, input_bins = np.histogram(input_image, 256, [0, 256])
    target_hist, target_bins = np.histogram(target_image, 256, [0, 256])
    input_hist_sum = np.cumsum(input_hist).astype('float64')
    target_hist_sum = np.cumsum(target_hist).astype('float64')
    input_hist_sum /= np.max(input_hist_sum)
    target_hist_sum /= np.max(target_hist_sum)
    result = np.zeros(input_image.shape)
    for i in range(0, 256):
        value = map_value(input_hist_sum[i], target_hist_sum)
        result[input_image == i] = value
    return result


def map_value(param, target_hist_sum):
    for i in range(0, 256):
        if target_hist_sum[i] >= param:
            return i


def show_image_hist(img, file_name):
    hist, bins = np.histogram(img, 256, [0, 256])
    hist = (hist / np.max(hist))
    plt.plot(hist)
    plt.savefig(file_name)


def show_cum_sum(img, file_name):
    hist, bins = np.histogram(img, 256, [0, 256])
    hist = np.cumsum(hist).astype('float64')
    hist = (hist / np.max(hist))
    plt.plot(hist)
    plt.savefig(file_name)


Dark = read("Dark.jpg", "RGB")
Pink = read("Pink.jpg", "RGB")

result = process(Dark, Pink).astype('uint8')

show_image_hist(result, "res10.jpg")
plt.clf()
show_cum_sum(result, "res10_cumsum.jpg")
plt.imshow(result)
plt.savefig("res11.jpg")
plt.show()
