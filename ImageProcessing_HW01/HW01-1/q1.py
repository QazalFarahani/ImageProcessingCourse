import numpy as np
import cv2
from matplotlib import pyplot as plt


def enhance(file_name):
    image = cv2.imread(file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = enhance_intensity(image, 2)

    r, g, b = cv2.split(image)
    r = enhance_contrast(r)
    g = enhance_contrast(g)
    b = enhance_contrast(b)
    return cv2.merge((r, g, b))


def enhance_intensity(image, alpha):
    return image * alpha


def enhance_contrast(image):
    hist, bins = np.histogram(image, 256)
    cumulative_sum = np.cumsum(hist)
    normalized = cumulative_sum * 255 / np.max(cumulative_sum)
    normalized = normalized.astype('uint8')
    shape = image.shape
    image = image.ravel()
    result = normalized[image]
    return np.reshape(result, shape)


result = enhance("Enhance1.JPG")
result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
cv2.imwrite("res01.jpg", result)
