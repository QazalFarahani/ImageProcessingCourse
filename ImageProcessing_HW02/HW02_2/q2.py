import cv2
import numpy as np
from matplotlib import pyplot as plt


def read(file_name, image_type):
    img = cv2.imread(file_name)
    if image_type == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif image_type == "HSV":
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def match_template(img, temp):
    h_i, w_i = img.shape
    h_t, w_t = temp.shape
    h, w = h_i - h_t, w_i - w_t
    score = np.zeros((h, w))
    for i in range(0, h):
        for j in range(0, w):
            diff = ncc(img, temp, i, h_t, j, w_t)
            score[i, j] = diff.sum()
    return score


def ncc(img, temp, i, h_t, j, w_t):
    a = img[i:i + h_t, j:j + w_t]
    b = temp
    a = a - np.mean(a)
    b = b - np.mean(b)
    return (a * b) / (np.linalg.norm(a) * np.linalg.norm(b))


def process(img_colored, img_grayscale, template):
    h, w = template.shape
    s = 8
    img_grayscale = cv2.resize(img_grayscale, (img_grayscale.shape[1] // s, img_grayscale.shape[0] // s))
    template = cv2.resize(template, (template.shape[1] // s, template.shape[0] // s))
    score = match_template(img_grayscale, template)
    threshold = 0.36
    points = np.argwhere(score > threshold)
    points = points[np.argsort(points[:, 1])]
    unique_indexes = get_one_match_for_each(points)
    draw_rectangles(unique_indexes, points, s, img_colored, w, h)
    return img_colored


def get_one_match_for_each(points):
    unique_values = np.unique(points[:, 1])
    unique_indexes = []
    for value in unique_values:
        indices = np.where(points[:, 1] == value)
        index = indices[len(indices) // 2]
        unique_indexes.append(index)
    return unique_indexes


def draw_rectangles(unique_indexes, points, s, img_colored, w, h):
    for i in unique_indexes:
        p = points[i][0]
        pnt = (p[1] * s, p[0] * s)
        cv2.rectangle(img_colored, pnt, (pnt[0] + w, pnt[1] + h), (0, 0, 255), 2)


image_colored = cv2.imread("Greek-ship.jpg")
image_grayscale = cv2.imread("Greek-ship.jpg", 0)
patch = cv2.imread("patch.png", 0)
result = process(image_colored, image_grayscale, patch)
cv2.imwrite("res15.jpg", result)
