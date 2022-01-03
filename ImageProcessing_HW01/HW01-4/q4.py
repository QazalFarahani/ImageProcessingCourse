import numpy as np
import cv2
from matplotlib import pyplot as plt


def read(file_name, image_type):
    img = cv2.imread(file_name)
    if image_type == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif image_type == "HSV":
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def process(img):
    low = 135
    high = 165
    h, s, v = cv2.split(image)
    mask = cv2.inRange(h, low, high)

    flower = cv2.bitwise_and(img, img, mask=mask)
    flower = change_color(flower, mask)
    flower = cv2.cvtColor(flower, cv2.COLOR_HSV2RGB)

    colored_image = change_color(img, mask)
    colored_image = cv2.cvtColor(colored_image, cv2.COLOR_HSV2RGB)
    colored_image = blur(colored_image)

    colored_image_r, colored_image_g, colored_image_b = cv2.split(colored_image)
    flower_r, flower_g, flower_b = cv2.split(flower)
    r = np.where(mask == 0, colored_image_r, flower_r)
    g = np.where(mask == 0, colored_image_g, flower_g)
    b = np.where(mask == 0, colored_image_b, flower_b)
    colored_image = cv2.merge((r, g, b))

    return colored_image


def blur(img):
    filter = np.ones((10, 10), np.float64) / 100
    return cv2.filter2D(src=img, ddepth=-1, kernel=filter)


def change_color(img, mask):
    h, s, v = cv2.split(img)
    h[mask > 0] = 27
    img = cv2.merge((h, s, v))
    return img


image = read("Flowers.jpg", "HSV")
image = process(image)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imwrite("res06.jpg", image)
