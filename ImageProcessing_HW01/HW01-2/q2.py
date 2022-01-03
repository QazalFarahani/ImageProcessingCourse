import numpy as np
import cv2
from matplotlib import pyplot as plt


def read(file_name, image_type):
    img = cv2.imread(file_name)
    if image_type == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif image_type == "HSV":
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def enhance_intensity(img):
    h, s, v = cv2.split(img)
    alpha = 0.07
    v = np.apply_along_axis(lambda x: 255 * np.log(1 + alpha * x) / np.log(1 + 255 * alpha), 0, v)
    v = v.astype('uint8')
    img = cv2.merge((h, s, v))
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


image = read("Enhance2.jpg", "HSV")
image = enhance_intensity(image)
plt.imshow(image)
plt.show()
result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imwrite("res02.jpg", result)
