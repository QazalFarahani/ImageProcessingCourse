import cv2
import numpy as np
import time
import math
import ffmpeg
import tqdm
from matplotlib import pyplot as plt
from skimage import filters
import ffmpy


def save_image(img, points, path):
    points = np.array(points, np.int32)
    points = np.roll(points, 1, axis=1)
    points = points.reshape((-1, 1, 2))
    copy = img.copy()
    cv2.polylines(copy, [points], True, (200, 200, 200), thickness=3)
    cv2.imwrite(path, copy)


def get_initial_points(center, radius, num_of_points):
    points = np.zeros((num_of_points, 2), dtype=np.int64)
    for i in range(num_of_points):
        theta = float(i) * (2 * np.pi) / num_of_points
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        points[i] = [y, x]
    return points


def get_gradient(img):
    gradient = cv2.Canny(img, 150, 250).astype(float)
    gradient = filters.gaussian(gradient, 7)
    gradient /= gradient.max()
    return gradient


def sigmoid(x):
    return 0.5 - (1 / (1 + math.exp(-10 * x)) - 0.5)


def get_first_derivative_mean(points):
    prev_indexes = np.roll(points, 1, axis=0)
    first_derivative = points - prev_indexes
    first_derivative = np.sqrt(first_derivative[:, 0] ** 2 + first_derivative[:, 1] ** 2)
    return np.mean(first_derivative, axis=0)


def find_best_neighbour(points, point, delta_list, current, viterbi_mat, alpha, neighbour_size, first_derivative):
    min_value = np.inf
    min_index = 0
    for last_neighbour in range(neighbour_size):
        previous = points[point - 1] + delta_list[last_neighbour]
        e = calculate_alpha_term(viterbi_mat, point, last_neighbour, alpha, current, previous, first_derivative)
        if e < min_value:
            min_value = e
            min_index = last_neighbour
    return min_value, min_index


def calculate_alpha_term(viterbi_mat, point, last_neighbour, alpha, current, previous, first_derivative):
    e = viterbi_mat[last_neighbour, point - 1, 0]
    e += alpha * (np.linalg.norm(current - previous) ** 2 - first_derivative) ** 2
    return e


def update_points(points, number_of_points, viterbi_mat, delta_list):
    min_index = viterbi_mat[:, number_of_points - 1, 0].argmin()
    min_index = int(min_index)
    points[0] = points[number_of_points - 1] + delta_list[min_index]
    for point in range(number_of_points - 1):
        current = number_of_points - point - 1
        min_index = viterbi_mat[int(min_index), current, 1]
        points[current] = points[current - 1] + delta_list[int(min_index)]
    return points


def calculate_beta_term(points, beta, current, img_gr):
    second_derivative = (np.linalg.norm(current - np.mean(points, axis=0)) ** 2) ** 2
    return beta * second_derivative * sigmoid(img_gr[tuple(current.astype(int))])


def active_contour(number_of_iterations, number_of_points, points, neighbour_len, img_gr, delta_list, img_o, alpha,
                   beta, gamma):
    neighbour_size = neighbour_len ** 2
    for iteration in tqdm.tqdm(range(number_of_iterations)):
        viterbi_mat = np.zeros((neighbour_size, number_of_points, 2))
        first_derivative = get_first_derivative_mean(points)

        for point in range(number_of_points):
            for neighbour in range(neighbour_size):
                current = points[point] + delta_list[neighbour]
                alpha_value, min_index = find_best_neighbour(points, point, delta_list, current, viterbi_mat, alpha,
                                                             neighbour_size, first_derivative)
                beta_value = calculate_beta_term(points, beta, current, img_gr)
                internal_e = alpha_value + beta_value
                external_e = -gamma * img_gr[tuple(current.astype(int))]
                min_value = internal_e + external_e

                viterbi_mat[neighbour][point][0] = min_value
                viterbi_mat[neighbour][point][1] = min_index

        points = update_points(points, number_of_points, viterbi_mat, delta_list)
        save_image(img_o, points, path="./frames/iteration{i}.jpg".format(i=iteration))
    return points


def create_neighbour_mat(neighbour_len):
    half_len = int(neighbour_len / 2)
    delta_list = []
    for i in range(-half_len, half_len + 1):
        for j in range(-half_len, half_len + 1):
            delta_list.append((i, j))
    return delta_list


def save_video():
    img = []
    for i in range(0, 100):
        img.append(cv2.imread("./frames/iteration" + str(i) + ".jpg"))
    h, w, _ = img[1].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('contour.mp4', fourcc, 5, (w, h))
    for j in range(0, 100):
        video.write(img[j])
    video.release()


def process(img, number_of_points, number_of_iterations):
    center = input_points[0, :]
    radius = np.linalg.norm(center - input_points[1, :])
    initial_points = get_initial_points(center, radius, number_of_points)
    gradient_img = get_gradient(img)
    alpha = 0.00001
    beta = .0000005
    gamma = 10
    delta_list = create_neighbour_mat(5)
    points = active_contour(number_of_iterations=number_of_iterations, number_of_points=number_of_points,
                            points=initial_points.copy(), neighbour_len=5, img_gr=gradient_img, delta_list=delta_list,
                            img_o=img, alpha=alpha, beta=beta, gamma=gamma)
    return points


image = cv2.imread("tasbih.jpg")
plt.imshow(image)
input_points = []

while True:
    while len(input_points) < 2:
        input_points = np.asarray(plt.ginput(-1, timeout=-1))
        if len(input_points) < 2:
            time.sleep(0.5)
    if plt.waitforbuttonpress():
        break
plt.close()

contour = process(image, 100, 100)
save_image(image, contour, "res11.jpg")
save_video()
