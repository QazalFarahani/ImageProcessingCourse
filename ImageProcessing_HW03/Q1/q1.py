import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max
from pylab import *


def hough_transform(h, w, coord, threshold, d_theta, d_r, dist):
    d_theta = np.rad2deg(d_theta)
    theta_range = np.deg2rad(np.arange(0, 180, d_theta))
    max_r = int(np.sqrt(h ** 2 + w ** 2))
    r_range = np.arange(-max_r, max_r, d_r)
    n_r, n_theta = len(r_range), len(theta_range)
    accumulator = get_accumulator(coord, theta_range, max_r, n_r, n_theta, d_r)
    lines = get_lines(accumulator, theta_range, r_range, threshold, dist)
    return lines, accumulator


def get_accumulator(coord, theta_range, max_r, n_r, n_theta, d_r):
    accumulator = np.zeros((n_r, n_theta))
    for i in range(len(coord[0])):
        x = coord[1][i]
        y = coord[0][i]
        for t_idx in range(len(theta_range)):
            t = theta_range[t_idx]
            r = int(round(x * np.cos(t) + y * np.sin(t))) + max_r
            r = int(r / d_r)
            accumulator[r, t_idx] += 1
    return accumulator


def get_lines(accumulator, thetas, rhos, threshold, dist):
    accumulator = np.where(accumulator >= threshold, accumulator, 0)
    maximums = peak_local_max(accumulator, dist)
    rho, theta = [], []
    for i in range(len(maximums)):
        t = thetas[maximums[i][1]]
        r = rhos[maximums[i][0]]
        rho.append(r)
        theta.append(t)
    return [rho, theta]


def draw_lines(img, lines):
    for i in range(len(lines)):
        r, theta = lines[i][0], lines[i][1]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * r, b * r
        x1 = int(x0 + 10000 * (-b))
        x2 = int(x0 - 10000 * (-b))
        y1 = int(y0 + 10000 * a)
        y2 = int(y0 - 10000 * a)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return img


def get_final_lines(lines, threshold):
    bins = []
    points = np.ndarray((len(lines[0]), 2))
    points[:, 0] = lines[0]
    points[:, 1] = lines[1]
    points = points[np.argsort(points[:, 1])]
    i = 0
    while i < len(points[:, 0]):
        bin = []
        bin.append(points[i, :])
        while i + 1 < len(points[:, 0]) and abs(points[i + 1, :][1] - points[i, :][1]) < threshold:
            i += 1
            bin.append(points[i, :])
        i += 1
        bins.append(bin)
    counts = [len(bins[i]) for i in range(len(bins))]
    counts = np.argsort(counts)
    n = len(counts)
    return bins[counts[n - 1]], bins[counts[n - 2]]


def update_lines(lines1, lines2, dist1_x, dist1_y, dist2_x, dist2_y, search_gap, block_size, img, w_threshold,
                 h_threshold):
    updated_lines = []
    for i in range(len(lines1)):
        points_on_line = []
        for j in range(len(lines2)):
            point = get_point(lines1[i], lines2[j])
            if point[0] < img.shape[1] - search_gap and point[1] < img.shape[0] - search_gap and \
                    is_target_point(point, dist1_x, dist1_y, dist2_x, dist2_y, block_size, img, w_threshold,
                                    h_threshold):
                points_on_line.append(point)
        if len(points_on_line) > len(lines2) * 2 / 4:
            updated_lines.append(lines1[i])
    return updated_lines


def get_intersections(lines1, lines2):
    points = []
    for i in range(len(lines1)):
        for j in range(len(lines2)):
            points.append(get_point(lines1[i], lines2[j]))
    return points


def is_target_point(point, dist1_x, dist1_y, dist2_x, dist2_y, block_size, img, w_threshold, h_threshold):
    x, y = point[0], point[1]
    up = get_block_mean(x + dist1_x, y + dist1_y, block_size, img)
    down = get_block_mean(x - dist1_x, y - dist1_y, block_size, img)
    left = get_block_mean(x + dist2_x, y + dist2_y, block_size, img)
    right = get_block_mean(x - dist2_x, y - dist2_y, block_size, img)
    h_mean = left / 2 + right / 2
    w_mean = up / 2 + down / 2
    if (h_mean > w_threshold and w_mean < h_threshold) or (h_mean < h_threshold and w_mean > w_threshold):
        return True
    elif up > w_threshold and left < h_threshold and right < h_threshold and down < h_threshold:
        return True
    elif left > w_threshold and up < h_threshold and right < h_threshold and down < h_threshold:
        return True
    elif right > w_threshold and left < h_threshold and up < h_threshold and down < h_threshold:
        return True
    elif down > w_threshold and left < h_threshold and right < h_threshold and up < h_threshold:
        return w_threshold
    return False


def get_point(line1, line2):
    rho1, theta1 = line1[0], line1[1]
    rho2, theta2 = line2[0], line2[1]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]


def get_block_mean(y, x, block_size, img):
    return np.mean(img[x: x + block_size, y: y + block_size])
    # return img[x, y]


def draw_points(img, points):
    for point in points:
        img = cv2.circle(img, (point[0], point[1]), radius=5, color=(0, 0, 255), thickness=-1)
    return img


def remove_extra_lines(lines, threshold):
    arr_lines = np.ndarray((len(lines), 2))
    for i in range(len(lines)):
        arr_lines[i] = lines[i]
    arr_lines = arr_lines[np.argsort(arr_lines[:, 0])]
    updated_lines = [arr_lines[0, :]]
    for i in range(1, len(lines)):
        if abs(arr_lines[i, 0] - arr_lines[i - 1, 0]) > threshold:
            updated_lines.append(arr_lines[i, :])
    return updated_lines


def process(img, canny_min, canny_max, line_threshold, n_theta, n_rho, peak_threshold, l_detect_threshold, search_gap,
            block_size, w_threshold, h_threshold, rho_gap1, rho_gap2):
    image_edges = cv2.Canny(img, canny_min, canny_max)
    coord = np.where(image_edges == 255)
    h, w = image_edges.shape
    lines, accumulator = hough_transform(h, w, coord, line_threshold, np.pi / n_theta, n_rho, peak_threshold)

    lines1, lines2 = get_final_lines(lines, l_detect_threshold)
    lined_image1 = img.copy()
    lined_image1 = draw_lines(lined_image1, lines1)
    lined_image1 = draw_lines(lined_image1, lines2)
    theta1, theta2 = lines1[5][1], lines2[5][1]
    d = 20
    dist1, dist2 = [d, d * np.tan(theta1 + np.pi / 4)], [d, d * np.tan(theta2 + np.pi / 4)]
    dist1 = (dist1 / np.max(np.abs(dist1)) * search_gap).astype(int)
    dist2 = (dist2 / np.max(np.abs(dist2)) * search_gap).astype(int)
    dist1_x, dist1_y = dist1[0], dist1[1]
    dist2_x, dist2_y = dist2[0], dist2[1]
    lines1 = update_lines(lines1, lines2, dist1_x, dist1_y, dist2_x, dist2_y, search_gap, block_size, img, w_threshold,
                          h_threshold)
    lines2 = update_lines(lines2, lines1, dist1_x, dist1_y, dist2_x, dist2_y, search_gap, block_size, img, w_threshold,
                          h_threshold)
    lines1 = remove_extra_lines(lines1, rho_gap1)
    lines2 = remove_extra_lines(lines2, rho_gap2)
    final_points = get_intersections(lines1, lines2)
    lined_image2 = draw_lines(img, lines1)
    lined_image2 = draw_lines(lined_image2, lines2)
    image_corners = lined_image2.copy()
    image_corners = draw_points(image_corners, final_points)
    return image_edges, accumulator, lined_image1, lined_image2, image_corners


# image = cv2.imread("im01.jpg", cv2.IMREAD_GRAYSCALE)
# edges, acc, lined1, lined2, corners = process(image, 200, 600, 90, 100, 1, 18, 0.2, 10, 4, 110, 110, 40, 25)
# acc = (acc / np.max(acc) * 255).astype('uint8')
# cv2.imwrite("res01.jpg", edges)
# cv2.imwrite("res03-hough-space.jpg", acc)
# cv2.imwrite("res05-lines.jpg", lined1)
# cv2.imwrite("res07-chess.jpg", lined2)
# cv2.imwrite("res09-corners.jpg", corners)

image = cv2.imread("im02.jpg", cv2.IMREAD_GRAYSCALE)
edges, acc, lined1, lined2, corners = process(image, 100, 400, 77, 70, 1, 15, 0.2, 10, 4, 130, 130, 1, 1)
acc = (acc / np.max(acc) * 255).astype('uint8')
cv2.imwrite("res02.jpg", edges)
cv2.imwrite("res04-hough-space.jpg", acc)
cv2.imwrite("res06-lines.jpg", lined1)
cv2.imwrite("res08-chess.jpg", lined2)
cv2.imwrite("res10-corners.jpg", corners)
