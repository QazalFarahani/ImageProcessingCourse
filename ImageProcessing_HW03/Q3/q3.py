import random
from heapq import heapify, heappush, heappop
import cv2
import numpy as np
from matplotlib import pyplot as plt


def process(img, search_area, block_size, overlap, h_blocks, w_blocks, top_left_point, m, n, n_v, n_h, k, p):
    top_left_x, top_left_y = top_left_point[0], top_left_point[1]
    im = img.copy()
    im[top_left_y:top_left_y + n, top_left_x:top_left_x + m] = 0
    top_left_x, top_left_y = top_left_point[0] - overlap, top_left_point[1] - overlap
    mask = np.zeros((block_size, block_size), dtype='float32')
    mask[:, :overlap] = 1
    mask[-overlap:, :] = 1
    res = im.copy()

    y = top_left_y + n + overlap * 2
    for i in range(h_blocks):
        x = top_left_x
        for j in range(w_blocks):
            res = synthesize(search_area, block_size, overlap, res, mask, x, y)
            x += (block_size - overlap)
        y -= (block_size - overlap)

    x = (w_blocks - 1) * (block_size - overlap) + top_left_x + block_size
    y = top_left_y + n + overlap * 2
    plt.scatter([x], [y])
    res = remove_end_line(res, image2, x, y, k, p, n_v)
    x = (w_blocks - 1) * (block_size - overlap) + top_left_x + block_size
    y = top_left_y + n + overlap * 2 - ((h_blocks - 1) * (block_size - overlap) + block_size)
    res = remove_end_line(res.transpose(1, 0, 2), image2.transpose(1, 0, 2), y, x, k, p, n_h)
    return res.transpose(1, 0, 2)


def remove_end_line(img, search_image, x, y, mid, overlap, n):
    block_size = int(mid + 2 * overlap)
    x = int(x - mid / 2 - overlap)

    mask = np.zeros((block_size, block_size), dtype='float32')
    mask[:, :overlap] = 1
    mask[:, -overlap:] = 1
    mask[-overlap:, :] = 1

    for i in range(n):
        print(x, y)
        plt.scatter([x], [y])
        patch = get_match(search_image, block_size, img, mask, x, y)
        patch = three_side_min_cut(patch, block_size, overlap, img, x, y)
        img[y - block_size:y, x:x + block_size] = patch
        y -= (block_size - overlap)
    return img


def three_side_min_cut(patch, block_size, overlap, img, x, y):
    patch = patch.copy()
    cut = np.zeros(patch.shape, dtype=bool)
    cut = get_min_cut_horizontal(patch, block_size, overlap, x, y, img, cut)
    cut = get_min_cut_vertical(patch, block_size, overlap, x, y, img, cut)
    cut = get_min_cut_vertical_right(patch, block_size, overlap, x, y, img, cut)
    np.copyto(patch, img[y - block_size:y, x:x + block_size], where=cut)
    return patch


def synthesize(search_image, block_size, overlap, res, mask, x, y):
    patch = get_match(search_image, block_size, res, mask, x, y)
    patch = min_cut(patch, block_size, overlap, res, x, y)
    res[y - block_size:y, x:x + block_size] = patch
    return res


def get_random_from_k_best(matches, k):
    flattened = matches.flatten()
    k = len(flattened) - k
    flattened = np.partition(flattened, k)
    max_values = flattened[k:]
    points = []
    for value in max_values[0:2]:
        maximums_ind = np.where(matches == value)
        max_points = list(zip(maximums_ind[0], maximums_ind[1]))
        points.extend(max_points)
    index = random.randint(0, len(points) - 1)
    return points[index]


def get_match(search_image, block_size, res, mask, x, y):
    template = res[y - block_size:y, x:x + block_size]
    matches = cv2.matchTemplate(search_image, template, cv2.TM_CCORR_NORMED, mask=mask)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matches)
    # j, i = max_loc
    i, j = get_random_from_k_best(matches, 5)
    return search_image[i:i + block_size, j:j + block_size]


def min_cut(patch, block_size, overlap, res, x, y):
    patch = patch.copy()
    cut = np.zeros(patch.shape, dtype=bool)
    cut = get_min_cut_vertical(patch, block_size, overlap, x, y, res, cut)
    cut = get_min_cut_horizontal(patch, block_size, overlap, x, y, res, cut)
    np.copyto(patch, res[y - block_size:y, x:x + block_size], where=cut)
    return patch


def get_min_cut_vertical_right(patch, block_size, overlap, x, y, res, cut):
    diff = np.sum((patch[:, -overlap:] - res[y - block_size:y, x + block_size - overlap:x + block_size]) ** 2, axis=2)
    path = get_min_cut_path(diff, diff.shape)
    index = 0
    for i in path:
        cut[index, block_size - overlap + i:] = True
        index += 1
    return cut


def get_min_cut_horizontal(patch, block_size, overlap, x, y, res, cut):
    diff = np.sum((patch[-overlap:, :] - res[y - overlap:y, x:x + block_size]) ** 2, axis=2)
    path = get_min_cut_path(np.transpose(diff), np.transpose(diff).shape)
    index = 0
    for i in path:
        cut[-overlap + i:, index] = True
        index += 1
    return cut


def get_min_cut_vertical(patch, block_size, overlap, x, y, res, cut):
    diff = np.sum((patch[:, :overlap] - res[y - block_size:y, x:x + overlap]) ** 2, axis=2)
    path = get_min_cut_path(diff, diff.shape)
    index = 0
    for i in path:
        cut[index, :i] = True
        index += 1
    return cut


def make_heap(diff):
    paths = []
    for i in range(len(diff[0])):
        paths.append([diff[0][i], [i]])
    heapify(paths)
    return paths


def next_index_update(next_index, length, seen, diff, paths, path, error):
    if (length, next_index) not in seen:
        heappush(paths, [error + diff[length, next_index], path + [next_index]])
        seen.add((length, next_index))
    return paths, seen


def get_min_cut_path(diff, shape):
    h, w = shape
    paths = make_heap(diff)
    seen = set()
    while True:
        error, path = heappop(paths)
        length = len(path)
        index = path[length - 1]
        if length >= h:
            best_path = path
            break
        for next_index in [index - 1, index, index + 1]:
            if 0 <= next_index < w:
                paths, seen = next_index_update(next_index, length, seen, diff, paths, path, error)
    return best_path


# image = cv2.imread("im04.jpg")
# image2 = image[1180:, :]
# result = process(image, image2, 40, 10, 16, 10, [740, 700], 200, 460, 40, 20, 4, 8)
# cv2.imwrite("res16.jpg", result)

image = cv2.imread("im03.jpg")
image2 = image[200:, :800]
result = process(image, image2, 70, 30, 5, 4, [830, 750], 140, 175, 7, 9, 9, 15)
result = process(result, image2, 50, 20, 4, 4, [1130, 610], 100, 100, 5, 5, 8, 15)
result = process(result, image2, 50, 20, 4, 5, [420, 70], 100, 100, 6, 6, 8, 15)
result = process(result, image2, 50, 20, 4, 4, [320, 70], 100, 100, 6, 6, 8, 15)
cv2.imwrite("res15.jpg", result)

