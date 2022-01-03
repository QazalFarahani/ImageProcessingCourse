import random

import cv2
import numpy as np
from heapq import heapify, heappush, heappop


def process(img, block_size, overlap, num_block):
    img = img.astype('float32')

    h = ((num_block - 1) * (block_size - overlap)) + block_size
    w = ((num_block - 1) * (block_size - overlap)) + block_size

    mask1 = np.zeros((block_size, block_size), dtype='float32')
    mask2 = np.zeros((block_size, block_size), dtype='float32')
    mask3 = np.zeros((block_size, block_size), dtype='float32')

    mask1[:, :overlap] = 1
    mask2[:overlap, :] = 1
    mask3[:, :overlap] = 1
    mask3[:overlap, :] = 1

    res = np.zeros((h, w, 3), dtype='float32')
    res[:block_size, :block_size] = img[:block_size, :block_size]

    for i in range(num_block):
        for j in range(num_block):
            if i == 0:
                mask = mask1
            elif j == 0:
                mask = mask2
            else:
                mask = mask3

            res = synthesize(img, block_size, overlap, res, mask, i, j)

    return res


def synthesize(texture, block_size, overlap, res, mask, i, j):
    y = i * (block_size - overlap)
    x = j * (block_size - overlap)
    patch = get_match(texture, block_size, res, mask, x, y)
    patch = min_cut(patch, block_size, overlap, res, x, y)
    res[y:y + block_size, x:x + block_size] = patch
    return res


def get_match(texture, block_size, res, mask, x, y):
    template = res[y:y + block_size, x:x + block_size]
    matches = cv2.matchTemplate(texture, template, cv2.TM_CCORR_NORMED, mask=mask)
    i, j = get_random_from_k_best(matches, 3)
    print(i, j)
    return texture[i:i + block_size, j:j + block_size]


def get_random_from_k_best(matches, k):
    flattened = matches.flatten()
    k = len(flattened) - k
    flattened = np.partition(flattened, k)
    max_values = flattened[k:]
    points = []
    for value in max_values:
        maximums_ind = np.where(matches == value)
        max_points = list(zip(maximums_ind[0], maximums_ind[1]))
        points.extend(max_points)
    index = random.randint(0, len(points) - 1)
    return points[index]


def min_cut(patch, block_size, overlap, res, x, y):
    patch = patch.copy()
    cut = np.zeros(patch.shape, dtype=bool)
    if x != 0:
        cut = get_min_cut_vertical(patch, block_size, overlap, x, y, res, cut)
    if y != 0:
        cut = get_min_cut_horizontal(patch, block_size, overlap, x, y, res, cut)
    np.copyto(patch, res[y:y + block_size, x:x + block_size], where=cut)
    return patch


def get_min_cut_horizontal(patch, block_size, overlap, x, y, res, cut):
    diff = np.sum((patch[:overlap, :] - res[y:y + overlap, x:x + block_size]) ** 2, axis=2)
    path = get_min_cut_path(np.transpose(diff), np.transpose(diff).shape)
    index = 0
    for i in path:
        cut[:i, index] = True
        index += 1
    return cut


def get_min_cut_vertical(patch, block_size, overlap, x, y, res, cut):
    diff = np.sum((patch[:, :overlap] - res[y:y + block_size, x:x + overlap]) ** 2, axis=2)
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
        updated_err = error + diff[length, next_index]
        updated_path = path + [next_index]
        heappush(paths, [updated_err, updated_path])
        seen.add((length, next_index))
    return paths, seen


def get_min_cut_path(diff, shape):
    h, w = shape
    paths = make_heap(diff)
    seen = set()
    while True:
        top = heappop(paths)
        error, path = top[0], top[1]
        length = len(path)
        index = path[length - 1]
        if length >= h:
            best_path = path
            break
        for next_index in [index - 1, index, index + 1]:
            if 0 <= next_index < w:
                paths, seen = next_index_update(next_index, length, seen, diff, paths, path, error)
    return best_path


def main(image_path, res_path, block_size, overlap, num_blocks):
    image = cv2.imread(image_path)
    result = process(image, block_size, overlap, num_blocks)
    res_texture = np.ones((2520, 4000, 3)) * 255
    h, w, _ = image.shape
    res_texture[20: 20 + h, 20: 20 + w, :] = image
    res_texture[:, -2520:, :] = result
    cv2.imwrite(res_path, res_texture)


main("texture06.jpg", "res11.jpg", 120, 20, 25)
# main("texture11.jpeg", "res12.jpg", 120, 20, 25)
# main("texture_1.jpg", "res13.jpg", 120, 20, 25)
# main("texture_2.jpg", "res14.jpg", 120, 20, 25)
