import cv2
import imageio
import numpy as np
import scipy.spatial as scp
import matplotlib.pyplot as plt
import tqdm


def process(src, tar, src_points, tar_points, number_of_iterations):
    triangles = scp.Delaunay(src_points)
    triangles = triangles.simplices

    for iter in tqdm.tqdm(range(0, number_of_iterations + 1)):
        fraction = iter / number_of_iterations
        current_points = fraction * target_points + (1 - fraction) * src_points
        h, w, _ = src.shape
        frame = np.zeros((w, h, 3), dtype='float32')

        for i in range(0, triangles.shape[0]):
            src_points_src = get_points(src_points, triangles[i])
            tar_point_src = get_points(tar_points, triangles[i])
            frame_points = get_points(current_points, triangles[i])
            mask = get_triangle_mask(src, frame_points, frame)
            frame_points[:, [0, 1]] = frame_points[:, [1, 0]]
            src_warped = warp_image(src_points_src, frame_points, src, mask)
            tar_warped = warp_image(tar_point_src, frame_points, tar, mask)
            frame += mask * (fraction * tar_warped + (1 - fraction) * src_warped)
        save_frames(iter, frame, number_of_iterations)


def get_points(defined_points, indexes):
    p1, p2, p3 = defined_points[indexes[0]], defined_points[indexes[1]], defined_points[indexes[2]]
    return np.array([p1, p2, p3], dtype='float32')


def warp_image(point, frame_points, image, mask):
    transform_mat = cv2.getAffineTransform(point, frame_points)
    h, w, _ = image.shape
    warped_image = cv2.warpAffine(image, transform_mat, (h, w)) * mask
    return warped_image


def get_triangle_mask(image, frame_points, frame):
    h, w, _ = image.shape
    mask = np.zeros((w, h, 3), dtype='uint8')
    temp = np.zeros(frame_points.shape).astype('float32')
    temp[:, 0], temp[:, 1] = frame_points[:, 1], frame_points[:, 0]
    mask = cv2.fillPoly(mask, np.int32([temp]), color=(1, 1, 1))
    mask *= np.equal(frame, 0)
    return mask


def save_frames(iter, frame, number_of_iterations):
    cv2.imwrite("{i}.jpg".format(i=iter), np.transpose(frame, (1, 0, 2)))
    cv2.imwrite("{i}.jpg".format(i=number_of_iterations * 2 + 1 - iter), np.transpose(frame, (1, 0, 2)))


def create_gif():
    images = []
    for i in range(0, 90):
        file_path = "{i}.jpg".format(i=i)
        images.append(imageio.imread(file_path))
    imageio.mimsave('morph.gif', images, fps=30)


source = cv2.imread('res01.jpg')
target = cv2.imread('res02.jpg')
source_points = np.loadtxt('src.txt', delimiter=' ')
target_points = np.loadtxt('dst.txt', delimiter=' ')


process(source, target, source_points, target_points, number_of_iterations=44)

image = cv2.imread("{i}.jpg".format(i=14))
cv2.imwrite('res03.jpg', image)
image = cv2.imread("{i}.jpg".format(i=29))
cv2.imwrite('res04.jpg', image)

create_gif()
