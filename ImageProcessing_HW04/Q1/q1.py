import copy
import math
import random
import numpy as np
from matplotlib import pyplot as plt


def K_Means(data, k):
    _, n = data.shape
    centroids = get_centroids(data, k, n)
    clusters = cluster(data, centroids, k, n)
    return clusters


def get_centroids(data, k, n):
    centroids_init = []
    for i in range(k):
        index = random.randint(0, n - 1)
        centroids_init.append(np.ravel(data[:, index]))
    return np.array(centroids_init)


def cluster(data, centroids, k, n):
    centroids_prev = np.zeros(centroids.shape)
    threshold = 0.0001

    while True:
        if distance(centroids_prev, centroids) < threshold:
            break
        clusters = np.zeros(n)
        for i in range(n):
            distances = []
            for j in range(k):
                distances.append(distance(data[:, i], centroids[j]))
            cluster = np.argmin(distances)
            clusters[i] = cluster

        centroids_prev = copy.deepcopy(centroids)

        for i in range(k):
            points = data[:, clusters == i]
            centroids[i] = np.mean(points, 1)

    return clusters


def distance(a, b):
    return np.sqrt(np.sum((np.array(a) - np.array(b)) ** 2))


def to_polar(p):
    x, y = p[0], p[1]
    return np.sqrt(x ** 2 + y ** 2), math.atan(y / x)


def to_cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


data = np.loadtxt('Points.txt', delimiter=' ', unpack=True, skiprows=1)
k = 2
plt.scatter(data[0, :], data[1, :])
plt.savefig("res01.jpg")
plt.clf()

clusters = K_Means(data, k)
plt.scatter(data[0, :], data[1, :], c=clusters)
plt.savefig("res02.jpg")
plt.clf()

clusters = K_Means(data, k)
plt.scatter(data[0, :], data[1, :], c=clusters)
plt.savefig("res03.jpg")
plt.clf()

data_p = np.apply_along_axis(to_polar, 0, data)
clusters = K_Means(data_p, k)
plt.scatter(data[0, :], data[1, :], c=clusters)
plt.savefig("res04.jpg")
