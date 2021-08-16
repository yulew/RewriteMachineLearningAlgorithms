#  Expectation-Maximization
#  Major goal: Clustering for these data
#  1. k = ? # e.g. k=2
#  2. Centroids?: a. Initialization (random); b. later: determined by the maximization step and repeat ...
#  3. Data samples belong to which cluster: followed by expectation step and repeat
#  4. k= ?

import numpy as np

def standardization():
    pass

def centroids_init(k, D):
    centroids = []
    for _ in range(k):
        c = np.random.random(D) # 0, 1, 1) Random probability follows Gaussian distribution or mean distribution
        centroids.append(c)


def expectation(centroids, data):
    """
    to determine samples belong to which cluster
    :param centroids: k-len list
    :param data: samples, D*N array
    :return: sample_clusters: 1D np array, to get every sample's cluster label
    """
    N, D = data.shape
    sample_clusters = np.empty(N)
    total_distance = 0
    for i, d in enumerate(data):
        min_distance = float("inf")
        for j, c in enumerate(centroids):
            distance = np.sqrt(np.dot(d - c, d - c))
            if min_distance > distance:
                min_distance = distance
                sample_clusters[i] = j  # belongs to cluster j
            total_distance += min_distance
    return sample_clusters, total_distance



def maximization(data, sample_clusters, k):
    """
    Estimate centroids for next iteration
    :param sample_clusters:
    :return: centroids
    """
    centroids = []
    for i in range(k):
        c = np.sum(data[sample_clusters == i], axis=0)
        centroids.append(c)
    return centroids



def EM(k, data, iter_delta=0.01, iterations=100):
    """

    :param k: number of clusters
    :param data:
    :param iter_delta: the largest total distance difference between two iterations we can make iterations stop
    :return:
    """
    N, D = data.shape
    centroids = centroids_init(k, D)
    prev_tot_distance = float('inf')
    tot_distance = -float('inf')
    while abs(tot_distance - prev_tot_distance) > iter_delta or iter < iterations:
        prev_tot_distance = tot_distance
        sample_clusters, tot_distance = expectation(centroids, data)
        centroids = maximization(data, sample_clusters, k)
    return sample_clusters, centroids, tot_distance

def kmeans(data, highest_k = 10, iter_delta=0.01, iterations=100):
    """
    Choose a suitable k by finding elbow in the total_distance
    :param data:
    :param highest_k: upper limit of number of clusters
    :param iter_delta:
    :param iterations:
    :return:
    """
    # standardization

    total_distances = []
    for k in range(2, highest_k+1):
        sample_clusters, centroids, tot_distance = EM(k, data, iter_delta=iter_delta,iterations=iterations)
        total_distances.append(tot_distance)
    return total_distances  # and then plot it to find elbow to find a good k







