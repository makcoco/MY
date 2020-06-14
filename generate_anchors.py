'''
Basic version, fork from yolo V2:
    Created on Feb 20, 2017
    @author: jumabek
'''
from os import listdir
from os.path import isfile, join
import argparse
# import cv2
import numpy as np
import sys
import os
import shutil
import random
import math

width_in_cfg_file = 416.
height_in_cfg_file = 416.


def IOU(x, centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w, c_h = centroid
        w, h = x
        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)  # will become (k,) shape
    return np.array(similarities)


def avg_IOU(X, centroids):
    n, d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        # note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective, but I am too lazy
        sum += max(IOU(X[i], centroids))
    return sum / n


def write_anchors_to_file(centroids, X, anchor_file):
    f = open(anchor_file, 'w')

    anchors = centroids.copy()
    print(anchors.shape)

    for i in range(anchors.shape[0]):
        anchors[i][0] *= width_in_cfg_file / 32.
        anchors[i][1] *= height_in_cfg_file / 32.

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    print('Anchors = ', anchors[sorted_indices])

    for i in sorted_indices[:-1]:
        f.write('%0.2f,%0.2f, ' % (anchors[i, 0], anchors[i, 1]))

    # there should not be comma after last anchor, that's why
    f.write('%0.2f,%0.2f\n' % (anchors[sorted_indices[-1:], 0], anchors[sorted_indices[-1:], 1]))

    f.write('%f\n' % (avg_IOU(X, centroids)))
    print()


def kmeans(X, centroids, eps, anchor_file):
    N = X.shape[0]
    iterations = 0
    k, dim = centroids.shape
    prev_assignments = np.ones(N) * (-1)
    iter = 0
    old_D = np.zeros((N, k))

    while True:
        D = []
        iter += 1
        for i in range(N):
            d = 1 - IOU(X[i], centroids)
            D.append(d)
        D = np.array(D)  # D.shape = (N,k)

        print("iter {}: dists = {}".format(iter, np.sum(np.abs(old_D - D))))

        # assign samples to centroids
        assignments = np.argmin(D, axis=1)

        if (assignments == prev_assignments).all():
            print("Centroids = ", centroids)
            write_anchors_to_file(centroids, X, anchor_file)
            return

        # calculate new centroids
        centroid_sums = np.zeros((k, dim), np.float)
        for i in range(N):
            centroid_sums[assignments[i]] += X[i]
        for j in range(k):
            centroids[j] = centroid_sums[j] / (np.sum(assignments == j))

        prev_assignments = assignments.copy()
        old_D = D.copy()

def get_w_h(labpath):
    bs = np.loadtxt(labpath)
    bs = np.reshape(bs, (-1, 21))
    for i in range(bs.shape[0]):
        x0 = bs[i][1]
        y0 = bs[i][2]
        x1 = bs[i][3]
        y1 = bs[i][4]
        x2 = bs[i][5]
        y2 = bs[i][6]
        x3 = bs[i][7]
        y3 = bs[i][8]
        x4 = bs[i][9]
        y4 = bs[i][10]
        x5 = bs[i][11]
        y5 = bs[i][12]
        x6 = bs[i][13]
        y6 = bs[i][14]
        x7 = bs[i][15]
        y7 = bs[i][16]
        x8 = bs[i][17]
        y8 = bs[i][18]
    x_min = min([x0, x1, x2, x3, x4, x5, x6, x7, x8])
    x_max = max([x0, x1, x2, x3, x4, x5, x6, x7, x8])
    y_min = min([y0, y1, y2, y3, y4, y5, y6, y7, y8])
    y_max = max([y0, y1, y2, y3, y4, y5, y6, y7, y8])

    return (x_max-x_min), (y_max-y_min)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_dir', default='\\path\\to\\LINEMOD',
                        help='path to filelist\n')
    parser.add_argument('-output_dir', default='generated_anchors', type=str,
                        help='Output anchor directory\n')
    parser.add_argument('-num_clusters', default=5, type=int,
                        help='number of clusters\n')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # obj_list = ['milk_multi', 'shuimitao','mangguo']
    obj_list = ['milk','milk10']

    annotation_dims = []

    for obj in obj_list:
        filelist = args.dataset_dir + '/' + obj + '/train.txt'
        print(filelist)

        f = open(filelist)

        lines = [line.rstrip('\n') for line in f.readlines()]

        for line in lines:

            # line = line.replace('images','labels')
            # line = line.replace('img1','labels')
            line = line.replace('JPEGImages', 'labels')

            line = line.replace('.jpg', '.txt')
            line = line.replace('.png', '.txt')

            w, h = get_w_h(line)

            annotation_dims.append(tuple(map(float, (w, h))))

    annotation_dims = np.array(annotation_dims)
    eps = 0.005

    if args.num_clusters == 0:
        for num_clusters in range(1, 11):  # we make 1 through 10 clusters
            anchor_file = join(args.output_dir, 'anchors%d.txt' % (num_clusters))

            indices = [random.randrange(annotation_dims.shape[0]) for i in range(num_clusters)]
            centroids = annotation_dims[indices]
            kmeans(annotation_dims, centroids, eps, anchor_file)
            print('centroids.shape', centroids.shape)
    else:
        anchor_file = join(args.output_dir, 'anchors%d.txt' % (args.num_clusters))
        indices = [random.randrange(annotation_dims.shape[0]) for i in range(args.num_clusters)]
        centroids = annotation_dims[indices]
        kmeans(annotation_dims, centroids, eps, anchor_file)
        print('centroids.shape', centroids.shape)


if __name__ == "__main__":
    main(sys.argv)