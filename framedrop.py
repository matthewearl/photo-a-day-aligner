#!/usr/bin/env python

# Copyright (c) 2015 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.


import collections
import glob
import os

import cv2
import numpy


MASK_PATH = "mask.png"
ALIGNED_GLOB = "./aligned/*.jpg"
FILE_LIST = "files.txt"
MAX_FRAME_SKIP = 8
ERODE_AMOUNT = 101


names = list(sorted(glob.glob(ALIGNED_GLOB)))[:50]
mask = (cv2.imread(MASK_PATH) > 0).astype(numpy.float32)
mask = cv2.GaussianBlur(mask, (ERODE_AMOUNT, ERODE_AMOUNT), 0) > 0.99


def read_ims():
    for n in names:
        im = cv2.imread(n)
        yield n, im


def find_lengths():

    xs = []
    ys = []

    lengths = collections.defaultdict(dict) 
    prev_masked = []
    for n, im in read_ims():
        prev_masked.append((n, (im * mask).astype(numpy.float32)))

        if len(prev_masked) >= MAX_FRAME_SKIP:
            prev_masked = prev_masked[-MAX_FRAME_SKIP:]
            n1, m1 = prev_masked[0]
            for n2, m2 in prev_masked[1:]:
                d = numpy.linalg.norm(m2 - m1) ** 2.

                lengths[n1][n2] = d
                xs.append(d)

            ys += range(1, MAX_FRAME_SKIP)

    #import matplotlib.pyplot as plt
    #plt.plot(numpy.array(xs), numpy.array(ys), 'b+')
    #plt.show()

    return lengths


lengths = find_lengths()
dist = {n: (0 if n == names[0] else None) for n in names}
parent = {}
for u in names:
    for v, weight in lengths[u].items():
        if dist[v] is None or dist[v] > dist[u] + weight:
            dist[v] = dist[u] + weight
            parent[v] = u


def drain_to_source(v):
    yield v
    while v != names[0]:
        v = parent[v]
        yield v


with open(FILE_LIST, "w") as f:
    path = list(reversed(list(drain_to_source(names[-1]))))
    for n in path:
        f.write("{}\n".format(n))


print "{} / {}  =  {} %".format(len(path), len(names),
                                100. * len(path) / len(names))

