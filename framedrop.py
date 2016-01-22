#!/usr/bin/env python

# Copyright (c) 2016 Matthew Earl
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
FRAME_SKIP = 10
ERODE_AMOUNT = 101

names = list(sorted(glob.glob(ALIGNED_GLOB)))
mask = (cv2.imread(MASK_PATH) > 0).astype(numpy.float32)
mask = cv2.GaussianBlur(mask, (ERODE_AMOUNT, ERODE_AMOUNT), 0) > 0.99


def find_weights():
    weights = collections.defaultdict(dict) 
    prev_layer = None
    layer = []

    def link_layers(layer1, layer2):
        for n1, m1 in layer1:
            for n2, m2 in layer2:
                weights[n1][n2] = numpy.linalg.norm(m2 - m1) ** 2.

    for n in names:
        im = cv2.imread(n)
        layer.append((n, (im * mask).astype(numpy.float32)))

        if len(layer) == FRAME_SKIP:
            if prev_layer is not None:
                link_layers(prev_layer, layer)
            prev_layer = layer
            layer = []

    if layer:
        link_layers(prev_layer, layer)

    assert weights, "Need at least {} input images".format(FRAME_SKIP + 1)

    return weights


weights = find_weights()
sources = names[:FRAME_SKIP]
if len(names) % FRAME_SKIP != 0:
    drains = names[-(len(names) % FRAME_SKIP):]
else:
    drains = names[-FRAME_SKIP:]
dist = {n: (0 if n in sources else None) for n in names}
parent = {}
for u in names:
    for v, weight in weights[u].items():
        if dist[v] is None or dist[v] > dist[u] + weight:
            dist[v] = dist[u] + weight
            parent[v] = u

def drain_to_source():
    v = min(drains, key=lambda v: dist[v])
    yield v
    while v not in sources:
        v = parent[v]
        yield v


with open(FILE_LIST, "w") as f:
    path = list(reversed(list(drain_to_source())))
    for n in path:
        f.write("{}\n".format(n))


print "{} / {}  =  {} %".format(len(path), len(names),
                                100. * len(path) / len(names))

