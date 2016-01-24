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


__all__ = (
    'filter_files',
)


import collections
import glob
import os

import cv2
import numpy

from . import landmarks
from .logging import logger


def find_weights(names, mask, frame_skip):
    weights = collections.defaultdict(dict) 
    prev_layer = None
    layer = []

    def link_layers(layer1, layer2):
        for n1, m1 in layer1:
            for n2, m2 in layer2:
                weights[n1][n2] = numpy.linalg.norm(m2 - m1)

    for n in names:
        im = cv2.imread(n)
        masked_im = (im * mask[:, :, numpy.newaxis]).astype(numpy.float32)
        layer.append((n, masked_im))

        if len(layer) == frame_skip:
            if prev_layer is not None:
                link_layers(prev_layer, layer)
            prev_layer = layer
            layer = []

    if layer:
        link_layers(prev_layer, layer)

    assert weights, "Need at least {} input images".format(frame_skip + 1)

    return weights


def make_mask(im_name, erode_amount, landmark_finder):
    """
    Define a mask which is the eroded convex hull of the face in the given
    image.

    The returned mask is used for measuring frame difference in the filtering
    algorithm.

    """
    im = cv2.imread(im_name)
    lm = landmark_finder.get(im)
    mask = landmarks.get_face_mask(im.shape, lm)
    mask = cv2.GaussianBlur(mask, (erode_amount, erode_amount), 0) > 0.99

    return mask
    

def filter_files(input_files, frame_skip, erode_amount, landmark_finder):
    """
    Filter video frames, minimizing total frame different.

    :param input_files:
        Iterable of file names of video frames.

    :param frame_skip:
        Number of frames in each layer.

    :param erode_amount:
        Amount to erode the input mask by.

    :param landmark_finder:
        An instance of :class:`.LandmarkFinder`, used to find the facial
        landmarks.

    """
    input_files = list(input_files)
    logger.info("Filtering %s files", len(input_files))

    # Make a mask, which defines the area over which frame difference is
    # measured.
    logger.debug("Making mask")
    mask = make_mask(input_files[0], erode_amount, landmark_finder)

    # Find the nodes in the first and last layer.
    logger.debug("Finding weights")
    weights = find_weights(input_files, mask, frame_skip)
    sources = input_files[:frame_skip]
    if len(input_files) % frame_skip != 0:
        drains = input_files[-(len(input_files) % frame_skip):]
    else:
        drains = input_files[-frame_skip:]

    # Compute `dist` which gives the minimum distance from any frame to a start
    # frame, and `parent` which given a node returns the previous node on the
    # shortest path to that node.
    logger.debug("Computing distances")
    dist = {n: (0 if n in sources else None) for n in input_files}
    parent = {}
    for u in input_files:
        for v, weight in weights[u].items():
            if dist[v] is None or dist[v] > dist[u] + weight:
                dist[v] = dist[u] + weight
                parent[v] = u

    # Find the end node which has least distance, and step through the shortest
    # path until we hit a start node.
    def drain_to_source():
        v = min(drains, key=lambda v: dist[v])
        yield v
        while v not in sources:
            v = parent[v]
            yield v

    # Reverse to find the shortest path.
    path = list(reversed(list(drain_to_source())))
    for n in path:
        yield n

    logger.info("Kept %s / %s (%s %%) frames", 
                                           len(path), len(input_files),
                                           100. * len(path) / len(input_files))

