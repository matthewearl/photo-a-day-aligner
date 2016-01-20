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


import glob
import os

import cv2
import dlib
import numpy
import scipy


PREDICTOR_PATH = "/home/matt/dlib-18.16/shape_predictor_68_face_landmarks.dat"
IMG_THRESH = 2000.
INPUT_GLOB = "*.jpg"
OUT_PATH = "./aligned"
MASK_PATH = "mask.png"

# Mean face colour to adjust to. Change to `None` to use the first face, as
# reference, 
REF_COLOR = [108.46239451, 129.73562552, 164.23183483]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


def read_ims(names):
    count = 0
    total = 0
    prev_im = None
    for n in names:
        im = cv2.imread(n)
        if prev_im is None or numpy.linalg.norm(prev_im - im) > IMG_THRESH:
            yield (n, im)
            count += 1
            prev_im = im
        total += 1
    print "Read {} / {} images".format(count, total)


class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass


def get_landmarks(im):
    rects = detector(im, 1)
    
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def orthogonal_procrustes(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:

        sum ||s*R*p1,i + T - p2,i||^2

    is minimized.

    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])


def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im


def read_ims_and_landmarks():
    count = 0
    for n, im in read_ims(sorted(glob.glob(INPUT_GLOB))):
        try:
            l = get_landmarks(im)
        except NoFaces:
            print "Warning: No faces in image {}".format(n)
        except TooManyFaces:
            print "Warning: Too many faces in image {}".format(n)
        else:
            yield (n, im, l)
            count += 1
    print "Read {} images with landmarks".format(count)


def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


def get_face_mask(shape, landmarks):
    im = numpy.zeros(shape[:2], dtype=numpy.float64)
    draw_convex_hull(im,
                     landmarks,
                     color=1)

    return im


ref_landmarks = None
ref_color = numpy.array(REF_COLOR) if REF_COLOR else None
prev_masked_ims = []
for n, im, landmarks in read_ims_and_landmarks():
    mask = get_face_mask(im.shape, landmarks)
    masked_im = mask[:, :, numpy.newaxis] * im
    color = ((numpy.sum(masked_im, axis=(0, 1)) /
              numpy.sum(mask, axis=(0, 1))))
    if ref_landmarks is None:  
        cv2.imwrite(MASK_PATH, mask * 255.)
        ref_landmarks = landmarks
    if ref_color is None:
        ref_color = color
    M = orthogonal_procrustes(ref_landmarks, landmarks)
    warped = warp_im(im, M, im.shape)
    warped_corrected = warped * ref_color / color
    cv2.imwrite(os.path.join(OUT_PATH, n), warped_corrected)
