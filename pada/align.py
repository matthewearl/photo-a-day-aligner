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
    'align_images',
)


import glob
import os

import cv2
import dlib
import numpy
import scipy

from . import landmarks
from .logging import logger


def read_ims(names, img_thresh):
    count = 0
    total = 0
    prev_im = None
    for n in names:
        logger.debug("Reading image %s", n)
        im = cv2.imread(n)
        if prev_im is None or numpy.linalg.norm(prev_im - im) > img_thresh:
            yield (n, im)
            count += 1
            prev_im = im
        else:
            logger.debug("Ignoring %s as it is a duplicate", n)
        total += 1
    logger.info("Read %s / %s images", count, total)


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


def get_ims_and_landmarks(images, landmark_finder):
    count = 0
    for n, im in images:
        try:
            l = landmark_finder.get(im)
        except landmarks.NoFaces:
            logger.warn("No faces in image %s", n)
        except landmarks.TooManyFaces:
            logger.warn("Too many faces in image %s", n)
        else:
            yield (n, im, l)
            count += 1
    logger.info("Read %s images with landmarks", count)


def align_images(input_files, out_path, out_extension, landmark_finder,
                 img_thresh=0.0):
    """
    Align a set of images of a person's face.

    :param input_files:

        A list of file names to be aligned.

    :param out_path:

        Directory to write the aligned files to. The output files have the same
        basename as the corresponding input file.

    :param out_extension:

        Extension to use for aligned images.

    :param landmark_finder:

        An instance of :class:`.LandmarkFinder`, used to find the facial
        landmarks.

    :param img_thresh:

        Images with an with this distance of the previous image (using the L2
        norm) are considered duplicates and are ignored.

    """
    ref_landmarks = None
    ref_color = None
    prev_masked_ims = []

    # Clean up the out_path, or create it it if necessary.
    if os.path.exists(out_path):
        if not os.path.isdir(out_path):
            raise Exception("Path {} exists, but it is not a directory".format(
                                                                     out_path))
        logger.info("%s already exists. Removing existing images.", out_path)
        for fname in glob.glob(os.path.join(out_path,
                                            "*.{}".format(out_extension))):
            os.remove(fname)
    else:
        logger.info("%s does not exist. Creating it.", out_path)
        os.mkdir(out_path)

    # Process each file in turn.
    ims_and_landmarks = get_ims_and_landmarks(
                                  read_ims(input_files, img_thresh=img_thresh),
                                  landmark_finder)
    for idx, (n, im, lms) in enumerate(ims_and_landmarks):
        mask = landmarks.get_face_mask(im.shape, lms)
        masked_im = mask[:, :, numpy.newaxis] * im
        color = ((numpy.sum(masked_im, axis=(0, 1)) /
                  numpy.sum(mask, axis=(0, 1))))
        if ref_landmarks is None:  
            ref_landmarks = lms
        if ref_color is None:
            ref_color = color
        M = orthogonal_procrustes(ref_landmarks, lms)
        warped = warp_im(im, M, im.shape)
        warped_corrected = warped * ref_color / color
        out_fname = os.path.join(out_path,
                                 "{:08d}.{}".format(idx, out_extension))
        cv2.imwrite(out_fname, warped_corrected)
        logger.debug("Wrote file %s", out_fname)

