
# system
from __future__ import print_function

# python lib
import importlib
import numpy as np

# tf_render
import tensorflow as tf

# self

"""
*******************************     numpy functions     *******************************
"""

def read_syn_info_file(path_calib_file):
    with open(path_calib_file, 'r') as f:
        intrinsic = f.readline()
        intrinsic = np.array([float(x) for x in intrinsic.split()])
        mtx_intrinsic = np.reshape(intrinsic, [3, 3])

        ext = f.readline()
        ext = np.array([float(x) for x in ext.split()])
        eular_angle = ext[0:3]
        translation = ext[3:]

        lm2d_num = f.readline()
        lm2d = f.readline()
        lm2d = np.array([float(x) for x in lm2d.split()])

        f.close()

    return mtx_intrinsic, eular_angle, translation, lm2d

def scale_intrinsics(mat, sx, sy):
    out = np.copy(mat)
    out[0,0] *= sx
    out[0,2] *= sx
    out[1,1] *= sy
    out[1,2] *= sy
    return out

def read_mpie_info_file(path_calib_file):
    with open(path_calib_file, 'r') as f:
        intrinsic = f.readline()
        intrinsic = np.array([float(x) for x in intrinsic.split(',')])
        mtx_intrinsic = np.reshape(intrinsic, [3, 3])

        lm2d_num = f.readline()
        lm2d_num = lm2d_num.strip()
        list_lm2d = []
        if not len(lm2d_num):
            pass
        else:
            for i in range(int(lm2d_num)):
                lm2d = f.readline()
                lm2d = np.array([float(x) for x in lm2d.split(',')])
                list_lm2d.append(lm2d)

    return mtx_intrinsic, list_lm2d

"""
*******************************     tensor functions     *******************************
"""


def make_intrinsics_matrix(fx, fy, cx, cy):
    # Assumes batch input
    batch_size = tf.shape(fx)[0]
    zeros = tf.zeros_like(fx)
    r1 = tf.stack([fx, zeros, cx], axis=1)
    r2 = tf.stack([zeros, fy, cy], axis=1)
    r3 = tf.constant([0., 0., 1.], shape=[1, 3])
    r3 = tf.tile(r3, [batch_size, 1])
    intrinsics = tf.stack([r1, r2, r3], axis=1)
    return intrinsics

def data_augmentation(im, intrinsics, out_h, out_w, matches=None):
    out_h = tf.cast(out_h, dtype=tf.int32)
    out_w = tf.cast(out_w, dtype=tf.int32)

    # Random scaling
    def random_scaling(im, intrinsics, matches):
        # print(tf_render.unstack(tf_render.shape(im)))
        # print(im.get_shape().as_list())
        _, in_h, in_w, _ = tf.unstack(tf.shape(im))
        in_h = tf.cast(in_h, dtype=tf.float32)
        in_w = tf.cast(in_w, dtype=tf.float32)
        scaling = tf.random_uniform([1], 1.0, 1.2)
        x_scaling = scaling[0]
        y_scaling = scaling[0]

        out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
        out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)

        im = tf.image.resize_area(im, [out_h, out_w])
        fx = intrinsics[:, 0, 0] * x_scaling
        fy = intrinsics[:, 1, 1] * y_scaling
        cx = intrinsics[:, 0, 2] * x_scaling
        cy = intrinsics[:, 1, 2] * y_scaling
        intrinsics = make_intrinsics_matrix(fx, fy, cx, cy)

        if matches is None:
            return im, intrinsics, None
        else:
            x = matches[:, :, :, 0] * x_scaling
            y = matches[:, :, :, 1] * y_scaling
            matches = tf.stack([x, y], axis=3)  # bs, tar, num, axis
            return im, intrinsics, matches

    # Random cropping
    def random_cropping(im, intrinsics, out_h, out_w, matches):
        # batch_size, in_h, in_w, _ = im.get_shape().as_list()
        batch_size, in_h, in_w, _ = tf.unstack(tf.shape(im))
        offset_y = tf.random_uniform([1], 0, in_h-out_h+1, dtype=tf.int32)[0]
        offset_x = offset_y#tf_render.random_uniform([1], 0, in_w-out_w+1, dtype=tf_render.int32)[0]
        im = tf.image.crop_to_bounding_box(
            im, offset_y, offset_x, out_h, out_w)
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2] - tf.cast(offset_x, dtype=tf.float32)
        cy = intrinsics[:, 1, 2] - tf.cast(offset_y, dtype=tf.float32)
        intrinsics = make_intrinsics_matrix(fx, fy, cx, cy)

        if matches is None:
            return im, intrinsics, None
        else:
            x = matches[:, :, :, 0] - tf.cast(offset_x, dtype=tf.float32)
            y = matches[:, :, :, 1] - tf.cast(offset_y, dtype=tf.float32)
            matches = tf.stack([x, y], axis=3)  # bs, tar, num, axis
            return im, intrinsics, matches

    batch_size, in_h, in_w, _ = tf.unstack(tf.shape(im))
    im, intrinsics, matches = random_scaling(im, intrinsics, matches)
    im, intrinsics, matches = random_cropping(im, intrinsics, out_h, out_w, matches)
    #im, intrinsics, matches = random_scaling(im, intrinsics, matches, in_h, in_w)
    im = tf.cast(im, dtype=tf.uint8)

    if matches is None:
        return im, intrinsics, intrinsics
    else:
        return im, intrinsics, matches

# different at intrinsics is [bs, num_src+1, 3, 3]
def data_augmentation_mul(im, intrinsics, out_h, out_w, matches=None):
    out_h = tf.cast(out_h, dtype=tf.int32)
    out_w = tf.cast(out_w, dtype=tf.int32)

    # Random scaling
    def random_scaling(im, intrinsics, matches):
        # print(tf_render.unstack(tf_render.shape(im)))
        # print(im.get_shape().as_list())
        _, in_h, in_w, _ = tf.unstack(tf.shape(im))
        in_h = tf.cast(in_h, dtype=tf.float32)
        in_w = tf.cast(in_w, dtype=tf.float32)
        scaling = tf.random_uniform([2], 1.0, 1.2)
        x_scaling = scaling[0]
        y_scaling = scaling[0]

        out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
        out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)

        im = tf.image.resize_area(im, [out_h, out_w])

        list_intrinsics = []
        for i in range(intrinsics.shape[1]): # bs, num_src+1, 3, 3
            fx = intrinsics[:, i, 0, 0] * x_scaling
            fy = intrinsics[:, i, 1, 1] * y_scaling
            cx = intrinsics[:, i, 0, 2] * x_scaling
            cy = intrinsics[:, i, 1, 2] * y_scaling
            intrinsics_new = make_intrinsics_matrix(fx, fy, cx, cy)
            list_intrinsics.append(intrinsics_new)
        intrinsics = tf.stack(list_intrinsics, axis=1)

        if matches is None:
            return im, intrinsics, None
        else:
            x = matches[:, :, :, 0] * x_scaling
            y = matches[:, :, :, 1] * y_scaling
            matches = tf.stack([x, y], axis=3)  # bs, tar, num, axis
            return im, intrinsics, matches

    # Random cropping
    def random_cropping(im, intrinsics, out_h, out_w, matches):
        # batch_size, in_h, in_w, _ = im.get_shape().as_list()
        batch_size, in_h, in_w, _ = tf.unstack(tf.shape(im))
        offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
        offset_x = offset_y
        im = tf.image.crop_to_bounding_box(
            im, offset_y, offset_x, out_h, out_w)

        list_intrinsics = []
        for i in range(intrinsics.shape[1]): # bs, num_src+1, 3, 3
            fx = intrinsics[:, i, 0, 0]
            fy = intrinsics[:, i, 1, 1]
            cx = intrinsics[:, i, 0, 2] - tf.cast(offset_x, dtype=tf.float32)
            cy = intrinsics[:, i, 1, 2] - tf.cast(offset_y, dtype=tf.float32)
            intrinsics_new = make_intrinsics_matrix(fx, fy, cx, cy)
            list_intrinsics.append(intrinsics_new)
        intrinsics = tf.stack(list_intrinsics, axis=1)

        if matches is None:
            return im, intrinsics, None
        else:
            x = matches[:, :, :, 0] - tf.cast(offset_x, dtype=tf.float32)
            y = matches[:, :, :, 1] - tf.cast(offset_y, dtype=tf.float32)
            matches = tf.stack([x, y], axis=3)  # bs, tar, num, axis
            return im, intrinsics, matches

    batch_size, in_h, in_w, _ = tf.unstack(tf.shape(im))
    im, intrinsics, matches = random_scaling(im, intrinsics, matches)
    im, intrinsics, matches = random_cropping(im, intrinsics, out_h, out_w, matches)
    # im, intrinsics, matches = random_scaling(im, intrinsics, matches, in_h, in_w)
    im = tf.cast(im, dtype=tf.uint8)

    if matches is None:
        return im, intrinsics, None
    else:
        return im, intrinsics, matches
