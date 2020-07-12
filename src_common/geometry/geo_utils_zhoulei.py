import tensorflow as tf
import numpy as np
import math

def bilinear_sampler(imgs, coords):
    """
    Construct an new image based on the src images and coordinates of pixels in the src
    :param imgs: BxH1xW1x3
    :param coords: BxH2xH2x2
    :return: BxH2xH2x3
    """
    def _repeat(x, n_repeats):
        ones = tf.ones(shape=tf.stack([1, n_repeats]), dtype=tf.float32) #1xN
        x = tf.matmul(tf.reshape(x, shape=[-1, 1]), ones)
        return tf.reshape(x, [-1])


    with tf.name_scope("bilinear_sampler"):
        img_size = imgs.get_shape()
        coord_size = coords.get_shape()
        out_size = coord_size.as_list()
        out_size[3] = img_size.as_list()[3]

        # compute bilinear weights
        coords_x, coords_y = tf.split(coords, [1, 1], axis=-1)
        coords_x = tf.cast(coords_x, tf.float32)
        coords_y = tf.cast(coords_y, tf.float32)

        x0 = tf.floor(coords_x)
        x1 = x0 + 1
        y0 = tf.floor(coords_y)
        y1 = y0 + 1

        y_max = tf.cast(img_size[1] - 1, tf.float32)
        x_max = tf.cast(img_size[2] - 1, tf.float32)
        zero = tf.zeros([1], dtype=tf.float32)
        x0_safe = tf.clip_by_value(x0, zero, x_max)
        y0_safe = tf.clip_by_value(y0, zero, y_max)
        x1_safe = tf.clip_by_value(x1, zero, x_max)
        y1_safe = tf.clip_by_value(y1, zero, y_max)

        wt_x0 = x1_safe - coords_x
        wt_x1 = coords_x - x0_safe
        wt_y0 = y1_safe - coords_y
        wt_y1 = coords_y - y0_safe

        # indices in the flat image to sample from
        dim2 = tf.cast(img_size[2], tf.float32)
        dim1 = tf.cast(img_size[1] * img_size[2], tf.float32)
        rep = _repeat(tf.cast(tf.range(coord_size[0]), tf.float32) * dim1, coord_size[1] * coord_size[2])
        base = tf.reshape(rep, shape=[out_size[0], out_size[1], out_size[2], 1])
        base_y0 = base + y0_safe * dim2
        base_y1 = base + y1_safe * dim2
        idx00 = tf.reshape(x0_safe, base_y0, [-1])
        idx01 = x0_safe + base_y1
        idx10 = x1_safe + base_y0
        idx11 = x1_safe + base_y1

        # sample from imgs
        imgs_flat = tf.reshape(imgs, shape=tf.stack(-1, img_size[3]))
        imgs_flat = tf.cast(imgs_flat, dtype=tf.float32)
        im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, tf.int32)), out_size)
        im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, tf.int32)), out_size)
        im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, tf.int32)), out_size)
        im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, tf.int32)), out_size)

        w00 = wt_x0 * wt_y0
        w01 = wt_x0 * wt_y1
        w10 = wt_x1 * wt_y0
        w11 = wt_x1 * wt_y1

        output = tf.add_n([w00 * im00, w01 * im01, w10 * im10, w11 * im11])
        return output

############################ EPnP ##############################
def extendCoordMap(coord_map, spec):
    """
    Extend the coordinates from [a1, a2, a3] to [a1, a2, a3, 1-a1-a2-a3]
    :param coord_map: BxHxWx3
    :return: BxHxWx4
    """
    sum_coord_map = tf.reduce_sum(coord_map, -1)  # BxHxW
    coord_map_4th = tf.ones([spec.batch_size, spec.crop_size[0], spec.crop_size[1]]) - sum_coord_map  # BxHxW
    coord_map_4th = tf.expand_dims(coord_map_4th, axis=-1)
    coord_map_4 = tf.concat([coord_map, coord_map_4th], axis=-1, name='coord_map_4')  # BxHxWx4
    return coord_map_4

def get_EPnP_coefficient1(coord_map, pixel_map, spec):
    """
    Get the coefficients of the linear equation used in EPnP: for each pixel,
    [a1, a2, a3, a4, 0, 0, 0, 0, -ua1, -ua2, -ua3, -ua4]
    :param coord_map: BxHxWx4
    :param pixel_map: BxHxWx2
    :return: BxHxWx12
    """
    batch_size, height, width, _ = coord_map.get_shape().as_list()

    # get [0, 0, 0, 0]
    zero_map_4 = tf.zeros([batch_size, height, width, 4])

    # get [-ua1, -ua2, -ua3, -ua4]
    u_map = tf.slice(pixel_map, [0, 0, 0, 0], [-1, -1, -1, 1], name='u_map')  # BxHxWx1
    u_map_4 = tf.tile(u_map, [1, 1, 1, 4])  # BxHxWx4
    u_coord_map = u_map_4 * coord_map
    u_coord_map = tf.multiply(-1.0, u_coord_map, name='u_coord_map')

    # get [a1, a2, a3, a4, 0, 0, 0, 0, -ua1, -ua2, -ua3, -ua4]
    full_map = tf.concat([coord_map, zero_map_4, u_coord_map], axis=-1, name='coefficients1')  # BxHxWx12
    return full_map

def get_EPnP_coefficient2(coord_map, pixel_map, spec):
    """
    Get the coefficients of the linear equation used in EPnP: for each pixel,
    [0, 0, 0, 0, a1, a2, a3, a4, -va1, -va2, -va3, -va4]
    :param coord_map: BxHxWx4
    :param pixel_map: BxHxWx2
    :return: BxHxWx12
    """
    batch_size, height, width, _ = coord_map.get_shape().as_list()

    # get [0, 0, 0, 0]
    zero_map_4 = tf.zeros([batch_size, height, width, 4])

    # get [-va1, -va2, -va3, -va4]
    v_map = tf.slice(pixel_map, [0, 0, 0, 1], [-1, -1, -1, 1])  # BxHxWx1
    v_map_4 = tf.tile(v_map, [1, 1, 1, 4])  # BxHxWx4
    v_coord_map = v_map_4 * coord_map
    v_coord_map = tf.multiply(-1.0, v_coord_map, name='v_coord_map')

    # get [0, 0, 0, 0, a1, a2, a3, a4, -va1, -va2, -va3, -va4]
    full_map = tf.concat([zero_map_4, coord_map, v_coord_map], axis=-1, name='coefficients2')  # BxHxWx12
    return full_map

def bary_to_euc_coord(bary_coord_map, control_points_x, control_points_y, control_points_z):
    """
    Convert barycentric coordinates to euclidean coordinates
    :param bary_coord_map: HXWX4
    :param control_points: 1x4
    :return: HxWx3
    """
    coord_x = np.matmul(bary_coord_map, control_points_x)  # HxWx1
    coord_y = np.matmul(bary_coord_map, control_points_y)  # HxWx1
    coord_z = np.matmul(bary_coord_map, control_points_z)  # HxWx1
    coord = np.stack((coord_x, coord_y, coord_z), axis=-1)  # HxWx3
    return coord

def GetEPnPControlPoints(coord_map, pixel_map, spec):
    """
    Implementation of EPnP algorithm
    :param coord_map: BxHxWx4   Barycentric coord_map
    :param pixel_map: BxHxWx3   pixel map
    :return: Bx12   camera control points
    """
    batch_size, height, width, _ = coord_map.get_shape().as_list()

    coefficients1 = get_EPnP_coefficient1(coord_map, pixel_map, spec)  # BxHxWx12
    coefficients2 = get_EPnP_coefficient2(coord_map, pixel_map, spec)  # BxHxWx12
    coefficients1 = tf.reshape(coefficients1, shape=[batch_size, -1, 12], name='coeff1')  # BxHWx12
    coefficients2 = tf.reshape(coefficients2, shape=[batch_size, -1, 12], name='coeff2')  # BxHWx12

    # regular loss
    square_coefficients1 = tf.matmul(coefficients1, coefficients1, transpose_a=True)  # Bx12x12
    square_coefficients2 = tf.matmul(coefficients2, coefficients2, transpose_a=True)  # Bx12x12
    square_coefficients = square_coefficients1 + square_coefficients2
    eigen_values, eigen_vectors = tf.self_adjoint_eig(square_coefficients)
    min_eigen_vectors = tf.slice(eigen_vectors, [0, 0, 0], [-1, -1, 1])
    min_eigen_vectors = tf.squeeze(min_eigen_vectors, axis=[-1])
    min_eigen_vectors = tf.nn.l2_normalize(min_eigen_vectors, axis=-1)  # Bx12
    control_points = tf.nn.l2_normalize(min_eigen_vectors, axis=-1, name='est_cps')    # Bx12
    return control_points

def GetWeightedEPnPControlPoints(coord_map, pixel_map, world_cps, spec, weight_map=None, mask=None):
    """
    Implementation of EPnP algorithm
    :param coord_map: BxHxWx4   Barycentric coord_map
    :param pixel_map: BxHxWx3   pixel map
    :param weight_map: BxHxWx1
    :param mask: BxHxWx1
    :return: Bx12   camera control points
    """
    batch_size, height, width, _ = coord_map.get_shape().as_list()

    coefficients1 = get_EPnP_coefficient1(coord_map, pixel_map, spec)  # BxHxWx12
    coefficients2 = get_EPnP_coefficient2(coord_map, pixel_map, spec)  # BxHxWx12

    if mask is not None:
        mask_12 = tf.tile(mask, [1, 1, 1, 12])
        coefficients1 = mask_12 * coefficients1
        coefficients2 = mask_12 * coefficients2

    coefficients1 = tf.reshape(coefficients1, shape=[batch_size, -1, 12], name='coeff1')  # BxHWx12
    coefficients2 = tf.reshape(coefficients2, shape=[batch_size, -1, 12], name='coeff2')  # BxHWx12

    if weight_map is not None:
        weight_map_12 = tf.tile(tf.reshape(weight_map, [spec.batch_size, -1, 1]), [1, 1, 12])  # BxHWx12
        weighted_coefficients1 = coefficients1 * weight_map_12
        weighted_coefficients2 = coefficients2 * weight_map_12
        square_coefficients1 = tf.matmul(weighted_coefficients1, weighted_coefficients1, transpose_a=True)  # Bx12x12
        square_coefficients2 = tf.matmul(weighted_coefficients2, weighted_coefficients2, transpose_a=True)  # Bx12x12
    else:
        square_coefficients1 = tf.matmul(coefficients1, coefficients1, transpose_a=True)  # Bx12x12
        square_coefficients2 = tf.matmul(coefficients2, coefficients2, transpose_a=True)  # Bx12x12

    square_coefficients = square_coefficients1 + square_coefficients2
    eigen_values, eigen_vectors = tf.self_adjoint_eig(square_coefficients)
    min_eigen_vectors = tf.slice(eigen_vectors, [0, 0, 0], [-1, -1, 1])
    min_eigen_vectors = tf.squeeze(min_eigen_vectors, axis=[-1])
    control_points = tf.nn.l2_normalize(min_eigen_vectors, axis=-1, name='est_cps')    # Bx12
    world_cps_reshape = tf.reshape(world_cps, [3, 4])
    camera_cps_reshape = tf.reshape(control_points, [3, 4])
    scale = ScaleFromPoints(camera_cps_reshape, world_cps_reshape)
    control_points = scale * control_points

    coefficient_error1 = tf.matmul(coefficients1, tf.expand_dims(control_points, axis=-1))
    coefficient_error1 = tf.reduce_sum(tf.square(coefficient_error1), axis=-1)  # BxHW
    coefficient_error2 = tf.matmul(coefficients2, tf.expand_dims(control_points, axis=-1))
    coefficient_error2 = tf.reduce_sum(tf.square(coefficient_error2), axis=-1)  # BxHW
    coefficient_error = tf.sqrt(coefficient_error1 + coefficient_error2) * spec.focal_x / 2.
    coefficient_error = tf.reshape(coefficient_error, shape=[batch_size, height, width, 1])
    return control_points, coefficient_error

def GetWeightedEPnPControlPointsInterative(coord_map, pixel_map, world_cps, weight_map, spec,
                                           mask=None, num_iteration=10):

    # weight_mask = tf_render.cast(tf_render.greater(weight_map, 0.9), tf_render.float32)
    # if mask is not None:
    #     mask = mask * weight_mask
    # else:
    #     mask = weight_mask
    batch_size, height, width, _ = coord_map.get_shape().as_list()

    for i in range(num_iteration):
        control_points, coefficient_error = GetWeightedEPnPControlPoints(coord_map, pixel_map, world_cps, spec, weight_map, mask)
        errors = tf.reshape(coefficient_error, [-1])
        num_nonzeros = tf.count_nonzero(errors)
        quatile_position = tf.cast(num_nonzeros * 3 // 4, tf.int32)
        quatile_error = tf.nn.top_k(errors, quatile_position).values[-1]
        weight_map1 = tf.cast(tf.less(errors, quatile_error), tf.float32)
        weight_map2 = tf.cast(tf.less(errors, 10.0), tf.float32)
        weight_map = weight_map1 + weight_map2
        weight_map = tf.cast(tf.greater_equal(weight_map, 1.), tf.float32)

        weight_map = tf.reshape(weight_map, [batch_size, height, width, 1])
        if mask is not None:
            weight_map = mask * weight_map

    return control_points, weight_map


def EPnPReprojectionLoss(coord_map, pixel_map, control_points, spec):
    """
    Compute reprojection loss given barycentric coordinates and camera control points
    :param coord_map: BxHxWx4
    :param control_points: Bx12
    :return:
    """
    batch_size, height, width, _ = coord_map.get_shape().as_list()

    coefficients1 = get_EPnP_coefficient1(coord_map, pixel_map, spec)  # BxHxWx12
    coefficients2 = get_EPnP_coefficient2(coord_map, pixel_map, spec)  # BxHxWx12
    coefficients1 = tf.reshape(coefficients1, shape=[batch_size, -1, 12], name='coeff1')  # BxHWx12
    coefficients2 = tf.reshape(coefficients2, shape=[batch_size, -1, 12], name='coeff2')  # BxHWx12

    if len(control_points.get_shape()) == 2:
        control_points = tf.expand_dims(control_points, axis=-1)
    pnp_mul1 = tf.matmul(coefficients1, control_points, name='pnp_error_vector1')  # BxHWx1
    pnp_mul2 = tf.matmul(coefficients2, control_points, name='pnp_error_vector2')  # BxHWx1
    pnp_loss = tf.reduce_mean(tf.abs(pnp_mul1) + tf.abs(pnp_mul2), name='pnp_loss')
    return pnp_loss

def ControlPointLoss(est_cps, gt_cps):
    """
    :param est_cps: Bx12
    :param gt_cps: Bx12
    :return:
    """
    est_cps = tf.nn.l2_normalize(est_cps, axis=-1)  # Bx12
    norms = tf.norm(gt_cps, axis=-1, keepdims=True) # Bx12
    est_cps = est_cps * norms
    loss1 = tf.reduce_sum(tf.square(est_cps - gt_cps))
    loss2 = tf.reduce_sum(tf.square(est_cps + gt_cps))
    loss = tf.minimum(loss1, loss2)
    return loss

############################ eof EPnP ##############################

def DrawFromDistribution(probs):
    """
    :param probs: A list of probs, unnormalized
    :return:
    """
    num = len(probs)
    prob_vec = tf.stack(probs, axis=0, name='prob_distribution')  # N
    sum = tf.maximum(tf.reduce_sum(prob_vec), 1e-6)
    prob_vec = tf.div(prob_vec, sum)

    cum_prob = [0]
    sum = 0
    for i in range(num):
        prob = tf.gather(prob_vec, i)
        sum += prob
        cum_prob.append(sum)
    cum_prob = tf.stack(cum_prob, axis=0, name='CDF')   # N+1

    rand = tf.random_uniform((), 0, 1)
    cum_prob = rand - cum_prob
    negatives = tf.cast(tf.less(cum_prob, 0.0), tf.float32)
    cum_prob += 2 * negatives
    index = tf.argmin(cum_prob)
    return index

def TransformFromPointsNumpy(left_points, right_points):
    """
    Numpy implementatin of aligning left points to right points
    :param refer_points: list of 3D numpy arrays
    :param align_points: list of 3D numpy arrays
    :return:
    """
    def _CenterOfPoints(points):
        center = np.array([0., 0., 0.])
        for pt in points:
            center += pt
        center = center / len(points)
        return center

    def _UncenterPoints(points):
        center = _CenterOfPoints(points)
        u_points = []
        for pt in points:
            u_pt = pt - center
            u_points.append(u_pt)
        return u_points

    def _Quaternion2Mat(quat):
        w = quat[0]
        x = quat[1]
        y = quat[2]
        z = quat[3]
        rotation = np.zeros([3, 3])
        rotation[0, 0] = 1 - 2 * y * y - 2 * z * z
        rotation[0, 1] = 2 * x * y - 2 * z * w
        rotation[0, 2] = 2 * x * z + 2 * y * w
        rotation[1, 0] = 2 * x * y + 2 * z * w
        rotation[1, 1] = 1 - 2 * x * x - 2 * z * z
        rotation[1, 2] = 2 * y * z - 2 * x * w
        rotation[2, 0] = 2 * x * z - 2 * y * w
        rotation[2, 1] = 2 * y * z + 2 * x * w
        rotation[2, 2] = 1 - 2 * x * x - 2 * y * y
        return rotation


    assert len(left_points) == len(right_points)
    lefts = _UncenterPoints(left_points)
    rights = _UncenterPoints(right_points)

    ## Compute scale
    left_norm_square = 0.
    right_norm_square = 0.
    for pt in lefts:
        left_norm_square += np.linalg.norm(pt) * np.linalg.norm(pt)
    for pt in rights:
        right_norm_square += np.linalg.norm(pt) * np.linalg.norm(pt)
    scale = np.sqrt(right_norm_square / left_norm_square)
    print 'scale', scale

    ## Compute rotation
    M = np.zeros([3, 3])
    for (pt1, pt2) in zip(lefts, rights):
        M += np.outer(pt1, pt2)
    N = np.zeros([4, 4])
    N[0, 0] = M[0, 0] + M[1, 1] + M[2, 2]
    N[1, 1] = M[0, 0] - M[1, 1] - M[2, 2]
    N[2, 2] = -M[0, 0] + M[1, 1] - M[2, 2]
    N[3, 3] = -M[0, 0] - M[1, 1] + M[2, 2]
    N[0, 1] = M[1, 2] - M[2, 1]
    N[1, 0] = M[1, 2] - M[2, 1]
    N[0, 2] = M[2, 0] - M[0, 2]
    N[2, 0] = M[2, 0] - M[0, 2]
    N[0, 3] = M[0, 1] - M[1, 0]
    N[3, 0] = M[0, 1] - M[1, 0]
    N[1, 2] = M[0, 1] + M[1, 0]
    N[2, 1] = M[0, 1] + M[1, 0]
    N[1, 3] = M[0, 2] + M[2, 0]
    N[3, 1] = M[0, 2] + M[2, 0]
    N[2, 3] = M[1, 2] + M[2, 1]
    N[3, 2] = M[1, 2] + M[2, 1]

    eigen_vals, eigen_vecs = np.linalg.eig(N)
    print 'eigen values:', eigen_vals
    max_eigen_id = np.argmax(eigen_vals)
    eigen_vec = eigen_vecs[:, max_eigen_id]
    print 'eigen vector:', eigen_vec
    rotation = _Quaternion2Mat(eigen_vec)
    print 'rotation:', rotation

    ## Compute translation
    left_center = _CenterOfPoints(left_points)
    right_center = _CenterOfPoints(right_points)
    rot_left_center = rotation.dot(left_center)
    translation = right_center - scale * rot_left_center
    print 'translation', translation

    return scale, rotation, translation

def Quaternion2Mat(quat):
    """
    :param quat: 4
    :return: 3x3
    """
    quat = tf.squeeze(quat)
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    val00 = 1 - 2 * y * y - 2 * z * z
    val01 = 2 * x * y - 2 * z * w
    val02 = 2 * x * z + 2 * y * w
    val10 = 2 * x * y + 2 * z * w

    val11 = 1 - 2 * x * x - 2 * z * z
    val12 = 2 * y * z - 2 * x * w
    val20 = 2 * x * z - 2 * y * w
    val21 = 2 * y * z + 2 * x * w
    val22 = 1 - 2 * x * x - 2 * y * y
    rotation = tf.stack([val00, val01, val02, val10, val11, val12, val20, val21, val22], axis=0)
    rotation = tf.reshape(rotation, shape=[3,3])
    return rotation

def Mat2Quaternion(rotation):
    """
    :param rotation: 3x3
    :return: 4
    """
    w = tf.sqrt(1. + rotation[0, 0] + rotation[1, 1] + rotation[2, 2]) / 2.
    x = (rotation[2, 1] - rotation[1, 2]) / (4. * w)
    y = (rotation[0, 2] - rotation[2, 0]) / (4. * w)
    z = (rotation[1, 0] - rotation[0, 1]) / (4. * w)
    quat = tf.stack([w, x, y, z], axis=0)
    quat = tf.nn.l2_normalize(quat)
    return quat

def Quaternion2AngleAxis(quat):
    """
    :param quat: 4
    :return: 4
    """
    quat = tf.squeeze(quat)
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]
    angle = 2. * tf.acos(qw)
    x = qx / tf.sqrt(1 - qw * qw)
    y = qy / tf.sqrt(1 - qw * qw)
    z = qz / tf.sqrt(1 - qw * qw)
    angle_axis = tf.stack([angle, x, y, z], axis=0)
    return angle_axis

def Mat2AngleAxis(m):
    angle = tf.acos((m[0,0]+m[1,1]+m[2,2]-1.) / 2.)
    val = tf.sqrt(tf.square(m[2,1]-m[1,2]) + tf.square(m[0,2]-m[2,0]) + tf.square(m[1,0]-m[0,1]))
    x = (m[2,1] - m[1,2]) / val
    y = (m[0,2] - m[2,0]) / val
    z = (m[1,0] - m[0,1]) / val
    angle_axis = tf.stack([angle, x, y, z], axis=0)
    return angle_axis

def AngleBetweenQuat(quat1, quat2):
    rotation1 = Quaternion2Mat(quat1)
    rotation2 = Quaternion2Mat(quat2)
    rotation1 = tf.add(rotation1, 0.0, name='rotation1')
    rotation2 = tf.add(rotation2, 0.0, name='rotation2')
    diff_rotation = tf.matmul(rotation2, rotation1, transpose_b=True, name='diff_rotation')
    angle_axis = Mat2AngleAxis(diff_rotation)
    return angle_axis[0] / math.pi * 180.

def CenterOfPoints(points):
    center = tf.reduce_mean(points, axis=1) # 3
    return center

def UncenterPoints(points):
    point_num = points.get_shape().as_list()[1]
    center = CenterOfPoints(points)    # 2
    center = tf.expand_dims(center, axis=-1)    # 2x1
    center_tile = tf.tile(center, [1, point_num])
    u_points = points - center_tile
    return u_points

def ScaleFromPoints(left_points, right_points):
    """
    Compute relative scale from left points to right points
    :param left_points: 3xN
    :param right_points: 3xN
    :return:
    """
    lefts = UncenterPoints(left_points)  # 3xN
    rights = UncenterPoints(right_points)

    ## Compute scale
    left_norm_square = tf.reduce_sum(tf.square(tf.norm(lefts, axis=0)))
    right_norm_square = tf.reduce_sum(tf.square(tf.norm(rights, axis=0)))
    scale = tf.sqrt(right_norm_square / left_norm_square)
    return scale

def TransformFromPointsTF(left_points, right_points):
    """
    Tensorflow implementatin of aligning left points to right points
    :param left_points: 3xN
    :param right_points: 3xN
    :return:
    """

    lefts = UncenterPoints(left_points)    # 3xN
    rights = UncenterPoints(right_points)
    # lefts = left_points
    # rights = right_points

    ## Compute scale
    left_norm_square = tf.reduce_sum(tf.square(tf.norm(lefts, axis=0)))
    right_norm_square = tf.reduce_sum(tf.square(tf.norm(rights, axis=0)))
    scale = tf.sqrt(right_norm_square / (left_norm_square+1e-6))

    ## Compute rotation
    #rights = tf.Print(rights, [rights], message='rights', summarize=2 * 68)
    M = tf.matmul(lefts, rights, transpose_b=True)  # 3x3
    #M = tf.Print(M, [M.shape, M], message="M", summarize=64)

    N00 = M[0, 0] + M[1, 1] + M[2, 2]
    N11 = M[0, 0] - M[1, 1] - M[2, 2]
    N22 = -M[0, 0] + M[1, 1] - M[2, 2]
    N33 = -M[0, 0] - M[1, 1] + M[2, 2]

    N01 = M[1, 2] - M[2, 1]
    N10 = M[1, 2] - M[2, 1]
    N02 = M[2, 0] - M[0, 2]
    N20 = M[2, 0] - M[0, 2]

    N03 = M[0, 1] - M[1, 0]
    N30 = M[0, 1] - M[1, 0]
    N12 = M[0, 1] + M[1, 0]
    N21 = M[0, 1] + M[1, 0]

    N13 = M[0, 2] + M[2, 0]
    N31 = M[0, 2] + M[2, 0]
    N23 = M[1, 2] + M[2, 1]
    N32 = M[1, 2] + M[2, 1]
    N = tf.stack([N00,N01,N02,N03,N10,N11,N12,N13,N20,N21,N22,N23,N30,N31,N32,N33], axis=0)
    N = tf.reshape(N, [4,4])

    #N = tf.Print(N, [N.shape, N], message="N", summarize=64)

    eigen_vals, eigen_vecs = tf.self_adjoint_eig(N)
    quaternion = tf.squeeze((tf.slice(eigen_vecs, [0, 3], [4, 1])))    # 4
    #quaternion = tf_render.Print(quaternion, [quaternion], message='quaternion', summarize=4)
    rotation = Quaternion2Mat(quaternion)   # 3x3

    ## Compute translation
    left_center = CenterOfPoints(left_points)
    right_center = CenterOfPoints(right_points)
    rot_left_center = tf.squeeze(tf.matmul(rotation, tf.expand_dims(left_center, axis=-1))) # 3
    translation = right_center - scale * rot_left_center

    return scale, rotation, translation
