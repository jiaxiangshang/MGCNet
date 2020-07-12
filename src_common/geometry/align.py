#
import tensorflow as tf

# zhoulei
from geo_utils_zhoulei import TransformFromPointsTF, Quaternion2Mat


def lm2d_trans_zhoulei(lm_src, lm_tar):
    filler_mtx = tf.constant([0.0, 0.0, 1.0], shape=[1, 3])
    list_trans_mtx = []
    for b in range(lm_src.shape[0]):
        filler_z = tf.constant([0.0], shape=[1, 1])
        filler_z = tf.tile(filler_z, multiples=[lm_src.shape[1], 1])
        b_src = lm_src[b]
        b_src = tf.concat([b_src, filler_z], axis=1)
        b_src = tf.transpose(b_src)
        b_tar = lm_tar[b]
        b_tar = tf.concat([b_tar, filler_z], axis=1)
        b_tar = tf.transpose(b_tar)

        #b_src = tf.Print(b_src, [b_src], message='b_src', summarize=2 * 68)
        # b_tar = tf_render.Print(b_tar, [b_tar], message='b_tar', summarize=16)
        s, rot_mat, translation = TransformFromPointsTF(b_src, b_tar)

        # s = tf_render.Print(s, [s, s.shape], message='s', summarize=1)

        # rot_mat = tf_render.Print(rot_mat, [rot_mat], message='rot_mat', summarize=9)
        # translation = tf_render.Print(translation, [translation], message='translation', summarize=3)
        rot_mat = rot_mat[0:2, 0:2] * s
        translation = translation[0:2]
        translation = tf.expand_dims(translation, axis=-1)

        ext_mat = tf.concat([rot_mat, translation], axis=1)
        ext_mat = tf.concat([ext_mat, filler_mtx], axis=0)
        list_trans_mtx.append(ext_mat)

    trans_mtx = tf.stack(list_trans_mtx)
    return trans_mtx
